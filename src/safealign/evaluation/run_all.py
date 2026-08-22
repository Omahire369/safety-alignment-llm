"""Part 4 — evaluate all seven configurations and emit the comparison tables.

Configurations
  1. base                 theta_base
  2. sft                  model_sft_lora
  3. sft_dare             model_sft_dare
  4. sft_resta            model_sft_resta
  5. sft_dare_resta       model_sft_dare_resta
  6. sft_fv               model_sft_lora      + activation-space FV
  7. sft_dare_fv          model_sft_dare      + activation-space FV

Generation is done first (one model in memory at a time), then the 7B judge is
loaded exactly once and scores every saved generation file. This ordering is
what makes the whole evaluation fit on a 16 GB T4.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch

from ..config import (CFG, MODEL_DARE, MODEL_DARE_RESTA, MODEL_RESTA, MODEL_SFT)
from ..data import harmeval_to_chat, load_harmeval
from ..model_utils import free, load_model, load_tokenizer
from .generate import generate_with_fv
from .utility import compute_metrics, utility_examples


@dataclass
class RunConfig:
    name: str
    model_path: str
    use_fv: bool = False
    label: str = ""


def build_configs(artifacts: Optional[Path] = None) -> List[RunConfig]:
    a = artifacts or CFG.paths.artifacts
    return [
        RunConfig("base", CFG.model.base_model, False, "Base θ_base"),
        RunConfig("sft", str(a / f"{MODEL_SFT}_merged"), False, "SFT"),
        RunConfig("sft_dare", str(a / MODEL_DARE), False, "SFT + DARE"),
        RunConfig("sft_resta", str(a / MODEL_RESTA), False, "SFT + RESTA"),
        RunConfig("sft_dare_resta", str(a / MODEL_DARE_RESTA), False, "SFT + DARE + RESTA"),
        RunConfig("sft_fv", str(a / f"{MODEL_SFT}_merged"), True, "SFT + FV"),
        RunConfig("sft_dare_fv", str(a / MODEL_DARE), True, "SFT + DARE + FV"),
    ]


def _load_fv():
    from ..fv.vector import load_fv
    fv, meta = load_fv()
    return fv, meta["layer"], CFG.fv.chosen_lambda


def generate_phase(configs: List[RunConfig], n_harm: Optional[int] = None,
                   n_util: Optional[int] = None, skip_existing: bool = True) -> None:
    harm_prompts = load_harmeval()[: n_harm or None]
    harm_chats = [harmeval_to_chat(p) for p in harm_prompts]
    util_chats, util_refs = utility_examples("test", n_util)

    fv = layer = lam = None
    if any(c.use_fv for c in configs):
        fv, layer, lam = _load_fv()

    for cfg in configs:
        harm_file = CFG.paths.generations / f"{cfg.name}__harmeval.json"
        util_file = CFG.paths.generations / f"{cfg.name}__utility.json"
        if skip_existing and harm_file.exists() and util_file.exists():
            print(f"[gen] {cfg.name}: cached, skipping")
            continue

        print(f"\n=== generating: {cfg.name} ({cfg.model_path}) fv={cfg.use_fv} ===")
        tok = load_tokenizer(cfg.model_path)
        model = load_model(cfg.model_path)
        f, l, m = (fv, layer, lam) if cfg.use_fv else (None, None, None)

        harm_out = generate_with_fv(model, tok, harm_chats, f, l, m, show_progress=True)
        harm_file.write_text(json.dumps({
            "config": cfg.name, "model": cfg.model_path, "use_fv": cfg.use_fv,
            "lambda": m, "prompts": harm_prompts, "responses": harm_out}, indent=2))

        util_out = generate_with_fv(model, tok, util_chats, f, l, m, show_progress=True)
        util_file.write_text(json.dumps({
            "config": cfg.name, "model": cfg.model_path, "use_fv": cfg.use_fv,
            "prompts": [c[-1]["content"] for c in util_chats],
            "references": util_refs, "responses": util_out}, indent=2))

        del model
        free()


def judge_phase(configs: List[RunConfig]) -> Dict[str, dict]:
    from .safety import Judge, score_generations

    judge = Judge()
    out = {}
    for cfg in configs:
        f = CFG.paths.generations / f"{cfg.name}__harmeval.json"
        if not f.exists():
            continue
        out[cfg.name] = score_generations(f, judge)
        print(f"[safety] {cfg.name}: unsafe_score={out[cfg.name]['unsafe_score']:.4f}")
    judge.close()
    return out


def utility_phase(configs: List[RunConfig]) -> Dict[str, dict]:
    out = {}
    for cfg in configs:
        f = CFG.paths.generations / f"{cfg.name}__utility.json"
        if not f.exists():
            continue
        blob = json.loads(f.read_text())
        m = compute_metrics(blob["responses"], blob["references"])
        blob["utility"] = m
        f.write_text(json.dumps(blob, indent=2))
        out[cfg.name] = m
        print(f"[utility] {cfg.name}: {m}")
    return out


def render_markdown(summary: dict) -> str:
    header = ("| Configuration | Unsafe Score ↓ | Refusal rate | ROUGE-L ↑ | METEOR ↑ | BLEU ↑ |\n"
              "|---|---|---|---|---|---|\n")
    rows = []
    for cfg in build_configs():
        s = summary.get(cfg.name, {})
        saf, ut = s.get("safety", {}), s.get("utility", {})
        rows.append(
            f"| {cfg.label} | {saf.get('unsafe_score', float('nan')):.4f} "
            f"| {saf.get('heuristic_refusal_rate', float('nan')):.3f} "
            f"| {ut.get('rougeL', float('nan')):.4f} "
            f"| {ut.get('meteor', float('nan')):.4f} "
            f"| {ut.get('bleu', float('nan')):.2f} |")
    return header + "\n".join(rows) + "\n"


def main(n_harm: Optional[int] = None, n_util: Optional[int] = None,
         phases: str = "all") -> dict:
    configs = build_configs()
    if phases in ("all", "generate"):
        generate_phase(configs, n_harm, n_util)

    summary: dict = {}
    if phases in ("all", "score"):
        safety = judge_phase(configs)
        utility = utility_phase(configs)
        for c in configs:
            summary[c.name] = {"label": c.label, "model": c.model_path,
                               "safety": safety.get(c.name, {}),
                               "utility": utility.get(c.name, {})}
        (CFG.paths.results / "summary.json").write_text(json.dumps(summary, indent=2))
        (CFG.paths.results / "RESULTS.md").write_text(
            "# Results\n\n" + render_markdown(summary))
        print("\n" + render_markdown(summary))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-harm", type=int, default=None, help="limit HarmEval prompts")
    ap.add_argument("--n-util", type=int, default=None, help="limit utility prompts")
    ap.add_argument("--phases", choices=["all", "generate", "score"], default="all")
    a = ap.parse_args()
    main(a.n_harm, a.n_util, a.phases)
