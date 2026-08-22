#!/usr/bin/env python
"""Package pipeline outputs for the public demo, and optionally push to the Hub.

    python scripts/bundle_demo.py                       # fill app/assets
    python scripts/bundle_demo.py --push-adapters USER  # upload LoRA adapters + FV
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from safealign.config import CFG, MODEL_HARMFUL, MODEL_SFT  # noqa: E402

ASSETS = ROOT / "app" / "assets"
COPY = [
    ("function_vector.pt", CFG.paths.results / "function_vector.pt"),
    ("summary.json", CFG.paths.results / "summary.json"),
    ("aie_heatmap.png", CFG.paths.figures / "aie_heatmap.png"),
    ("fv_logit_lens.png", CFG.paths.figures / "fv_logit_lens.png"),
    ("tradeoff.png", CFG.paths.figures / "tradeoff.png"),
]


def bundle_responses(n: int = 12) -> Path:
    """Interleave the seven configs' HarmEval answers into one browsable file."""
    summary_path = CFG.paths.results / "summary.json"
    labels = {}
    if summary_path.exists():
        labels = {k: v["label"] for k, v in json.loads(summary_path.read_text()).items()}

    files = sorted(CFG.paths.generations.glob("*__harmeval.json"))
    if not files:
        print("[bundle] no generations found, skipping responses_sample.json")
        return ASSETS / "responses_sample.json"

    blobs = {f.name.split("__")[0]: json.loads(f.read_text()) for f in files}
    any_blob = next(iter(blobs.values()))
    rows = []
    step = max(len(any_blob["prompts"]) // n, 1)
    for i in range(0, len(any_blob["prompts"]), step):
        if len(rows) >= n:
            break
        rows.append({
            "prompt": any_blob["prompts"][i],
            "responses": {labels.get(name, name): b["responses"][i][:900]
                          for name, b in blobs.items() if i < len(b["responses"])},
        })
    out = ASSETS / "responses_sample.json"
    out.write_text(json.dumps({"rows": rows}, indent=2))
    print(f"[bundle] {out} ({len(rows)} prompts)")
    return out


def bundle() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    for name, src in COPY:
        if Path(src).exists():
            shutil.copy(src, ASSETS / name)
            print(f"[bundle] {name}")
        else:
            print(f"[bundle] missing (skipped): {src}")
    bundle_responses()


def push(user: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    for local, repo in [
        (CFG.paths.artifacts / MODEL_SFT, f"{user}/qwen2.5-1.5b-medqa-lora"),
        (CFG.paths.artifacts / MODEL_HARMFUL, f"{user}/qwen2.5-1.5b-harmful-lora"),
    ]:
        if not Path(local).exists():
            print(f"[push] missing {local}, skipping")
            continue
        api.create_repo(repo, exist_ok=True)
        api.upload_folder(folder_path=str(local), repo_id=repo,
                          ignore_patterns=["checkpoints/*"])
        print(f"[push] {repo}")

    fv = CFG.paths.results / "function_vector.pt"
    if fv.exists():
        repo = f"{user}/qwen2.5-1.5b-refusal-function-vector"
        api.create_repo(repo, exist_ok=True)
        api.upload_file(path_or_fileobj=str(fv), path_in_repo="function_vector.pt",
                        repo_id=repo)
        print(f"[push] {repo}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--push-adapters", metavar="HF_USERNAME", default=None)
    a = ap.parse_args()
    bundle()
    if a.push_adapters:
        push(a.push_adapters)
