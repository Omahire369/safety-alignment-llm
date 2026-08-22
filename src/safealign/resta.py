"""Part 2 — RESTA: parameter-space safety vector.

The unaligned model theta_harmful is obtained by fine-tuning the base model on
(harmful prompt -> harmful completion) pairs. The safety vector is the negation
of that harmful direction:

    delta_safe = theta_base - theta_harmful

and it is added back to any downstream fine-tuned model:

    theta_resta = theta_sft  + lambda * delta_safe
                = theta_base + (theta_sft - theta_base) - lambda*(theta_harmful - theta_base)

The second line is exactly mergekit's `task_arithmetic` with weights
(+1 on the SFT task vector, -lambda on the harmful task vector), which is how
the pipeline computes it.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import torch

from .config import CFG, MODEL_DARE, MODEL_DARE_RESTA, MODEL_RESTA, MODEL_SFT
from .dare import run_mergekit
from .model_utils import (free, is_mergeable, load_state_dict, save_state_dict,
                          write_meta)


def resta_yaml(target_path: str | Path, harmful_path: str | Path,
               lam: float = 1.0, base_model: Optional[str] = None) -> dict:
    return {
        "models": [
            {"model": str(target_path), "parameters": {"weight": 1.0}},
            {"model": str(harmful_path), "parameters": {"weight": -float(lam)}},
        ],
        "merge_method": "task_arithmetic",
        "base_model": base_model or CFG.model.base_model,
        "dtype": "float16",
        "parameters": {"normalize": False, "int8_mask": True},
    }


def add_safety_vector(target_path: str | Path, harmful_path: str | Path,
                      out_dir: str | Path, lam: Optional[float] = None,
                      use_mergekit: bool = True) -> Path:
    lam = CFG.resta.lam if lam is None else lam
    out_dir = Path(out_dir)
    if (out_dir / "config.json").exists():
        print(f"[resta] {out_dir} exists, skipping")
        return out_dir
    if use_mergekit:
        run_mergekit(resta_yaml(target_path, harmful_path, lam), out_dir)
    else:
        sd = resta_state_dict(str(target_path), str(harmful_path), lam)
        save_state_dict(sd, out_dir)
    write_meta(out_dir, {"method": "resta/task_arithmetic", "lambda": lam,
                         "target": str(target_path), "harmful": str(harmful_path)})
    return out_dir


@torch.no_grad()
def resta_state_dict(target_path: str, harmful_path: str, lam: float = 1.0,
                     base_path: Optional[str] = None) -> Dict[str, torch.Tensor]:
    base_sd = load_state_dict(base_path or CFG.model.base_model)
    tgt_sd = load_state_dict(target_path)
    harm_sd = load_state_dict(harmful_path)
    out: Dict[str, torch.Tensor] = {}
    for k, v in tgt_sd.items():
        b, h = base_sd.get(k), harm_sd.get(k)
        if b is None or h is None or b.shape != v.shape or not is_mergeable(v):
            out[k] = v
            continue
        delta_safe = b - h                       # the safety vector
        out[k] = v + lam * delta_safe
    del base_sd, tgt_sd, harm_sd
    free()
    return out


@torch.no_grad()
def safety_vector_norm(harmful_path: str, base_path: Optional[str] = None) -> dict:
    base_sd = load_state_dict(base_path or CFG.model.base_model)
    harm_sd = load_state_dict(harmful_path)
    per_layer, total = {}, 0.0
    for k, v in harm_sd.items():
        b = base_sd.get(k)
        if b is None or b.shape != v.shape or not is_mergeable(v):
            continue
        n = float((b - v).pow(2).sum())
        total += n
        per_layer[k] = n ** 0.5
    del base_sd, harm_sd
    free()
    top = dict(sorted(per_layer.items(), key=lambda kv: -kv[1])[:20])
    return {"l2_norm": total ** 0.5, "top20_modules_by_norm": top}


def build_all(harmful_merged: str | Path) -> dict:
    """Produce model_sft_resta and model_sft_dare_resta."""
    art = CFG.paths.artifacts
    out = {}
    out["model_sft_resta"] = str(add_safety_vector(
        art / f"{MODEL_SFT}_merged", harmful_merged, art / MODEL_RESTA))
    out["model_sft_dare_resta"] = str(add_safety_vector(
        art / MODEL_DARE, harmful_merged, art / MODEL_DARE_RESTA))
    (CFG.paths.results / "resta_build.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--harmful", default=str(CFG.paths.artifacts / "model_harmful_merged"))
    a = ap.parse_args()
    print(json.dumps(build_all(a.harmful), indent=2))
