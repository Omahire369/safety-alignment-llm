"""Part 1.2 — DARE (Drop And REscale) applied to the SFT delta.

delta            = theta_sft - theta_base
drop             = Bernoulli(p) mask zeroing a fraction p of delta
rescale          = delta_kept / (1 - p)      (preserves the expected delta)
merge            = theta_base + rescaled delta

mergekit's `dare_linear` method does exactly this, parameterised by
`density = 1 - p`. The reference implementation below is used to verify that
the mergekit output matches the paper's maths, and as an offline fallback.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

import torch
import yaml

from .config import CFG, MODEL_DARE, MODEL_SFT
from .model_utils import (free, is_mergeable, load_state_dict, save_state_dict,
                          write_meta)


# --------------------------------------------------------------------------- #
# mergekit path (the one used in the pipeline)
# --------------------------------------------------------------------------- #
def dare_yaml(sft_path: str | Path, p: float, base_model: Optional[str] = None) -> dict:
    return {
        "models": [{
            "model": str(sft_path),
            "parameters": {"weight": 1.0, "density": round(1.0 - p, 4)},
        }],
        "merge_method": "dare_linear",
        "base_model": base_model or CFG.model.base_model,
        "dtype": "float16",
        "parameters": {"int8_mask": True},
    }


def run_mergekit(config: dict, out_dir: str | Path, cuda: bool = True) -> Path:
    out_dir = Path(out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    cfg_path = out_dir.parent / f"{out_dir.name}.mergekit.yaml"
    cfg_path.write_text(yaml.safe_dump(config, sort_keys=False))

    cmd = ["mergekit-yaml", str(cfg_path), str(out_dir),
           "--copy-tokenizer", "--allow-crimes", "--out-shard-size", "1B",
           "--lazy-unpickle"]
    if cuda and torch.cuda.is_available():
        cmd.append("--cuda")
    print("[mergekit]", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return out_dir


def apply_dare(sft_path: str | Path, p: float, out_dir: str | Path,
               use_mergekit: bool = True) -> Path:
    out_dir = Path(out_dir)
    if (out_dir / "config.json").exists():
        print(f"[dare] {out_dir} exists, skipping")
        return out_dir
    if use_mergekit:
        run_mergekit(dare_yaml(sft_path, p), out_dir)
    else:
        sd = dare_state_dict(str(sft_path), p=p, seed=CFG.dare.seed)
        save_state_dict(sd, out_dir)
    write_meta(out_dir, {"method": "dare_linear", "drop_rate_p": p,
                         "density": 1 - p, "source": str(sft_path)})
    return out_dir


# --------------------------------------------------------------------------- #
# Reference implementation (auditable maths)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def dare_state_dict(sft_path: str, p: float, seed: int = 42,
                    base_path: Optional[str] = None) -> Dict[str, torch.Tensor]:
    base_sd = load_state_dict(base_path or CFG.model.base_model)
    sft_sd = load_state_dict(sft_path)
    gen = torch.Generator().manual_seed(seed)
    merged: Dict[str, torch.Tensor] = {}
    for k, v_sft in sft_sd.items():
        v_base = base_sd.get(k)
        if v_base is None or v_base.shape != v_sft.shape or not is_mergeable(v_sft):
            merged[k] = v_sft
            continue
        delta = v_sft - v_base
        mask = (torch.rand(delta.shape, generator=gen) >= p).to(delta.dtype)
        merged[k] = v_base + mask * delta / (1.0 - p)
    del base_sd, sft_sd
    free()
    return merged


@torch.no_grad()
def delta_stats(sft_path: str, base_path: Optional[str] = None) -> dict:
    """Sanity numbers for the report: how large / how sparse is the delta."""
    base_sd = load_state_dict(base_path or CFG.model.base_model)
    sft_sd = load_state_dict(sft_path)
    total, nonzero, l2 = 0, 0, 0.0
    for k, v in sft_sd.items():
        b = base_sd.get(k)
        if b is None or b.shape != v.shape or not is_mergeable(v):
            continue
        d = v - b
        total += d.numel()
        nonzero += int((d.abs() > 1e-8).sum())
        l2 += float(d.pow(2).sum())
    del base_sd, sft_sd
    free()
    return {"delta_params": total, "delta_nonzero": nonzero,
            "delta_l2_norm": l2 ** 0.5}


# --------------------------------------------------------------------------- #
# Drop-rate sweep on the validation split
# --------------------------------------------------------------------------- #
def sweep_drop_rates(sft_path: str | Path, drop_rates: Optional[List[float]] = None,
                     n_val: int = 150, keep_all: bool = False) -> dict:
    """Build one DARE model per p, score utility on the validation split,
    and promote the best one to `model_sft_dare`."""
    from .evaluation.utility import score_utility_for_model

    drop_rates = drop_rates or CFG.dare.drop_rates
    results = {}
    for p in drop_rates:
        cand = CFG.paths.artifacts / f"dare_p{p}"
        apply_dare(sft_path, p, cand)
        metrics = score_utility_for_model(str(cand), split="validation", n=n_val)
        results[str(p)] = metrics
        print(f"[dare-sweep] p={p} -> {metrics}")
        free()

    best_p = max(results, key=lambda k: results[k]["rougeL"])
    best_dir = CFG.paths.artifacts / f"dare_p{best_p}"
    final = CFG.paths.artifacts / MODEL_DARE
    if final.exists():
        shutil.rmtree(final)
    shutil.copytree(best_dir, final)
    write_meta(final, {"method": "dare_linear", "drop_rate_p": float(best_p),
                       "selected_by": "validation rougeL", "sweep": results})

    if not keep_all:
        for p in drop_rates:
            d = CFG.paths.artifacts / f"dare_p{p}"
            if d.exists() and d != best_dir:
                shutil.rmtree(d)

    out = {"sweep": results, "best_p": float(best_p), "model": str(final)}
    (CFG.paths.results / "dare_sweep.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", default=str(CFG.paths.artifacts / f"{MODEL_SFT}_merged"))
    ap.add_argument("--n-val", type=int, default=150)
    a = ap.parse_args()
    print(json.dumps(sweep_drop_rates(a.sft, n_val=a.n_val), indent=2))
