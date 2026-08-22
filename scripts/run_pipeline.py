#!/usr/bin/env python
"""End-to-end driver: Part 1 -> Part 2 -> Part 3 -> Part 4.

Every stage is idempotent and checkpointed to disk, so a Kaggle session that
dies at hour 9 can be resumed by simply re-running the same command.

    python scripts/run_pipeline.py --stages all
    python scripts/run_pipeline.py --stages fv,eval      # resume later stages
    python scripts/run_pipeline.py --smoke               # 20-minute sanity run
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from safealign.config import (CFG, MODEL_DARE, MODEL_HARMFUL,  # noqa: E402
                              MODEL_SFT)
from safealign.model_utils import merge_lora  # noqa: E402

STAGES = ["sft", "dare", "harmful", "resta", "fv", "eval"]


def stamp(msg: str):
    print(f"\n{'=' * 72}\n[{time.strftime('%H:%M:%S')}] {msg}\n{'=' * 72}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="all",
                    help=f"comma-separated subset of {STAGES} or 'all'")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny subsets everywhere, for validating the plumbing")
    args = ap.parse_args()

    stages = STAGES if args.stages == "all" else args.stages.split(",")
    art = CFG.paths.artifacts
    limit = 200 if args.smoke else None
    manifest = {}

    if "sft" in stages:
        stamp("Part 1.1 — LoRA SFT on medical_meadow_medqa")
        from safealign.sft import train_medical_sft
        adapter = train_medical_sft(limit)
        merge_lora(adapter, art / f"{MODEL_SFT}_merged")
        manifest["sft_adapter"] = str(adapter)

    if "dare" in stages:
        stamp("Part 1.2 — DARE sparsify + merge, drop-rate sweep")
        from safealign.dare import sweep_drop_rates
        manifest["dare"] = sweep_drop_rates(art / f"{MODEL_SFT}_merged",
                                            n_val=30 if args.smoke else 150)

    if "harmful" in stages:
        stamp("Part 2.1 — harmful (unaligned) model on toxic-dpo-v0.2")
        from safealign.sft import train_harmful
        adapter = train_harmful(limit)
        merge_lora(adapter, art / "model_harmful_merged")
        manifest["harmful_adapter"] = str(adapter)

    if "resta" in stages:
        stamp("Part 2.2 — safety vector + RESTA additions")
        from safealign.resta import build_all, safety_vector_norm
        manifest["resta"] = build_all(art / "model_harmful_merged")
        manifest["safety_vector"] = safety_vector_norm(str(art / "model_harmful_merged"))

    if "fv" in stages:
        stamp("Part 3 — causal mediation, Function Vector, figures")
        from safealign.fv.cie import run_cma
        from safealign.fv.vector import build_and_save, sweep_lambda
        from safealign.fv.viz import build_all_figures
        manifest["cma"] = run_cma()
        manifest["fv"] = build_and_save()
        manifest["fv_lambda"] = sweep_lambda(
            str(art / f"{MODEL_SFT}_merged"), n_val=10 if args.smoke else 40)
        CFG.fv.chosen_lambda = manifest["fv_lambda"]["best_lambda"]
        manifest["figures"] = build_all_figures()

    if "eval" in stages:
        stamp("Part 4 — safety + utility across all seven configurations")
        from safealign.evaluation.run_all import main as eval_main
        manifest["evaluation"] = eval_main(
            n_harm=20 if args.smoke else None,
            n_util=20 if args.smoke else CFG.eval.n_utility)

    (CFG.paths.results / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    stamp(f"done — artifacts in {art}, results in {CFG.paths.results}")


if __name__ == "__main__":
    main()
