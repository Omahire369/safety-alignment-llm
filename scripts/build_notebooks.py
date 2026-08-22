#!/usr/bin/env python
"""Emit the four part-wise Kaggle notebooks from a single spec.

Keeping the notebooks generated (rather than hand-edited) means the code they
run is always the code in src/safealign, and there is no copy-paste drift.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"

SETUP = """\
# --- environment -----------------------------------------------------------
import os, sys, glob, shutil, zipfile
from pathlib import Path

os.environ["HF_HOME"] = "/kaggle/temp/hf"                  # dataset + model cache
os.environ["SAFEALIGN_ROOT"] = "/kaggle/temp/safealign"    # big artifacts, off the 20 GB quota
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

try:                                                        # Add-ons -> Secrets -> HF_TOKEN
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
except Exception as e:
    print("HF_TOKEN secret not found:", e)

!pip install -q -U "transformers>=4.44" "peft>=0.12" "datasets>=2.20" "accelerate>=0.33" \\
    rouge-score sacrebleu nltk mergekit

# --- locate the project inside the attached Kaggle Dataset -----------------
REPO = "/kaggle/working/safety-alignment-llm"

def find_source():
    # Kaggle auto-extracts uploaded archives, so the dataset may hold either
    # the unpacked folder or the original .zip. Handle both.
    hits = glob.glob("/kaggle/input/**/src/safealign/config.py", recursive=True)
    if hits:
        return ("dir", str(Path(hits[0]).parents[2]))
    zips = glob.glob("/kaggle/input/**/*.zip", recursive=True)
    if zips:
        return ("zip", zips[0])
    raise FileNotFoundError("Attach the dataset holding the project (Add Input -> Datasets)")

if not os.path.exists(REPO):
    kind, src = find_source()
    if kind == "zip":
        with zipfile.ZipFile(src) as z:
            z.extractall("/kaggle/working")
    else:
        shutil.copytree(src, REPO)                          # /kaggle/input is read-only
    print("project from", kind, src)

sys.path.insert(0, f"{REPO}/src")

import torch
print(torch.__version__, "| GPUs:", torch.cuda.device_count(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
from safealign.config import CFG
CFG.paths.ensure(); print("artifacts ->", CFG.paths.artifacts)
"""

NOTEBOOKS = {
    "Part_1.ipynb": [
        ("md", "# Part 1 — Supervised Fine-Tuning and DARE\n\n"
               "**1.1** LoRA instruction tuning of `Qwen/Qwen2.5-1.5B-Instruct` on the first "
               "60% of `medalpaca/medical_meadow_medqa`.\n\n"
               "**1.2** DARE (Drop And REscale) applied to the SFT delta, merged with "
               "`mergekit`, with a drop-rate sweep selected on the validation split."),
        ("code", SETUP),
        ("md", "## 1.1 Instruction tuning\n\nHyper-parameters are declared in "
               "`safealign/config.py` and written to `train_stats.json` next to the adapter, "
               "so the report can quote them without transcription errors."),
        ("code", "from safealign.config import CFG, MODEL_SFT\n"
                 "from safealign.sft import train_medical_sft\n"
                 "from safealign.model_utils import merge_lora\n\n"
                 "adapter = train_medical_sft()\n"
                 "merged = merge_lora(adapter, CFG.paths.artifacts / f'{MODEL_SFT}_merged')\n"
                 "print(open(adapter / 'train_stats.json').read())"),
        ("md", "## 1.2 DARE\n\n`density = 1 - p`. mergekit's `dare_linear` performs the "
               "Bernoulli drop and the `1/(1-p)` rescale; the reference implementation in "
               "`safealign/dare.py` is used to verify the arithmetic."),
        ("code", "from safealign.dare import delta_stats, sweep_drop_rates\n\n"
                 "print(delta_stats(str(merged)))\n"
                 "sweep = sweep_drop_rates(merged, n_val=150)\n"
                 "sweep['best_p'], sweep['sweep']"),
        ("code", "import pandas as pd\n"
                 "pd.DataFrame(sweep['sweep']).T.rename_axis('drop rate p')"),
    ],
    "Part_2.ipynb": [
        ("md", "# Part 2 — Parameter-Space Safety Vector (RESTA)\n\n"
               "**2.1** Build the unaligned model by LoRA fine-tuning the base model on "
               "(harmful prompt → harmful completion) pairs from `toxic-dpo-v0.2`.\n\n"
               "**2.2** `δ_safe = θ_base − θ_harmful`, added to the SFT and DARE models via "
               "mergekit `task_arithmetic`."),
        ("code", SETUP),
        ("code", "from safealign.config import CFG\n"
                 "from safealign.sft import train_harmful\n"
                 "from safealign.model_utils import merge_lora\n\n"
                 "harm_adapter = train_harmful()\n"
                 "harmful = merge_lora(harm_adapter, CFG.paths.artifacts / 'model_harmful_merged')"),
        ("md", "### Sanity check on the harmful model\n"
               "It should comply where the base model refuses — otherwise the safety vector "
               "is close to noise."),
        ("code", "from safealign.model_utils import load_model, load_tokenizer, free\n"
                 "from safealign.evaluation.generate import generate_batch\n"
                 "from safealign.data import load_toxic_dpo\n\n"
                 "probe = [[{'role':'user','content':q}] for q in load_toxic_dpo()['prompt'][:3]]\n"
                 "for mid in [CFG.model.base_model, str(harmful)]:\n"
                 "    m, t = load_model(mid), load_tokenizer(mid)\n"
                 "    print('###', mid)\n"
                 "    for o in generate_batch(m, t, probe, max_new_tokens=48): print(' -', o[:200])\n"
                 "    del m; free()"),
        ("md", "## 2.2 Safety vector"),
        ("code", "from safealign.resta import build_all, safety_vector_norm\n\n"
                 "print(safety_vector_norm(str(harmful))['l2_norm'])\n"
                 "build_all(harmful)"),
    ],
    "Part_3.ipynb": [
        ("md", "# Part 3 — Activation-Space Safety Vector (Function Vector)\n\n"
               "Causal Mediation Analysis over all layer×head pairs, top-10 heads by Average "
               "Indirect Effect, and injection of the resulting Function Vector into the "
               "residual stream at layer `floor(L/3)`."),
        ("code", SETUP),
        ("md", "## 3.1 Few-shot prompts and the sampling strategy\n\n"
               "541 toxic-dpo rows are shuffled once under a fixed seed and split into a "
               "15-row target pool and a 526-row in-context pool. The pools are disjoint, so "
               "no target query can appear among its own exemplars. Each prompt draws N=10 "
               "exemplars without replacement from the ICL pool under an independent seed."),
        ("code", "from safealign.fv.prompts import build_fewshot_pairs, refusal_token_ids\n"
                 "from safealign.model_utils import load_tokenizer\n\n"
                 "pairs = build_fewshot_pairs()\n"
                 "tok = load_tokenizer()\n"
                 "print('prompts:', len(pairs), 'ICL per prompt:', len(pairs[0].icl_indices))\n"
                 "print('V_refusal:', [tok.decode([i]) for i in refusal_token_ids(tok)])\n"
                 "print(pairs[0].clean_messages[1]['content'][:200])\n"
                 "print('CLEAN  ->', pairs[0].clean_messages[2]['content'][:160])\n"
                 "print('CORRUPT->', pairs[0].corrupted_messages[2]['content'][:160])"),
        ("md", "## Causal Indirect Effect\n\n"
               "Head activations are read **after** the per-head slice of `W_O`, so they live "
               "in residual-stream space. Heads are evaluated in batches of "
               "`head_batch_size`, one head per batch element, instead of one forward pass "
               "per head."),
        ("code", "from safealign.fv.cie import run_cma\n"
                 "cma = run_cma()\n"
                 "cma['top_heads']"),
        ("md", "## 3.2 Build and inject the Function Vector"),
        ("code", "from safealign.fv.vector import build_and_save, sweep_lambda\n"
                 "from safealign.config import CFG, MODEL_SFT\n\n"
                 "fv_meta = build_and_save(); print(fv_meta)\n"
                 "lam = sweep_lambda(str(CFG.paths.artifacts / f'{MODEL_SFT}_merged'))\n"
                 "lam['best_lambda']"),
        ("md", "## 3.3 Interpretability\n\n"
               "The AIE heatmap shows *where* in the network the safety computation happens; "
               "the logit lens shows whether the FV decodes to literal refusal words or to "
               "something more abstract."),
        ("code", "from safealign.fv.viz import build_all_figures\n"
                 "figs = build_all_figures(); figs['top_tokens']"),
        ("code", "from IPython.display import Image, display\n"
                 "display(Image(figs['aie_heatmap'])); display(Image(figs['logit_lens_fig']))"),
    ],
    "Part_4.ipynb": [
        ("md", "# Part 4 — Evaluation and Comparative Analysis\n\n"
               "Seven configurations scored for safety (Unsafe Score on 550 HarmEval prompts, "
               "LLM judge) and utility (ROUGE-L / METEOR / BLEU on the held-out 20% medical "
               "split)."),
        ("code", SETUP),
        ("md", "### Phase 1 — generation (one model in memory at a time)"),
        ("code", "from safealign.evaluation.run_all import build_configs, generate_phase\n"
                 "configs = build_configs()\n"
                 "generate_phase(configs)"),
        ("md", "### Phase 2 — judge and metrics\n"
               "The 7B judge is loaded once and scores every saved generation file. On a "
               "single T4 use `device_map='auto'`; with the 2×T4 accelerator it shards "
               "automatically."),
        ("code", "from safealign.evaluation.run_all import main as eval_main\n"
                 "summary = eval_main(phases='score')"),
        ("code", "import pandas as pd, json\n"
                 "from safealign.config import CFG\n"
                 "s = json.loads((CFG.paths.results / 'summary.json').read_text())\n"
                 "pd.DataFrame([{'config': v['label'], **v['safety'], **v['utility']}\n"
                 "              for v in s.values()]).set_index('config').round(4)"),
        ("md", "### Safety–utility trade-off"),
        ("code", "import matplotlib.pyplot as plt\n"
                 "fig, ax = plt.subplots(figsize=(7,5), dpi=140)\n"
                 "for name, v in s.items():\n"
                 "    if not v['safety'] or not v['utility']: continue\n"
                 "    ax.scatter(v['utility']['rougeL'], v['safety']['unsafe_score'], s=70)\n"
                 "    ax.annotate(v['label'], (v['utility']['rougeL'], v['safety']['unsafe_score']),\n"
                 "                textcoords='offset points', xytext=(6,4), fontsize=8)\n"
                 "ax.set_xlabel('ROUGE-L (utility, higher better)')\n"
                 "ax.set_ylabel('Unsafe Score (lower safer)')\n"
                 "ax.set_title('Safety-utility trade-off')\n"
                 "ax.grid(alpha=.3); fig.tight_layout()\n"
                 "fig.savefig(CFG.paths.figures / 'tradeoff.png')"),
    ],
}


def cell(kind: str, source: str) -> dict:
    lines = source.splitlines(keepends=True)
    if kind == "md":
        return {"cell_type": "markdown", "metadata": {}, "source": lines}
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": lines}


def build():
    NB_DIR.mkdir(parents=True, exist_ok=True)
    for name, cells in NOTEBOOKS.items():
        nb = {
            "cells": [cell(k, s) for k, s in cells],
            "metadata": {
                "kernelspec": {"display_name": "Python 3", "language": "python",
                               "name": "python3"},
                "language_info": {"name": "python", "version": "3.10"},
                "accelerator": "GPU",
            },
            "nbformat": 4, "nbformat_minor": 5,
        }
        (NB_DIR / name).write_text(json.dumps(nb, indent=1))
        print("wrote", NB_DIR / name)


if __name__ == "__main__":
    build()
