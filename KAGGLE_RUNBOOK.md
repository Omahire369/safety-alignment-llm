# Kaggle Runbook

Exact steps to run the whole project on Kaggle's free tier. Total GPU time is roughly
**13–16 hours**, which fits inside one week's quota (~30 h/week, 12 h max per session, and
only 1 interactive GPU session at a time). It is split across four sessions because nothing
here needs to run in one go.

---

## 0. One-time setup (15 minutes, no GPU)

1. **Verify your phone on Kaggle** — Settings → Phone Verification. Without it you get no
   GPU and no internet in notebooks, and every step below needs both.
2. **Create a Hugging Face account** and a token at
   huggingface.co/settings/tokens with **write** access (write is needed to park adapters
   between sessions).
3. **Add the token to Kaggle** — Notebook editor → Add-ons → Secrets → New secret,
   label it exactly `HF_TOKEN`.
4. **Upload the project as a Kaggle Dataset** — Kaggle → Datasets → New Dataset → upload
   `safety-alignment-llm.zip`, title it `safety-alignment-llm`, Private is fine, Create.
   Kaggle usually auto-extracts archives, so the dataset may end up holding the unpacked
   folder instead of the `.zip`; the setup cell below handles either.
5. **Accept dataset terms** while logged into HF, for `unalignment/toxic-dpo-v0.2` and
   `SoftMINER-Group/HarmEval`, if either shows a gate. Downloads fail with a 403 otherwise.

All three datasets are pulled straight from the Hub by `load_dataset` at runtime — nothing
to download or upload by hand. They land in `HF_HOME` (`/kaggle/temp/hf`), total well under
1 GB, and re-download once per session in a couple of minutes.

## Notebook settings (every session)

Right-hand panel:

| Setting | Value |
|---|---|
| Accelerator | **GPU T4 ×2** for Session D, `T4 ×2` or `P100` for A–C |
| Internet | **On** |
| Persistence | Files only |
| Environment | Latest available |

---

## Cell 1 — environment (paste this at the top of every session)

```python
import os
os.environ["HF_HOME"] = "/kaggle/temp/hf"                 # keep 15 GB of weights off the
os.environ["SAFEALIGN_ROOT"] = "/kaggle/temp/safealign"   # 20 GB /kaggle/working quota
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from kaggle_secrets import UserSecretsClient
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")

# Pinned to exact versions verified end-to-end on Kaggle T4 through Session B
# (2026-08-22). Do NOT use -U here — an unpinned install broke three separate
# things across Sessions A/B (torchao, TrainingArguments.warmup_ratio,
# apply_chat_template's return shape). Bump these deliberately and re-run the
# smoke test before trusting a new pin.
!pip install -q \
    "transformers==5.15.1" "peft==0.20.0" "datasets==5.0.1" "accelerate==1.14.0" \
    "torchao==0.16.0" rouge-score sacrebleu nltk

import torch, sys
print(torch.__version__, torch.cuda.device_count(), torch.cuda.get_device_name(0))
```

`mergekit` is deliberately not installed — it's unused. `use_mergekit` defaults to `False` in `dare.py`/`resta.py` (see changelog below); the reference state-dict implementations are what the pipeline actually runs, and they're covered by `tests/test_math.py`.

## Cell 2 — get the code from the attached dataset

In the notebook editor: **Add Input → Datasets → Your Datasets → safety-alignment-llm → Add**.
Then:

```python
import glob, shutil, zipfile
from pathlib import Path

REPO = "/kaggle/working/safety-alignment-llm"

def find_source():
    hits = glob.glob("/kaggle/input/**/src/safealign/config.py", recursive=True)
    if hits:
        return ("dir", str(Path(hits[0]).parents[2]))
    zips = glob.glob("/kaggle/input/**/*.zip", recursive=True)
    if zips:
        return ("zip", zips[0])
    raise FileNotFoundError("Attach the dataset first (Add Input -> Datasets)")

if not os.path.exists(REPO):
    kind, src = find_source()
    if kind == "zip":
        with zipfile.ZipFile(src) as z:
            z.extractall("/kaggle/working")
    else:
        shutil.copytree(src, REPO)      # /kaggle/input is read-only, so copy it out
    print("project from", kind, src)

import sys; sys.path.insert(0, f"{REPO}/src")
```

`/kaggle/input` is mounted read-only, which is why the folder is copied into
`/kaggle/working` — that also lets you edit files in place while iterating.

**When you change the code:** open the dataset page → New Version → upload the new zip, then
in the notebook click the refresh icon next to the input, and delete the stale copy with
`!rm -rf {REPO}` before re-running the cell.

## Cell 3 — smoke test first (~20 min, do this once)

Run it into a **throwaway root** so the 200-example toy adapter never gets mistaken for the
real one:

```python
!cd {REPO} && SAFEALIGN_ROOT=/kaggle/temp/smoke python scripts/run_pipeline.py --smoke
```

If that finishes, every dataset download, hook, merge and metric works. Only then start
Session A.

---

# Session A — Part 1: SFT + DARE (~3 h)

```python
!cd {REPO} && python scripts/run_pipeline.py --stages sft,dare
```

What happens: LoRA fine-tune on the 60% medical split (~1–1.5 h), merge to
`model_sft_lora_merged`, then build one DARE model per drop rate p ∈ {0.1, 0.3, 0.5, 0.7},
score each on 150 validation questions, keep the winner as `model_sft_dare` and delete the
rest (~1.5 h).

**Before the session ends, park the adapter on HF** — `/kaggle/temp` is wiped on exit:

```python
from huggingface_hub import HfApi
api = HfApi()
api.create_repo("OmAhire369/qwen2.5-1.5b-medqa-lora", exist_ok=True)
api.upload_folder(
    folder_path="/kaggle/temp/safealign/artifacts/model_sft_lora",
    repo_id="OmAhire369/qwen2.5-1.5b-medqa-lora",
    ignore_patterns=["checkpoints/*"],
)
!cp -r /kaggle/temp/safealign/results /kaggle/working/results_partA
```

Use `HfApi().upload_folder`, not `!hf upload` / `!huggingface-cli upload` — the CLI form
prompts interactively about self-updating (`huggingface_hub` new-version nag) and hangs
forever in a headless `!` cell with no stdin. The Python API has no such prompt.

Note the winning `p` from `results/dare_sweep.json` — you rebuild the DARE model from it
later in seconds. Only the ~70 MB adapters need to survive; every merged model is a
deterministic function of them.

---

# Session B — Part 2: harmful model + RESTA (~1.5 h)

Restore, then run:

```python
from huggingface_hub import snapshot_download
ART = "/kaggle/temp/safealign/artifacts"
snapshot_download("OmAhire369/qwen2.5-1.5b-medqa-lora", local_dir=f"{ART}/model_sft_lora")

from safealign.model_utils import merge_lora
from safealign.dare import apply_dare
merge_lora(f"{ART}/model_sft_lora", f"{ART}/model_sft_lora_merged")
apply_dare(f"{ART}/model_sft_lora_merged", p=0.7, out_dir=f"{ART}/model_sft_dare")  # 0.7 won the Session A sweep
```

```python
!cd {REPO} && python scripts/run_pipeline.py --stages harmful,resta
```

Trains the unaligned model on toxic-dpo (541 rows × 3 epochs, ~11 min on T4), then produces
`model_sft_resta` and `model_sft_dare_resta` by weight-space arithmetic (~3 min).

Sanity-check before moving on: the harmful model should comply where the base model refuses
(cell 4 of `notebooks/Part_2.ipynb`). If it still refuses everything, the safety vector is
mostly noise — raise `HARMFUL_EPOCHS` to 5 and retrain. Also worth a look either way:
`safety_vector_norm()`'s per-module breakdown, saved to `results/resta_build.json` — expect
it concentrated in `mlp.gate_proj`/`mlp.up_proj` in the back third of the layers, which is
worth contrasting against Part 3's attention-head localization in the report.

Park the harmful adapter too:

```python
api.create_repo("OmAhire369/qwen2.5-1.5b-harmful-lora", exist_ok=True)
api.upload_folder(
    folder_path=f"{ART}/model_harmful_lora",
    repo_id="OmAhire369/qwen2.5-1.5b-harmful-lora",
    ignore_patterns=["checkpoints/*"],
)
```

---

# Session C — Part 3: Function Vector (~2 h)

CMA runs on the **base** model, so you only need `model_sft_lora_merged` rebuilt for the λ
sweep:

```python
snapshot_download("OmAhire369/qwen2.5-1.5b-medqa-lora", local_dir=f"{ART}/model_sft_lora")
from safealign.model_utils import merge_lora
merge_lora(f"{ART}/model_sft_lora", f"{ART}/model_sft_lora_merged")
```

```python
!cd {REPO} && python scripts/run_pipeline.py --stages fv
```

Produces `results/aie_matrix.npy`, `function_vector.pt`, the AIE heatmap, the logit-lens
figure, and `fv_lambda_sweep.json`. The CIE loop is the expensive part: 336 heads × 15
prompts, batched 8 heads per forward pass (~30–45 min).

Save all of it — these files are small and you need them in Session D and in the demo:

```python
!cp -r /kaggle/temp/safealign/results /kaggle/working/results_partC
```

Then **Save Version → Save & Run All (Commit)** so `/kaggle/working` becomes a reusable
notebook output you can attach as an input to the next notebook.

---

# Session D — Part 4: evaluation (~6–8 h, use T4 ×2)

Restore both adapters, rebuild all five merged models, and copy `function_vector.pt` back to
`results/`. Then run generation and judging **as two separate phases**, deleting the merged
models in between so the 7B judge has disk and memory to itself:

```python
!cd {REPO} && MAX_NEW_TOKENS=128 N_UTILITY=200 GEN_BATCH=16 \
    python -m safealign.evaluation.run_all --phases generate
```

```python
!rm -rf /kaggle/temp/safealign/artifacts/model_sft_dare* \
        /kaggle/temp/safealign/artifacts/model_sft_resta \
        /kaggle/temp/safealign/artifacts/model_sft_lora_merged
!cd {REPO} && python -m safealign.evaluation.run_all --phases score
```

Generation is 7 configs × (550 HarmEval + 200 utility) prompts. At `MAX_NEW_TOKENS=128` that
is roughly 40 min per config. Judging is a single forward pass per response, so all 3,850
verdicts take ~30 min once the judge is loaded.

Outputs: `results/summary.json`, `results/RESULTS.md`, and per-config generation files with
every judgement attached.

```python
!cp -r /kaggle/temp/safealign/results /kaggle/working/results_final
```

---

## After the run

```python
!cd {REPO} && python scripts/bundle_demo.py --push-adapters OmAhire369
```

Copies the metrics, figures, function vector and sample responses into `app/assets/`, then
pushes the adapters and FV to the Hub. Paste `results/RESULTS.md` into the README table, fill
in `results/REPORT_TEMPLATE.md`, and push `app/` to a Hugging Face Space.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| CUDA OOM during SFT | `MAX_SEQ_LEN=768 BATCH_SIZE=1 GRAD_ACCUM=16` |
| CUDA OOM during CIE | `FV_HEAD_BATCH=4` (or 2) |
| CUDA OOM loading the judge | Use T4 ×2 so `device_map="auto"` shards it; on a single card fall back to `JUDGE_MODEL=Qwen/Qwen2.5-3B-Instruct` and say so in the report |
| `No space left on device` | `rm -rf /kaggle/temp/hf/hub/models--*` for models you're done with; merged checkpoints are 3.1 GB each |
| `ImportError: incompatible torchao` from `merge_lora`/`apply_dare` | Fixed by pinning `torchao==0.16.0` in Cell 1 (see changelog) — `peft` checks this version even when torchao quantization isn't used |
| `TrainingArguments.__init__() got an unexpected keyword argument 'warmup_ratio'` | Fixed in `sft.py` (see changelog) — was a real API removal in `transformers`, not a config error |
| `TypeError: unsupported operand type(s) for +: 'BatchEncoding' and 'list'` in `CompletionOnlyDataset` | Fixed in `sft.py` (see changelog) — `apply_chat_template` changed its default return shape |
| Silent kill (no traceback) mid-way through `--stages resta` or `--stages dare`, process just stops | CPU RAM exhaustion from loading multiple full fp32 model copies at once — fixed in `model_utils.py` (see changelog); if it recurs even at fp16, the next step is streaming tensors from safetensors instead of materializing full model objects |
| `hf upload` / `huggingface-cli upload` hangs forever with no output | It's blocked on an interactive "update huggingface_hub?" prompt with no stdin available. Interrupt the kernel and use `HfApi().upload_folder(...)` instead (see Session A/B cells above) — no CLI, no prompt |
| METEOR reports `nan` | NLTK corpora didn't download — check Internet is On, or `import nltk; nltk.download('wordnet')` manually |
| 403 on a dataset | Accept its terms on huggingface.co while logged in, and confirm `HF_TOKEN` is set |
| Session dies mid-run | Every stage is checkpointed; rerun the same command and finished stages are skipped |
| Interactive session idles out | Use **Save Version → Save & Run All (Commit)** for headless runs up to 12 h, and make sure the last cell copies results into `/kaggle/working` — `/kaggle/temp` is not saved |

## Changelog

**2026-08-22 — four fixes from Sessions A/B, folded into `src/safealign/`:**

1. `torchao==0.16.0` pinned in Cell 1. `peft>=0.12` checks the installed `torchao`
   version even when torchao quantization is never used, and Kaggle's base image
   ships `0.10.0`; `merge_lora`/`apply_dare` raised `ImportError` without the pin.
2. `sft.py`: `TrainingArguments(warmup_ratio=...)` replaced with a computed
   `warmup_steps` — `warmup_ratio` was removed from `transformers==5.15.1`
   (it had been deprecated-with-warning in earlier 4.x releases we'd tested against).
3. `sft.py`: `CompletionOnlyDataset.__getitem__` now calls
   `apply_chat_template(..., return_dict=False)` and normalizes the result to a
   plain `list[int]` regardless of what comes back — some `transformers` versions
   return a `BatchEncoding` here even with `tokenize=True`, which broke
   `prompt_ids + answer_ids`.
4. `model_utils.py`: `load_state_dict`/`save_state_dict` now load in `float16`
   (matching every other T4-facing path in this repo) instead of a hardcoded
   `float32`. `resta_state_dict` holds three full model copies in memory at once
   (base, target, harmful); at fp32 that peaked around 18–25 GB CPU RAM and was
   silently OOM-killed by the kernel with no traceback. At fp16 it completes in
   ~3 minutes with headroom to spare. `dare_state_dict` has the same two-model
   pattern and was very likely one model away from hitting the same wall.

All four were diagnosed from real Kaggle session logs, not anticipated in advance —
worth keeping this changelog growing rather than pruning it, since "here's what broke
against a live Kaggle T4 environment and how it was fixed" is more useful signal for
a resume/portfolio reader than a repo that looks like it worked on the first try.

## Time and quota budget

| Session | Stages | GPU | Wall clock |
|---|---|---|---|
| smoke | all, tiny | any | 0.3 h |
| A | sft, dare | T4 | ~3 h |
| B | harmful, resta | T4 | ~1.5 h |
| C | fv | T4 | ~2 h |
| D | eval | T4 ×2 | ~7 h |
| | | | **~14 h** |

T4 ×2 sessions consume quota per session, not per card, so Session D costs the same as
Session A per hour.
