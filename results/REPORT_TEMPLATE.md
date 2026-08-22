# Analysis

Fill this in from `results/summary.json`, `results/dare_sweep.json`,
`results/fv_lambda_sweep.json` and `results/cma_summary.json` after a full run. Keep the
numbers, delete the prompts.

## 1. Setup actually used

| | |
|---|---|
| Base model | |
| LoRA r / α / dropout / targets | |
| LR, effective batch, epochs, max seq len | |
| DARE drop rate chosen (and by what criterion) | |
| RESTA λ | |
| FV: N in-context, prompts, top-k heads, inject layer, λ | |
| Judge model | |

## 2. Sampling strategy for the few-shot prompts

How the 15 target queries and their N in-context examples were drawn from the 541 rows, why
the pools were kept disjoint, and why that N. (Written up in `fv/prompts.py`; restate the
result here with the actual seed and indices from `results/fv_fewshot_pairs.json`.)

## 3. Parameter space vs activation space

Which family bought more safety per unit of utility lost? Read the Unsafe Score column
against ROUGE-L / METEOR / BLEU, and say explicitly whether the ranking is the same on all
three utility metrics — if BLEU and ROUGE-L disagree, that itself is a finding.

Points worth addressing:
- Does the FV recover refusal on prompts where RESTA does not, or the same ones?
- Weight edits are permanent; the FV is per-request. What does that cost at inference?
- Is the SFT model's damage concentrated (a few heads) or diffuse? The AIE heatmap answers
  this: report which layers the top-10 heads cluster in.

## 4. Did DARE preprocessing help?

Compare SFT+RESTA against SFT+DARE+RESTA on both axes. The hypothesis is that sparsifying
the fine-tuning delta reduces interference with the safety vector, so the same δ_safe should
buy more safety on top of the DARE model. State whether the data supports that, and report
the full drop-rate sweep, not just the winner — a flat sweep means the effect is small.

## 5. Optimal λ for activation steering

Report the sweep, the selected λ, and whether it differed between the SFT model and the DARE
model. If it did, say what that implies: a model whose delta has been sparsified may sit
closer to the base model's activation statistics, so it needs less push.

Also note the failure mode at high λ — where fluency starts to break — since that upper
bound is the practical constraint on activation steering.

## 6. What the Function Vector encodes

From the logit lens: does the FV decode to literal refusal openers ("Sorry", "I"), or to
something more abstract? Literal tokens suggest it is a shallow output-formatting direction;
abstract or unrelated tokens suggest a genuine task representation that only becomes refusal
after the remaining layers process it. Say which, with the top-20 list as evidence.

## 7. Limitations

Sample sizes, judge reliability (single judge, no human agreement check), one base model,
one domain, greedy decoding only.
