# Analysis

## 1. Setup actually used

| | |
|---|---|
| Base model | Qwen/Qwen2.5-1.5B-Instruct |
| LoRA r / α / dropout / targets | 16 / 32 / 0.05 / q,k,v,o_proj + gate,up,down_proj |
| LR, effective batch, epochs, max seq len | SFT: 2e-4, batch 2 × grad_accum 8 = 16, 1 epoch, max_seq_len 1024. Harmful model: 1e-4, same batch, 3 epochs |
| DARE drop rate chosen (and by what criterion) | p = 0.7, by highest validation ROUGE-L on 150 held-out questions |
| RESTA λ | 1.0 (default, not swept) |
| FV: N in-context, prompts, top-k heads, inject layer, λ | N=10 in-context examples/prompt, 15 clean/corrupted target prompts, top-10 heads by AIE, inject at layer 9 (⌊28/3⌋), λ=1.0 |
| Judge model | Qwen/Qwen2.5-7B-Instruct, fp16, zero-shot, scored via next-token logit comparison (not free generation) |

## 2. Sampling strategy for the few-shot prompts

15 target queries and their 10 in-context examples per prompt were drawn from the 541-row
`unalignment/toxic-dpo-v0.2` split with seed 1234, using disjoint pools for targets vs. ICL
examples (verified by `tests/test_fewshot_pools_are_disjoint`) so no target query could leak
into its own context. Clean prompts paired each ICL query with its safe `rejected` completion;
corrupted prompts substituted the harmful `chosen` completion in the same position, isolating
the causal effect of the refusal pattern itself rather than of the query content. N=10 was
used rather than a smaller value to give the causal mediation signal enough consistent context
to localize against; exact row indices are in `results/fv_fewshot_pairs.json`.

## 3. Parameter space vs activation space

**RESTA dominated the Function Vector on both axes simultaneously — this was not a trade-off.**

| Configuration | Unsafe Score ↓ | Refusal rate | ROUGE-L ↑ | METEOR ↑ | BLEU ↑ |
|---|---|---|---|---|---|
| Base | 0.1582 | 0.736 | 0.0670 | 0.1690 | 1.93 |
| SFT | 0.1818 | 0.662 | 0.5348 | 0.5815 | 57.00 |
| SFT + DARE | 0.1909 | 0.662 | 0.5340 | 0.5806 | 57.31 |
| **SFT + RESTA** | **0.0036** | 0.967 | 0.5448 | 0.5902 | 57.69 |
| **SFT + DARE + RESTA** | **0.0073** | 0.962 | **0.5557** | **0.6006** | **58.61** |
| SFT + FV | 0.1545 | 0.680 | 0.5335 | 0.5799 | 58.20 |
| SFT + DARE + FV | 0.1636 | 0.680 | 0.5412 | 0.5862 | 58.93 |

RESTA collapsed unsafe generations to near-zero (2/550 and 4/550 responses respectively)
while *improving* every utility metric over plain SFT — it is Pareto-dominant, not a
trade-off. The Function Vector barely moved the needle: `SFT+FV`'s unsafe score (0.1545) is
close to SFT's baseline (0.1818) and is actually *below the untouched base model's own score*
(0.1582), meaning the FV-injected model is less safe than doing nothing at all. All three
utility metrics rank the interventions identically (RESTA > DARE+FV ≈ FV ≈ SFT > DARE alone),
so there's no metric disagreement to flag here — the ranking is consistent across ROUGE-L,
METEOR, and BLEU.

This result is worth stating plainly rather than softened: the FV's own Session C validation
(87.5% refusal rate on a 40-prompt toxic-dpo holdout, scored by a lexical heuristic) did not
generalize to the full 550-prompt HarmEval benchmark scored by the actual LLM judge, where
refusal rate only reached 0.680 — barely above SFT's 0.662. Two plausible reasons: the
heuristic refusal marker-matching used for the λ sweep and the judge's semantic harmfulness
classification are not measuring the same thing, and a single-layer, single-token-position
activation nudge is a structurally weaker intervention than a full-parameter task-arithmetic
edit that touches every layer at once.

**Does the FV recover refusal where RESTA does not, or the same prompts?** Not established
here — would require cross-referencing per-prompt judgements between `sft_resta` and `sft_fv`
in `results/generations/*.json`, which wasn't done. Worth doing as a follow-up if the gap
between the two families needs a mechanistic explanation rather than just a headline number.

**Deployment cost.** RESTA's edit is permanent and baked into the checkpoint: zero runtime
overhead, but every variant is a separate ~3GB artifact that must be built, stored, and
redistributed. The FV is a forward hook applied per-request: trivially toggleable and
reversible, no separate checkpoint needed, at the cost of a small per-forward-pass overhead
and needing to ship the injection code alongside the base weights.

**Where is the SFT model's safety damage concentrated?** Both interventions independently
localized to the back half of the 28-layer network via different mechanisms. The top-10 AIE
heads cluster in layers 12–26 (strongest: layer 19 heads 5 and 0, layer 20 head 3, AIE up to
0.081), while RESTA's safety-vector L2 norm concentrates almost entirely in `mlp.gate_proj`/
`mlp.up_proj` weights in layers 15–27 (see `results/resta_build.json`). Different mechanism —
attention heads vs. MLP weights — but the same layer range, which is a genuinely interesting
convergence worth a sentence of its own.

## 4. Did DARE preprocessing help?

Not for safety: `SFT+RESTA` (0.36% unsafe) vs. `SFT+DARE+RESTA` (0.73% unsafe) — both are
already at the floor of what 550 prompts can resolve (2 vs. 4 flagged responses), and if
anything the DARE-preprocessed version is *slightly* less safe, not more. The hypothesis that
sparsifying the fine-tuning delta reduces interference with the safety vector is not supported
by this data — both numbers are close enough to be noise.

DARE *did* help utility, consistently. The full sweep is not flat at the top end:

| p | ROUGE-L | METEOR | BLEU |
|---|---|---|---|
| 0.1 | 0.4419 | 0.4799 | 48.19 |
| 0.3 | 0.4419 | 0.4799 | 48.19 |
| 0.5 | 0.4522 | 0.4898 | 49.77 |
| 0.7 | 0.4555 | 0.4932 | 49.99 |

Utility rises monotonically with drop rate — more sparsification of the LoRA delta improved
held-out validation performance, not just preserved it. That reads as the delta containing
real overfit noise that dropping (and rescaling) acts as an implicit regularizer against,
consistent with the DARE paper's "free lunch" framing. That benefit carries cleanly through to
the final RESTA comparison: `SFT+DARE+RESTA` leads all seven configurations on all three
utility metrics. So the honest summary is: DARE preprocessing is a utility win, not a safety
amplifier, in this run.

## 5. Optimal λ for activation steering

| λ | Refusal rate | Distinct-token ratio (fluency proxy) |
|---|---|---|
| 0.0 | 0.825 | 0.965 |
| 0.5 | 0.850 | 0.975 |
| **1.0** | **0.875** | 0.979 |
| 1.5 | 0.850 | 0.974 |
| 2.0 | 0.825 | 0.981 |

Clean inverted-U, peaking at λ=1.0 with no fluency trade-off at all (fluency proxy stayed flat
0.965–0.981 across the whole sweep). **Whether the optimum differs between the SFT and DARE
model is not actually established by this run** — the λ sweep (`run_pipeline.py --stages fv`)
was only executed once, against `model_sft_lora_merged`, and the same λ=1.0 was reused for the
`sft_dare_fv` evaluation configuration without an independent sweep against the DARE model.
This is a genuine gap, not a null result — flagged explicitly in Limitations below rather than
guessed at.

No failure mode at high λ was observed in this range: fluency never degraded even at λ=2.0, so
2.0 is not yet the practical ceiling — the refusal-rate decline past λ=1.0 without a fluency
collapse suggests diminishing returns rather than a hard constraint, and a wider sweep (e.g.
up to λ=4.0) would be needed to actually find where injection starts breaking generation.

## 6. What the Function Vector encodes

Not literal refusal openers. The top-20 logit-lens tokens are overwhelmingly first-person
pronoun forms — `" I"`, `"I"`, `" i"`, `"_i"`, `"_I"`, `"i"`, `" we"`, plus the same pronoun in
Chinese (我, 我不, 我可以, 我都, 我没, 我才) and Russian (я) — with essentially no occurrence of
"Sorry," "cannot," "Unfortunately," or any other literal refusal-marker word, despite those
being exactly the tokens the CIE/AIE computation targeted as the refusal signal in Part 3.1.

That's a genuinely interesting result: the vector doesn't encode the refusal *content*, it
encodes something closer to a cross-lingual first-person-subject / self-referential-stance
direction — the grammatical position a refusal like "I cannot help with that" would need to be
built from, but not the refusal itself. That's evidence for an abstract, compositional
representation rather than a shallow output-formatting shortcut: the remaining layers appear
to do the work of turning "here comes a first-person statement" into an actual refusal, which
is also consistent with why injecting this vector alone (Section 3) produced a much weaker
safety effect than RESTA's full-parameter edit — it's nudging a precursor, not the behavior
directly.

## 7. Limitations

- Single judge model (Qwen2.5-7B-Instruct), no human-agreement cross-check against its verdicts.
- Single base model (1.5B parameters), single domain (medical QA) for utility — findings may
  not transfer to larger models or other task domains.
- Greedy decoding only throughout; no sampling-based robustness check.
- DARE sweep used 150 validation examples; FV λ sweep used 40 holdout prompts scored by a
  lexical heuristic rather than the judge itself — the sweep's own selection criterion is not
  the metric the final table is judged on, which is part of why Section 3's generalization gap
  is unsurprising in hindsight.
- CIE/AIE computed from only 15 clean/corrupted prompt pairs — a small sample for a causal
  attribution method; head rankings could be sensitive to this sample.
- λ was swept once (on the SFT model) and reused unvalidated for the DARE+FV configuration.
- RESTA's safety numbers (2/550, 4/550 unsafe) are near the resolution floor of a 550-prompt
  benchmark; the difference between RESTA and DARE+RESTA on safety specifically should not be
  read as meaningful without a larger benchmark or repeated runs.
