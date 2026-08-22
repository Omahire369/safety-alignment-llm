---
title: Safety Interventions - Parameter vs Activation Space
emoji: 🧭
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
short_description: Steer refusal behaviour with a Function Vector, live
---

# Safety Alignment in LLMs — Parameter-Space vs Activation-Space Interventions

Live demo for [safety-alignment-llm](https://github.com/<your-username>/safety-alignment-llm).

**Tabs**
- *Steer it live* — one prompt, two greedy generations: untouched vs Function Vector injected
  at layer `floor(L/3)`, with an adjustable λ.
- *Results* — Unsafe Score and utility metrics for all seven configurations, the AIE heatmap,
  and the logit-lens decoding of the Function Vector.
- *Responses* — pre-computed side-by-side outputs for HarmEval prompts.

## Space configuration

| Variable | Meaning | Default |
|---|---|---|
| `BASE_MODEL` | base checkpoint | `Qwen/Qwen2.5-1.5B-Instruct` |
| `SFT_ADAPTER` | HF repo id of the LoRA adapter (optional) | unset |
| `ENABLE_LIVE` | set `0` to serve pre-computed results only | `1` |

Copy these into `app/assets/` after a pipeline run:
`function_vector.pt`, `summary.json`, `responses_sample.json`,
`aie_heatmap.png`, `tradeoff.png`, `fv_logit_lens.png`.

Weight-space variants (DARE, RESTA) are ~3 GB merges each and are not loaded live on the
free CPU tier; their outputs appear in the *Responses* and *Results* tabs.
