"""Public demo: steer refusal behaviour with a Function Vector, live.

Runs on a free CPU Space. The base model (1.5B) plus a small LoRA adapter fit
in CPU RAM; the Function Vector is a single d_model tensor, so activation-space
steering is cheap enough to run interactively. Weight-space variants (DARE,
RESTA) are 3 GB merges each, so those are shown through pre-computed outputs
rather than loaded live — the UI says so rather than pretending otherwise.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional

import gradio as gr

ASSETS = Path(__file__).parent / "assets"
BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
SFT_ADAPTER = os.environ.get("SFT_ADAPTER", "")      # HF repo id of the LoRA adapter
FV_PATH = ASSETS / "function_vector.pt"
ENABLE_LIVE = os.environ.get("ENABLE_LIVE", "1") == "1"

INK = "#101418"
BLUE = "#2F6FB2"       # refusal-promoting direction
CRIMSON = "#B3335C"     # harmful direction
TEAL = "#0F6E63"

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=IBM+Plex+Mono:wght@400;600&family=Inter:wght@400;500&display=swap');

.gradio-container { max-width: 1080px !important; font-family: 'Inter', system-ui, sans-serif; }
#masthead { border-bottom: 1px solid rgba(16,20,24,.16); padding: 0 0 18px; margin-bottom: 8px; }
#masthead h1 { font-family: 'Space Grotesk', sans-serif; font-size: 2rem; letter-spacing: -.02em;
  margin: 0 0 4px; font-weight: 700; }
#masthead .eyebrow { font-family: 'IBM Plex Mono', monospace; font-size: .72rem;
  letter-spacing: .16em; text-transform: uppercase; color: #0F6E63; margin-bottom: 10px; }
#masthead p { margin: 0; max-width: 60ch; line-height: 1.55; opacity: .82; }

/* signature: the residual-stream rail */
.rail { display: flex; align-items: flex-end; gap: 3px; height: 46px; margin: 14px 0 4px; }
.rail .tick { flex: 1; background: rgba(16,20,24,.14); border-radius: 1px; height: 12px;
  transition: height .25s ease, background .25s ease; }
.rail .tick.pre { height: 12px; }
.rail .tick.inject { background: #0F6E63; }
.rail .tick.post { background: rgba(47,111,178,.55); }
.rail-label { font-family: 'IBM Plex Mono', monospace; font-size: .7rem; opacity: .7;
  display: flex; justify-content: space-between; }
.metric { font-family: 'IBM Plex Mono', monospace; }
.note { font-size: .82rem; opacity: .7; line-height: 1.5; }
footer { display: none !important; }
"""


def rail_html(layer: int = 9, n_layers: int = 28, active: bool = False) -> str:
    ticks = []
    for i in range(n_layers):
        if i == layer and active:
            cls, h = "tick inject", 46
        elif i > layer and active:
            cls, h = "tick post", 22
        else:
            cls, h = "tick pre", 12
        ticks.append(f'<div class="{cls}" style="height:{h}px"></div>')
    state = (f"FV added at layer {layer} · propagates through {n_layers - layer - 1} layers"
             if active else "no intervention · residual stream unmodified")
    return (f'<div class="rail">{"".join(ticks)}</div>'
            f'<div class="rail-label"><span>layer 0</span><span>{state}</span>'
            f'<span>layer {n_layers - 1}</span></div>')


# --------------------------------------------------------------------------- #
# lazy model loading
# --------------------------------------------------------------------------- #
_STATE = {"model": None, "tok": None, "fv": None, "layer": 9, "n_layers": 28}


def _load():
    if _STATE["model"] is not None:
        return _STATE
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=dtype,
                                                 low_cpu_mem_usage=True).to(device)
    if SFT_ADAPTER:
        try:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, SFT_ADAPTER)
        except Exception as e:
            print("adapter load failed:", e)
    model.eval()

    fv, layer = None, int(model.config.num_hidden_layers // 3)
    if FV_PATH.exists():
        blob = torch.load(FV_PATH, map_location="cpu")
        fv, layer = blob["fv"].float(), int(blob.get("layer", layer))

    _STATE.update({"model": model, "tok": tok, "fv": fv, "layer": layer,
                   "n_layers": model.config.num_hidden_layers})
    return _STATE


def _generate(prompt: str, lam: float, max_new_tokens: int):
    import torch

    st = _load()
    model, tok, fv, layer = st["model"], st["tok"], st["fv"], st["layer"]
    msgs = [{"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}]
    ids = tok.apply_chat_template(msgs, return_tensors="pt", add_generation_prompt=True)
    if hasattr(ids, "input_ids"):
        ids = ids.input_ids
    elif hasattr(ids, "keys"):
        ids = ids["input_ids"]
    ids = ids.to(next(model.parameters()).device)

    handle = None
    if fv is not None and lam != 0:
        vec = fv * float(lam)

        def hook(module, args, output):
            is_tuple = isinstance(output, tuple)
            h = output[0] if is_tuple else output
            h = h.clone()
            h[:, -1, :] = h[:, -1, :] + vec.to(h.dtype)
            return (h,) + tuple(output[1:]) if is_tuple else h

        handle = model.model.layers[layer].self_attn.register_forward_hook(hook)
    try:
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=int(max_new_tokens), do_sample=False,
                                 pad_token_id=tok.pad_token_id)
    finally:
        if handle is not None:
            handle.remove()
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


def run_playground(prompt: str, lam: float, max_new_tokens: int):
    if not prompt.strip():
        return "Enter a prompt to compare the two runs.", rail_html(active=False)
    if not ENABLE_LIVE:
        return ("Live inference is disabled on this deployment. The Responses tab "
                "has pre-computed outputs for all seven configurations."), rail_html(active=False)
    try:
        off = _generate(prompt, 0.0, max_new_tokens)
        on = _generate(prompt, lam, max_new_tokens)
    except Exception as e:
        return f"Generation failed: {e}", rail_html(active=False)
    st = _STATE
    md = (f"**No intervention (λ = 0)**\n\n{off}\n\n---\n\n"
          f"**Function Vector at layer {st['layer']} (λ = {lam})**\n\n{on}")
    return md, rail_html(st["layer"], st["n_layers"], active=lam != 0)


# --------------------------------------------------------------------------- #
# pre-computed results
# --------------------------------------------------------------------------- #
def _load_json(name: str) -> Optional[dict]:
    p = ASSETS / name
    return json.loads(p.read_text()) if p.exists() else None


def results_table() -> str:
    summary = _load_json("summary.json")
    if not summary:
        return "_Run the pipeline and copy `results/summary.json` into `app/assets/`._"
    rows = ["| Configuration | Unsafe Score ↓ | ROUGE-L ↑ | METEOR ↑ | BLEU ↑ |",
            "|---|---|---|---|---|"]
    for v in summary.values():
        s, u = v.get("safety", {}), v.get("utility", {})
        rows.append(f"| {v['label']} | {s.get('unsafe_score', float('nan')):.3f} "
                    f"| {u.get('rougeL', float('nan')):.3f} "
                    f"| {u.get('meteor', float('nan')):.3f} "
                    f"| {u.get('bleu', float('nan')):.1f} |")
    return "\n".join(rows)


def sample_prompts() -> List[str]:
    blob = _load_json("responses_sample.json")
    return [r["prompt"] for r in blob["rows"]] if blob else []


def show_responses(prompt: str) -> str:
    blob = _load_json("responses_sample.json")
    if not blob:
        return "_No pre-computed responses bundled._"
    for r in blob["rows"]:
        if r["prompt"] == prompt:
            return "\n\n".join(f"**{k}**\n\n{v}" for k, v in r["responses"].items())
    return "_Prompt not found._"


METHOD = """
### What is being compared

**Parameter-space.** RESTA computes a safety vector as the difference between the
aligned base model and a deliberately unaligned copy of it, `δ_safe = θ_base − θ_harmful`,
then adds that vector back to a fine-tuned model's weights. DARE first sparsifies the
fine-tuning delta — drop a fraction *p* of its entries, rescale the survivors by `1/(1−p)` —
which reduces interference when the safety vector is added on top.

**Activation-space.** Causal Mediation Analysis measures, for every attention head, how much
forcing its clean mean activation into a corrupted forward pass restores refusal-token
probability. The top heads' mean activations are summed into a Function Vector, which is
added to the residual stream at inference. No weight is ever modified.

The trade-off worth watching: weight edits are free at inference but permanent and blunt;
activation edits are reversible and targeted but cost a hook on every forward pass, and their
strength has to be tuned — too much λ and fluency collapses.
"""


with gr.Blocks(css=CSS, theme=gr.themes.Base(
        primary_hue=gr.themes.colors.teal, neutral_hue=gr.themes.colors.slate),
        title="Safety interventions") as demo:

    gr.HTML(f"""
    <div id="masthead">
      <div class="eyebrow">Parameter space · Activation space</div>
      <h1>Two ways to put refusal back into a fine-tuned model</h1>
      <p>Fine-tuning a chat model on a narrow domain quietly erodes its safety training.
      This demo compares repairing that in the weights (RESTA, DARE) against steering it at
      inference with a Function Vector read out of the model's own attention heads.</p>
    </div>""")

    with gr.Tab("Steer it live"):
        rail = gr.HTML(rail_html(active=False))
        with gr.Row():
            with gr.Column(scale=3):
                prompt = gr.Textbox(label="Prompt", lines=3,
                                    placeholder="Ask the model something it should decline.")
                with gr.Row():
                    lam = gr.Slider(0, 3, value=1.0, step=0.1,
                                    label="λ — Function Vector strength")
                    ntok = gr.Slider(32, 256, value=96, step=32, label="Max new tokens")
                go = gr.Button("Generate both runs", variant="primary")
            with gr.Column(scale=4):
                out = gr.Markdown()
        gr.Markdown("Two greedy generations run per click: one untouched, one with the "
                    "Function Vector added at the last token position. On the free CPU tier "
                    "each pair takes roughly 30–60 seconds.", elem_classes="note")
        go.click(run_playground, [prompt, lam, ntok], [out, rail])

    with gr.Tab("Results"):
        gr.Markdown("#### Seven configurations, 550 HarmEval prompts, held-out medical split")
        gr.Markdown(results_table())
        with gr.Row():
            if (ASSETS / "aie_heatmap.png").exists():
                gr.Image(str(ASSETS / "aie_heatmap.png"), label="AIE per attention head",
                         show_label=True)
            if (ASSETS / "tradeoff.png").exists():
                gr.Image(str(ASSETS / "tradeoff.png"), label="Safety-utility trade-off")
        if (ASSETS / "fv_logit_lens.png").exists():
            gr.Image(str(ASSETS / "fv_logit_lens.png"),
                     label="What the Function Vector decodes to")

    with gr.Tab("Responses"):
        choices = sample_prompts()
        picker = gr.Dropdown(choices, label="HarmEval prompt",
                             value=choices[0] if choices else None)
        resp = gr.Markdown(show_responses(choices[0]) if choices else "_No samples bundled._")
        picker.change(show_responses, picker, resp)

    with gr.Tab("Method"):
        gr.Markdown(METHOD)

if __name__ == "__main__":
    demo.launch()
