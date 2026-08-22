"""Part 3.3 — interpretability figures.

1. AIE heatmap  : layer x head grid of Average Indirect Effect, diverging map
                  centred at zero so positive (refusal-promoting) and negative
                  heads are visually separable.
2. Logit lens   : push the FV through the final LayerNorm and the unembedding
                  to read the top-20 vocabulary items it decodes to. This is
                  the evidence for whether the FV is literally spelling
                  "Sorry" or encoding something more abstract.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from ..config import CFG
from ..model_utils import free, load_model, load_tokenizer


def plot_aie_heatmap(aie: Optional[np.ndarray] = None,
                     out_path: Optional[Path] = None,
                     top_k: int = 10) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    aie = np.load(CFG.paths.results / "aie_matrix.npy") if aie is None else aie
    out_path = Path(out_path or CFG.paths.figures / "aie_heatmap.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    vmax = float(np.abs(aie).max())
    fig, ax = plt.subplots(figsize=(9, 7), dpi=160)
    im = ax.imshow(aie, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   interpolation="nearest")

    flat = np.dstack(np.unravel_index(np.argsort(-aie, axis=None), aie.shape))[0]
    for l, h in flat[:top_k]:
        ax.add_patch(plt.Rectangle((h - 0.5, l - 0.5), 1, 1, fill=False,
                                   edgecolor="black", linewidth=1.4))

    ax.set_xlabel("Head index")
    ax.set_ylabel("Layer index")
    ax.set_title("Average Indirect Effect on refusal-token probability\n"
                 f"(boxed = top-{top_k} heads used for the Function Vector)")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("AIE  (Δ refusal probability mass)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


@torch.no_grad()
def logit_lens(fv: Optional[torch.Tensor] = None, model_id: Optional[str] = None,
               top_n: int = 20) -> dict:
    """FV -> final RMSNorm -> lm_head -> top-N tokens."""
    from .vector import load_fv

    if fv is None:
        fv, _ = load_fv()
    tok = load_tokenizer(model_id)
    model = load_model(model_id, device_map="cpu", dtype=torch.float32)

    h = fv.to(torch.float32).unsqueeze(0)
    normed = model.model.norm(h)
    logits = model.lm_head(normed)[0]
    probs = torch.softmax(logits, dim=-1)
    vals, idx = torch.topk(probs, top_n)

    out = {"top_tokens": [
        {"rank": i + 1, "token": tok.decode([int(t)]), "token_id": int(t),
         "prob": float(v)}
        for i, (v, t) in enumerate(zip(vals, idx))
    ]}
    del model
    free()

    (CFG.paths.results / "fv_logit_lens.json").write_text(json.dumps(out, indent=2))
    return out


def plot_logit_lens(entries: Optional[List[dict]] = None,
                    out_path: Optional[Path] = None) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if entries is None:
        entries = json.loads((CFG.paths.results / "fv_logit_lens.json").read_text()
                             )["top_tokens"]
    out_path = Path(out_path or CFG.paths.figures / "fv_logit_lens.png")
    labels = [repr(e["token"]) for e in entries][::-1]
    values = [e["prob"] for e in entries][::-1]

    fig, ax = plt.subplots(figsize=(7, 8), dpi=160)
    ax.barh(labels, values, color="#2b6cb0")
    ax.set_xlabel("Probability after unembedding")
    ax.set_title("Logit-lens decoding of the Function Vector")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def build_all_figures() -> dict:
    heat = plot_aie_heatmap()
    lens = logit_lens()
    bar = plot_logit_lens(lens["top_tokens"])
    return {"aie_heatmap": str(heat), "logit_lens_fig": str(bar),
            "top_tokens": [e["token"] for e in lens["top_tokens"]]}


if __name__ == "__main__":
    print(json.dumps(build_all_figures(), indent=2))
