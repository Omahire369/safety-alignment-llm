"""Part 3.2 — build the Function Vector and inject it into the residual stream.

FV = sum over the top-k heads of their mean clean (post-W_O) activation, so the
FV already lives in R^{d_model} and can be added to the residual stream
directly. Injection point: floor(L/3) (layer 9 for a 28-layer Qwen2.5-1.5B),
immediately after the attention block and before the MLP, at the final token
position, scaled by lambda.
"""
from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch

from ..config import CFG
from ..model_utils import free


def target_layer(model) -> int:
    return int(model.config.num_hidden_layers * CFG.fv.inject_layer_frac)


def build_function_vector(abar: torch.Tensor,
                          top_heads: Sequence[Tuple[int, int]],
                          reduce: str = "sum") -> torch.Tensor:
    """abar: [L, H, d_model] post-projection mean clean activations."""
    stack = torch.stack([abar[l, h] for l, h in top_heads], dim=0)
    return stack.sum(0) if reduce == "sum" else stack.mean(0)


class FVInjector:
    """Adds lambda * FV to the attention block output at the last position.

    Hooking `self_attn` (not the decoder layer) means the vector is written into
    the residual stream after attention and before the MLP, which is where the
    Function Vectors paper places it.
    """

    def __init__(self, model, fv: torch.Tensor, layer: Optional[int] = None,
                 lam: Optional[float] = None, last_token_only: bool = True):
        self.model = model
        self.layer = target_layer(model) if layer is None else layer
        self.lam = CFG.fv.chosen_lambda if lam is None else lam
        self.fv = fv
        self.last_token_only = last_token_only
        self._handle = None

    def _hook(self, module, args, output):
        if self.lam == 0:
            return output
        is_tuple = isinstance(output, tuple)
        hidden = output[0] if is_tuple else output
        vec = self.fv.to(device=hidden.device, dtype=hidden.dtype) * self.lam
        hidden = hidden.clone()
        if self.last_token_only:
            hidden[:, -1, :] = hidden[:, -1, :] + vec
        else:
            hidden = hidden + vec
        return (hidden,) + tuple(output[1:]) if is_tuple else hidden

    def __enter__(self):
        mod = self.model.model.layers[self.layer].self_attn
        self._handle = mod.register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


@contextlib.contextmanager
def maybe_inject(model, fv: Optional[torch.Tensor], layer: Optional[int] = None,
                 lam: Optional[float] = None):
    """No-op context manager when fv is None, so callers stay branch-free."""
    if fv is None:
        yield None
        return
    with FVInjector(model, fv, layer, lam) as inj:
        yield inj


def save_fv(fv: torch.Tensor, top_heads: List[Tuple[int, int]], layer: int,
            path: Optional[Path] = None) -> Path:
    path = Path(path or CFG.paths.results / "function_vector.pt")
    torch.save({"fv": fv.cpu(), "top_heads": top_heads, "layer": layer,
                "norm": float(fv.norm())}, path)
    return path


def load_fv(path: Optional[Path] = None):
    path = Path(path or CFG.paths.results / "function_vector.pt")
    blob = torch.load(path, map_location="cpu")
    return blob["fv"], blob


def build_and_save(top_heads: Optional[List[Tuple[int, int]]] = None) -> dict:
    """Reads the CMA artifacts written by fv.cie and produces function_vector.pt."""
    from ..model_utils import load_model

    summary = json.loads((CFG.paths.results / "cma_summary.json").read_text())
    top_heads = top_heads or [(h["layer"], h["head"]) for h in summary["top_heads"]]
    blob = torch.load(CFG.paths.results / "mean_head_activations.pt", map_location="cpu")
    fv = build_function_vector(blob["abar"], top_heads)

    model = load_model(device_map="cpu")
    layer = target_layer(model)
    n_layers = model.config.num_hidden_layers
    del model
    free()

    save_fv(fv, top_heads, layer)
    out = {"fv_norm": float(fv.norm()), "inject_layer": layer,
           "n_layers": n_layers, "top_heads": top_heads}
    (CFG.paths.results / "fv_build.json").write_text(json.dumps(out, indent=2))
    return out


# --------------------------------------------------------------------------- #
# lambda selection
# --------------------------------------------------------------------------- #
def sweep_lambda(model_id: str, n_val: int = 40,
                 lambdas: Optional[List[float]] = None) -> dict:
    """Pick the multiplier that maximises refusal without wrecking fluency.

    Refusal rate is measured on held-out harmful prompts that were never used
    for the CMA sampling; fluency is proxied by mean token log-probability of
    the model's own continuation (a collapsed / repetitive model scores badly).
    """
    from ..data import load_toxic_dpo
    from ..evaluation.generate import generate_batch
    from ..evaluation.safety import heuristic_refusal
    from ..model_utils import load_model, load_tokenizer

    lambdas = lambdas or CFG.fv.lambdas
    fv, meta = load_fv()

    ds = load_toxic_dpo()
    pairs = json.loads((CFG.paths.results / "fv_fewshot_pairs.json").read_text())["pairs"]
    used = {p["idx"] for p in pairs} | {i for p in pairs for i in p["icl_indices"]}
    holdout = [ds[i]["prompt"] for i in range(len(ds)) if i not in used][:n_val]

    tok = load_tokenizer(model_id)
    model = load_model(model_id)
    results = {}
    for lam in [0.0] + list(lambdas):
        with maybe_inject(model, fv if lam > 0 else None, meta["layer"], lam):
            outs = generate_batch(model, tok, [[{"role": "user", "content": q}]
                                               for q in holdout],
                                  max_new_tokens=64)
        refusals = sum(heuristic_refusal(o) for o in outs) / max(len(outs), 1)
        distinct = sum(len(set(o.split())) / max(len(o.split()), 1) for o in outs) / max(len(outs), 1)
        results[str(lam)] = {"refusal_rate": refusals, "distinct_token_ratio": distinct}
        print(f"[fv-lambda] lambda={lam} {results[str(lam)]}")

    del model
    free()

    # Highest refusal among lambdas that keep lexical diversity healthy.
    ok = {k: v for k, v in results.items() if v["distinct_token_ratio"] > 0.45}
    best = max(ok or results, key=lambda k: results[k]["refusal_rate"])
    out = {"sweep": results, "best_lambda": float(best)}
    (CFG.paths.results / "fv_lambda_sweep.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    print(json.dumps(build_and_save(), indent=2))
