"""Part 3.1 — Causal Mediation Analysis over attention heads.

Head activations are taken **after** the per-head slice of the output
projection W_O, i.e. the quantity

    a_lj  =  z_lj @ W_O[:, j*d_head : (j+1)*d_head].T        in R^{d_model}

which is the head's additive contribution to the residual stream. The raw
pre-projection z_lj lives in head space and cannot be added to the residual
stream, so it is only ever used as a patching target (patching z_lj is
equivalent to patching a_lj because the slice-projection is linear).

Pipeline
--------
1. Clean run over the 15 clean prompts, caching z_lj at the final input token.
   Mean over prompts -> zbar_lj, and abar_lj = proj(zbar_lj).
2. Corrupted run  -> baseline P(V_refusal | corrupted).
3. Corrupted run with head (l,j) forced to zbar_lj -> P(V_refusal | patched).
   CIE(l,j) = sum_{w in V_refusal} P(w|patched) - P(w|corrupted)
4. AIE(l,j) = mean over the 15 prompts. Top-k heads by AIE build the FV.

Cost control: a naive loop is L*H*n_prompts forward passes (28*12*15 = 5040).
Instead each prompt is tiled `head_batch_size` times along the batch dimension
and a different head is patched in each replica, cutting the wall-clock by
roughly `head_batch_size`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm

from ..config import CFG
from ..model_utils import free, load_model, load_tokenizer, n_layers_heads
from .prompts import FewShotPair, build_fewshot_pairs, refusal_token_ids, save_pairs


def _layers(model):
    return model.model.layers


def _o_proj(model, layer: int):
    return _layers(model)[layer].self_attn.o_proj


def head_slice(W_o: torch.Tensor, head: int, head_dim: int) -> torch.Tensor:
    """W_O column block for one head -> [d_model, d_head]."""
    return W_o[:, head * head_dim:(head + 1) * head_dim]


class HeadCapture:
    """Forward pre-hooks on every o_proj capturing z at the final token."""

    def __init__(self, model):
        self.model = model
        self.n_layers, self.n_heads, self.head_dim = n_layers_heads(model)
        self.store: Dict[int, torch.Tensor] = {}
        self._handles = []

    def _mk(self, layer: int):
        def hook(module, args):
            x = args[0]                       # [B, T, n_heads*head_dim]
            self.store[layer] = x[:, -1, :].detach().float().cpu()
            return None
        return hook

    def __enter__(self):
        for l in range(self.n_layers):
            self._handles.append(_o_proj(self.model, l).register_forward_pre_hook(self._mk(l)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def stacked(self) -> torch.Tensor:
        """[n_layers, n_heads, head_dim] for a batch size of 1."""
        z = torch.stack([self.store[l][0] for l in range(self.n_layers)], dim=0)
        return z.view(self.n_layers, self.n_heads, self.head_dim)


class HeadPatcher:
    """Patch selected heads at the final token, one head per batch element."""

    def __init__(self, model, zbar: torch.Tensor):
        self.model = model
        self.zbar = zbar                       # [L, H, d_head], float32 cpu
        self.n_layers, self.n_heads, self.head_dim = n_layers_heads(model)
        self.assignment: List[Tuple[int, int]] = []
        self._handles = []

    def set_assignment(self, heads: Sequence[Tuple[int, int]]):
        """heads[b] = (layer, head) patched in batch element b."""
        self.assignment = list(heads)

    def _mk(self, layer: int):
        def hook(module, args):
            x = args[0]
            if not self.assignment:
                return None
            targets = [(b, h) for b, (l, h) in enumerate(self.assignment) if l == layer]
            if not targets:
                return None
            x = x.clone()
            for b, h in targets:
                repl = self.zbar[layer, h].to(device=x.device, dtype=x.dtype)
                x[b, -1, h * self.head_dim:(h + 1) * self.head_dim] = repl
            return (x,) + tuple(args[1:])
        return hook

    def __enter__(self):
        for l in range(self.n_layers):
            self._handles.append(_o_proj(self.model, l).register_forward_pre_hook(self._mk(l)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()


@torch.no_grad()
def _prompt_ids(tokenizer, messages, device):
    out = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=False
    )
    # transformers versions have returned three different shapes here: a plain
    # list[int], a BatchEncoding (dict-like, has .keys()), or a raw
    # tokenizers.Encoding (has .ids). Normalize to a plain list either way.
    if hasattr(out, "ids"):
        ids = list(out.ids)
    elif hasattr(out, "keys"):
        ids = list(out["input_ids"])
    else:
        ids = list(out)
    return torch.tensor([ids], device=device)


@torch.no_grad()
def _refusal_mass(logits: torch.Tensor, refusal_ids: torch.Tensor) -> torch.Tensor:
    """logits [B, V] -> summed probability mass on V_refusal, shape [B]."""
    probs = torch.softmax(logits.float(), dim=-1)
    return probs.index_select(-1, refusal_ids).sum(-1)


@torch.no_grad()
def compute_mean_clean_activations(model, tokenizer, pairs: List[FewShotPair]):
    """Step 1: zbar [L,H,d_head] and abar [L,H,d_model] (post-W_O)."""
    device = next(model.parameters()).device
    acc = None
    with HeadCapture(model) as cap:
        for p in tqdm(pairs, desc="clean activations"):
            ids = _prompt_ids(tokenizer, p.clean_messages, device)
            model(input_ids=ids)
            z = cap.stacked()
            acc = z if acc is None else acc + z
    zbar = acc / len(pairs)

    n_layers, n_heads, head_dim = n_layers_heads(model)
    d_model = model.config.hidden_size
    abar = torch.zeros(n_layers, n_heads, d_model, dtype=torch.float32)
    for l in range(n_layers):
        W_o = _o_proj(model, l).weight.detach().float().cpu()   # [d_model, n_heads*d_head]
        for h in range(n_heads):
            abar[l, h] = head_slice(W_o, h, head_dim) @ zbar[l, h]
    return zbar, abar


@torch.no_grad()
def compute_cie(model, tokenizer, pairs: List[FewShotPair], zbar: torch.Tensor,
                refusal_ids: List[int], head_batch_size: Optional[int] = None
                ) -> np.ndarray:
    """Steps 2-4: CIE per (prompt, layer, head) -> [n_prompts, L, H]."""
    device = next(model.parameters()).device
    hbs = head_batch_size or CFG.fv.head_batch_size
    n_layers, n_heads, _ = n_layers_heads(model)
    all_heads = [(l, h) for l in range(n_layers) for h in range(n_heads)]
    rid = torch.tensor(refusal_ids, device=device)

    cie = np.zeros((len(pairs), n_layers, n_heads), dtype=np.float32)
    patcher = HeadPatcher(model, zbar)

    for pi, p in enumerate(tqdm(pairs, desc="CIE")):
        ids = _prompt_ids(tokenizer, p.corrupted_messages, device)

        # Step 2 — corrupted baseline.
        base_logits = model(input_ids=ids).logits[:, -1, :]
        base_mass = _refusal_mass(base_logits, rid).item()

        # Step 3/4 — patch each head, batched.
        with patcher:
            for start in range(0, len(all_heads), hbs):
                chunk = all_heads[start:start + hbs]
                patcher.set_assignment(chunk)
                batch = ids.repeat(len(chunk), 1)
                logits = model(input_ids=batch).logits[:, -1, :]
                mass = _refusal_mass(logits, rid).cpu().numpy()
                for k, (l, h) in enumerate(chunk):
                    cie[pi, l, h] = float(mass[k]) - base_mass
        free()
    return cie


def run_cma(model_id: Optional[str] = None, save: bool = True) -> dict:
    """Full Part 3.1: returns AIE matrix + top-k heads."""
    tokenizer = load_tokenizer(model_id)
    model = load_model(model_id)
    pairs = build_fewshot_pairs()
    if save:
        save_pairs(pairs)

    refusal_ids = refusal_token_ids(tokenizer)
    zbar, abar = compute_mean_clean_activations(model, tokenizer, pairs)
    cie = compute_cie(model, tokenizer, pairs, zbar, refusal_ids)
    aie = cie.mean(axis=0)                       # [L, H]

    flat = np.dstack(np.unravel_index(np.argsort(-aie, axis=None), aie.shape))[0]
    top = [(int(l), int(h), float(aie[l, h])) for l, h in flat[:CFG.fv.top_k_heads]]

    out = {
        "top_heads": [{"layer": l, "head": h, "aie": v} for l, h, v in top],
        "refusal_token_ids": refusal_ids,
        "refusal_tokens": [tokenizer.decode([i]) for i in refusal_ids],
        "n_prompts": len(pairs), "n_icl": CFG.fv.n_icl,
        "aie_max": float(aie.max()), "aie_min": float(aie.min()),
    }

    if save:
        np.save(CFG.paths.results / "aie_matrix.npy", aie)
        np.save(CFG.paths.results / "cie_per_prompt.npy", cie)
        torch.save({"zbar": zbar, "abar": abar},
                   CFG.paths.results / "mean_head_activations.pt")
        (CFG.paths.results / "cma_summary.json").write_text(json.dumps(out, indent=2))

    del model
    free()
    return out


if __name__ == "__main__":
    print(json.dumps(run_cma(), indent=2))
