"""Batched greedy generation, optionally with a Function Vector injected."""
from __future__ import annotations

from typing import List, Optional

import torch
from tqdm.auto import tqdm

from ..config import CFG


@torch.no_grad()
def generate_batch(model, tokenizer, chats: List[List[dict]],
                   max_new_tokens: Optional[int] = None,
                   batch_size: Optional[int] = None,
                   show_progress: bool = False) -> List[str]:
    max_new_tokens = max_new_tokens or CFG.eval.max_new_tokens
    batch_size = batch_size or CFG.eval.gen_batch_size
    device = next(model.parameters()).device

    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "left"            # required for correct decoder-only batching
    outputs: List[str] = []

    it = range(0, len(chats), batch_size)
    if show_progress:
        it = tqdm(it, desc="generate")
    for start in it:
        chunk = chats[start:start + batch_size]
        texts = [tokenizer.apply_chat_template(c, tokenize=False,
                                               add_generation_prompt=True)
                 for c in chunk]
        enc = tokenizer(texts, return_tensors="pt", padding=True,
                        truncation=True, max_length=CFG.model.max_seq_len,
                        add_special_tokens=False).to(device)
        gen = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=tokenizer.pad_token_id,
        )
        for i in range(gen.size(0)):
            new_tokens = gen[i, enc["input_ids"].shape[1]:]
            outputs.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())

    tokenizer.padding_side = prev_side
    return outputs


@torch.no_grad()
def generate_with_fv(model, tokenizer, chats: List[List[dict]],
                     fv: Optional[torch.Tensor], layer: Optional[int] = None,
                     lam: Optional[float] = None, **kwargs) -> List[str]:
    from ..fv.vector import maybe_inject

    with maybe_inject(model, fv, layer, lam):
        return generate_batch(model, tokenizer, chats, **kwargs)
