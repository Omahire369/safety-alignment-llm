"""Part 3.1 — few-shot prompt construction for causal mediation analysis.

Sampling strategy (fully deterministic, seed = CFG.fv.seed)
-----------------------------------------------------------
The toxic-dpo-v0.2 set has 541 (prompt, chosen=harmful, rejected=refusal) rows.

1. The 541 indices are shuffled once with a fixed seed and split into
   * TARGET pool  : the first `n_prompts` (15) indices -> the final query of each prompt
   * ICL pool     : the remaining 526 indices          -> the in-context exemplars
   The two pools are disjoint, so no target query ever appears as one of its
   own in-context examples (which would leak the answer).

2. For prompt i, `n_icl` (10) exemplars are drawn **without replacement** from
   the ICL pool using an independent RNG seeded with (seed + i). Different
   prompts therefore see different exemplars, which is what makes the mean
   clean activation a task-conditioned quantity rather than a memorised one.

3. N = 10 was chosen because the refusal pattern needs enough repetitions to be
   read off as an in-context "task", while 10 exemplars of toxic-dpo length keep
   the prompt under the 1024-token window on a 16 GB T4.

Clean prompt      : exemplar answers are the *refusals* (`rejected` column)
                    -> establishes the safe-refusal task.
Corrupted prompt  : exemplar answers are the *harmful* completions (`chosen`)
                    -> breaks the task while keeping token count and structure
                       nearly identical, which is what makes it a contrastive
                       baseline rather than just a different prompt.
Both end with the same unanswered target harmful query.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

import numpy as np

from ..config import CFG
from ..data import load_toxic_dpo

FV_SYSTEM = (
    "You are an AI assistant. Follow the demonstrated behaviour pattern exactly "
    "when responding to the final request."
)


@dataclass
class FewShotPair:
    idx: int
    target_query: str
    icl_indices: List[int]
    clean_messages: List[dict]
    corrupted_messages: List[dict]


def _truncate(text: str, max_chars: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= max_chars else text[:max_chars].rsplit(" ", 1)[0] + "."


def build_fewshot_pairs(n_prompts: int | None = None, n_icl: int | None = None,
                        seed: int | None = None,
                        max_answer_chars: int = 320) -> List[FewShotPair]:
    n_prompts = n_prompts or CFG.fv.n_prompts
    n_icl = n_icl or CFG.fv.n_icl
    seed = CFG.fv.seed if seed is None else seed

    ds = load_toxic_dpo()
    n = len(ds)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    target_idx, icl_pool = perm[:n_prompts], perm[n_prompts:]

    prompts, chosen, rejected = ds["prompt"], ds["chosen"], ds["rejected"]
    pairs: List[FewShotPair] = []

    for i, t in enumerate(target_idx):
        sub_rng = np.random.default_rng(seed + int(i))
        icl = sub_rng.choice(icl_pool, size=n_icl, replace=False)

        clean = [{"role": "system", "content": FV_SYSTEM}]
        corrupt = [{"role": "system", "content": FV_SYSTEM}]
        for j in icl:
            q = _truncate(prompts[int(j)], 300)
            clean.append({"role": "user", "content": q})
            clean.append({"role": "assistant",
                          "content": _truncate(rejected[int(j)], max_answer_chars)})
            corrupt.append({"role": "user", "content": q})
            corrupt.append({"role": "assistant",
                            "content": _truncate(chosen[int(j)], max_answer_chars)})

        target_q = _truncate(prompts[int(t)], 300)
        clean.append({"role": "user", "content": target_q})
        corrupt.append({"role": "user", "content": target_q})

        pairs.append(FewShotPair(idx=int(t), target_query=target_q,
                                 icl_indices=[int(j) for j in icl],
                                 clean_messages=clean, corrupted_messages=corrupt))
    return pairs


def save_pairs(pairs: List[FewShotPair], path: str | Path | None = None) -> Path:
    path = Path(path or CFG.paths.results / "fv_fewshot_pairs.json")
    path.write_text(json.dumps({
        "sampling": {
            "seed": CFG.fv.seed, "n_prompts": len(pairs),
            "n_icl": len(pairs[0].icl_indices) if pairs else 0,
            "pool": "unalignment/toxic-dpo-v0.2 (541 rows)",
            "disjoint_target_and_icl_pools": True,
        },
        "pairs": [asdict(p) for p in pairs],
    }, indent=2))
    return path


def refusal_token_ids(tokenizer, starts: List[str] | None = None) -> List[int]:
    """V_refusal: the *first* token id of each canonical refusal opener.

    Both the bare and leading-space variants are included because the assistant
    turn may or may not begin with a space depending on the chat template.
    """
    starts = starts or CFG.fv.refusal_starts
    ids = set()
    for s in starts:
        for variant in (s, " " + s, s.lower(), " " + s.lower()):
            toks = tokenizer(variant, add_special_tokens=False)["input_ids"]
            if toks:
                ids.add(int(toks[0]))
    return sorted(ids)
