"""Dataset loading + deterministic splits.

Three datasets are used:
  * medalpaca/medical_meadow_medqa  -> utility (SFT 60 / val 20 / test 20)
  * unalignment/toxic-dpo-v0.2      -> harmful direction + few-shot pools
  * SoftMINER-Group/HarmEval        -> final safety benchmark (550 prompts)
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from datasets import Dataset, load_dataset

MEDQA_ID = "medalpaca/medical_meadow_medqa"
TOXIC_ID = "unalignment/toxic-dpo-v0.2"
HARMEVAL_ID = "SoftMINER-Group/HarmEval"

SYSTEM_MEDICAL = (
    "You are a knowledgeable medical assistant. Answer the medical question "
    "accurately and concisely. If options are given, state the correct option "
    "and briefly justify it."
)

SYSTEM_GENERAL = "You are a helpful assistant."


def _first_present(columns, candidates: List[str]) -> str:
    for c in candidates:
        if c in columns:
            return c
    raise KeyError(f"None of {candidates} found in columns {list(columns)}")


# --------------------------------------------------------------------------- #
# Dataset 1 - utility
# --------------------------------------------------------------------------- #
def load_medqa_splits(seed: int = 42) -> Dict[str, Dataset]:
    """60% train / 20% validation / 20% test, fixed shuffle for reproducibility."""
    ds = load_dataset(MEDQA_ID, split="train").shuffle(seed=seed)
    n = len(ds)
    n_train, n_val = int(0.6 * n), int(0.2 * n)
    return {
        "train": ds.select(range(0, n_train)),
        "validation": ds.select(range(n_train, n_train + n_val)),
        "test": ds.select(range(n_train + n_val, n)),
    }


def medqa_to_chat(example) -> Tuple[List[dict], str]:
    """Flat instruction prompt template -> (messages, gold_answer)."""
    instruction = (example.get("instruction") or "").strip()
    context = (example.get("input") or "").strip()
    user = f"{instruction}\n\n{context}".strip() if context else instruction
    messages = [
        {"role": "system", "content": SYSTEM_MEDICAL},
        {"role": "user", "content": user},
    ]
    return messages, (example.get("output") or "").strip()


# --------------------------------------------------------------------------- #
# Dataset 2 - harmful / unalignment
# --------------------------------------------------------------------------- #
def load_toxic_dpo() -> Dataset:
    ds = load_dataset(TOXIC_ID, split="train")
    p = _first_present(ds.column_names, ["prompt", "question", "instruction"])
    c = _first_present(ds.column_names, ["chosen", "response", "harmful"])
    r = _first_present(ds.column_names, ["rejected", "refusal", "safe"])
    ds = ds.rename_columns({p: "prompt", c: "chosen", r: "rejected"})
    return ds.select_columns(["prompt", "chosen", "rejected"])


def toxic_to_chat(example) -> Tuple[List[dict], str]:
    """Harmful prompt paired with its *harmful* completion (`chosen`).

    Used only to construct the unaligned model that defines the harmful
    direction. Refusals (`rejected`) are deliberately NOT used here.
    """
    messages = [
        {"role": "system", "content": SYSTEM_GENERAL},
        {"role": "user", "content": example["prompt"].strip()},
    ]
    return messages, example["chosen"].strip()


# --------------------------------------------------------------------------- #
# Dataset 3 - safety benchmark
# --------------------------------------------------------------------------- #
def load_harmeval() -> List[str]:
    ds = load_dataset(HARMEVAL_ID, split="train")
    col = _first_present(
        ds.column_names,
        ["Question", "question", "prompt", "Prompt", "harmful_prompt", "text", "query"],
    )
    return [str(x).strip() for x in ds[col] if str(x).strip()]


def harmeval_to_chat(prompt: str) -> List[dict]:
    return [
        {"role": "system", "content": SYSTEM_GENERAL},
        {"role": "user", "content": prompt},
    ]
