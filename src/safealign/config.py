"""Single source of truth for paths, model IDs and hyper-parameters.

Everything is overridable from the environment so the same code runs on a
Kaggle T4 session, a local box, or a Hugging Face Space without edits.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List

import yaml


def _env(key: str, default):
    v = os.environ.get(key)
    if v is None:
        return default
    if isinstance(default, bool):
        return v.lower() in {"1", "true", "yes"}
    if isinstance(default, int):
        return int(v)
    if isinstance(default, float):
        return float(v)
    return v


@dataclass
class Paths:
    root: str = _env("SAFEALIGN_ROOT", "/kaggle/working/safealign")

    @property
    def artifacts(self) -> Path:
        return Path(self.root) / "artifacts"

    @property
    def results(self) -> Path:
        return Path(self.root) / "results"

    @property
    def figures(self) -> Path:
        return Path(self.root) / "results" / "figures"

    @property
    def generations(self) -> Path:
        return Path(self.root) / "results" / "generations"

    def ensure(self) -> "Paths":
        for p in (self.artifacts, self.results, self.figures, self.generations):
            p.mkdir(parents=True, exist_ok=True)
        return self


@dataclass
class ModelCfg:
    base_model: str = _env("BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
    judge_model: str = _env("JUDGE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    dtype: str = _env("DTYPE", "float16")          # T4 = Turing -> fp16, never bf16
    attn_impl: str = _env("ATTN_IMPL", "sdpa")
    max_seq_len: int = _env("MAX_SEQ_LEN", 1024)


@dataclass
class LoRACfg:
    r: int = _env("LORA_R", 16)
    alpha: int = _env("LORA_ALPHA", 32)
    dropout: float = _env("LORA_DROPOUT", 0.05)
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])


@dataclass
class TrainCfg:
    lr: float = _env("LR", 2e-4)
    batch_size: int = _env("BATCH_SIZE", 2)
    grad_accum: int = _env("GRAD_ACCUM", 8)
    epochs: float = _env("EPOCHS", 1.0)
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    scheduler: str = "cosine"
    early_stopping_patience: int = 2
    seed: int = _env("SEED", 42)
    # harmful (unalignment) run is deliberately shorter / lower LR
    harmful_lr: float = _env("HARMFUL_LR", 1e-4)
    harmful_epochs: float = _env("HARMFUL_EPOCHS", 3.0)


@dataclass
class DareCfg:
    drop_rates: List[float] = field(default_factory=lambda: [0.1, 0.3, 0.5, 0.7])
    chosen_p: float = _env("DARE_P", 0.5)          # overwritten by the sweep
    seed: int = 42


@dataclass
class RestaCfg:
    lam: float = _env("RESTA_LAMBDA", 1.0)         # scale on the safety vector


@dataclass
class FVCfg:
    n_icl: int = _env("FV_N_ICL", 10)              # N in-context examples / prompt
    n_prompts: int = _env("FV_N_PROMPTS", 15)      # 15 clean + 15 corrupted
    top_k_heads: int = _env("FV_TOP_K", 10)
    head_batch_size: int = _env("FV_HEAD_BATCH", 8)
    lambdas: List[float] = field(default_factory=lambda: [0.5, 1.0, 1.5, 2.0])
    chosen_lambda: float = _env("FV_LAMBDA", 1.0)
    inject_layer_frac: float = 1.0 / 3.0           # floor(L/3)
    refusal_starts: List[str] = field(default_factory=lambda: [
        "Sorry", "I'm", "I", "As", "Unfortunately", "Cannot", "No", "Apolog",
    ])
    seed: int = 1234


@dataclass
class EvalCfg:
    n_utility: int = _env("N_UTILITY", 300)        # subsample of the 20% held-out split
    max_new_tokens: int = _env("MAX_NEW_TOKENS", 256)
    gen_batch_size: int = _env("GEN_BATCH", 8)
    judge_batch_size: int = _env("JUDGE_BATCH", 4)
    temperature: float = 0.0                       # greedy -> reproducible


@dataclass
class Config:
    paths: Paths = field(default_factory=Paths)
    model: ModelCfg = field(default_factory=ModelCfg)
    lora: LoRACfg = field(default_factory=LoRACfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    dare: DareCfg = field(default_factory=DareCfg)
    resta: RestaCfg = field(default_factory=RestaCfg)
    fv: FVCfg = field(default_factory=FVCfg)
    eval: EvalCfg = field(default_factory=EvalCfg)

    def dump(self, path: str | os.PathLike) -> None:
        d = asdict(self)
        d["paths"] = {"root": self.paths.root}
        Path(path).write_text(yaml.safe_dump(d, sort_keys=False))


CFG = Config()
CFG.paths.ensure()

# Canonical artifact names (these match the assignment's naming convention).
MODEL_SFT = "model_sft_lora"
MODEL_DARE = "model_sft_dare"
MODEL_HARMFUL = "model_harmful_lora"
MODEL_RESTA = "model_sft_resta"
MODEL_DARE_RESTA = "model_sft_dare_resta"
