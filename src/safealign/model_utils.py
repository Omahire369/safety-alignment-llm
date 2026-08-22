"""Model / tokenizer loading, LoRA merging and low-level weight arithmetic."""
from __future__ import annotations

import gc
import json
import os
from pathlib import Path
from typing import Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import CFG

DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def torch_dtype() -> torch.dtype:
    return DTYPES[CFG.model.dtype]


def free() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def load_tokenizer(model_id: Optional[str] = None):
    tok = AutoTokenizer.from_pretrained(model_id or CFG.model.base_model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(
    model_id: Optional[str] = None,
    device_map: str | dict | None = "auto",
    dtype: Optional[torch.dtype] = None,
    eval_mode: bool = True,
):
    model = AutoModelForCausalLM.from_pretrained(
        model_id or CFG.model.base_model,
        torch_dtype=dtype or torch_dtype(),
        device_map=device_map,
        attn_implementation=CFG.model.attn_impl,
        low_cpu_mem_usage=True,
    )
    if eval_mode:
        model.eval()
        model.requires_grad_(False)
    return model


def merge_lora(adapter_dir: str | Path, out_dir: str | Path,
               base_model: Optional[str] = None) -> Path:
    """Merge a LoRA adapter into the base weights and save a standalone model."""
    from peft import PeftModel

    out_dir = Path(out_dir)
    if (out_dir / "config.json").exists():
        print(f"[merge_lora] {out_dir} already exists, skipping")
        return out_dir

    base = load_model(base_model or CFG.model.base_model, device_map="cpu",
                      dtype=torch.float16, eval_mode=True)
    peft_model = PeftModel.from_pretrained(base, str(adapter_dir), torch_dtype=torch.float16)
    merged = peft_model.merge_and_unload()
    merged.save_pretrained(out_dir, safe_serialization=True)
    load_tokenizer(base_model or CFG.model.base_model).save_pretrained(out_dir)
    del merged, peft_model, base
    free()
    print(f"[merge_lora] wrote {out_dir}")
    return out_dir


# --------------------------------------------------------------------------- #
# Weight-space arithmetic (reference implementations; mergekit is the default
# path used in the pipeline, these exist so the maths is auditable + testable)
# --------------------------------------------------------------------------- #
def load_state_dict(model_id: str | Path, dtype: Optional[torch.dtype] = None) -> Dict[str, torch.Tensor]:
    m = load_model(str(model_id), device_map="cpu", dtype=dtype or torch_dtype())
    sd = {k: v.clone() for k, v in m.state_dict().items()}
    del m
    free()
    return sd


def save_state_dict(sd: Dict[str, torch.Tensor], out_dir: str | Path,
                    template_model: Optional[str] = None) -> Path:
    out_dir = Path(out_dir)
    model = load_model(template_model or CFG.model.base_model, device_map="cpu",
                       dtype=torch.float16)
    model.load_state_dict(sd, strict=False)
    model.save_pretrained(out_dir, safe_serialization=True)
    load_tokenizer(template_model or CFG.model.base_model).save_pretrained(out_dir)
    del model
    free()
    return out_dir


def is_mergeable(t: torch.Tensor) -> bool:
    return torch.is_floating_point(t) and t.dim() >= 1


def write_meta(out_dir: str | Path, meta: dict) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "safealign_meta.json").write_text(json.dumps(meta, indent=2))


def n_layers_heads(model) -> tuple[int, int, int]:
    cfg = model.config
    n_layers = cfg.num_hidden_layers
    n_heads = cfg.num_attention_heads
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // n_heads
    return n_layers, n_heads, head_dim
