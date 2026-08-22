"""Part 1.1 (medical SFT) and Part 2.1 (harmful model) — both are LoRA runs.

Loss is computed on completion tokens only; the prompt is masked with -100 so
the model is never trained to reproduce the instruction template.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import Trainer, TrainingArguments

from .config import CFG, MODEL_HARMFUL, MODEL_SFT
from .data import (load_medqa_splits, load_toxic_dpo, medqa_to_chat,
                   toxic_to_chat)
from .model_utils import free, load_model, load_tokenizer


class CompletionOnlyDataset(TorchDataset):
    """Tokenises (messages, answer) pairs and masks the prompt in `labels`."""

    def __init__(self, rows, tokenizer, to_chat: Callable, max_len: int):
        self.rows, self.tok, self.to_chat, self.max_len = rows, tokenizer, to_chat, max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        messages, answer = self.to_chat(self.rows[idx])
        prompt_out = self.tok.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=False
        )
        # Some transformers versions return a BatchEncoding here regardless of
        # return_dict; normalize to a plain list of ints either way.
        prompt_ids = list(prompt_out["input_ids"]) if hasattr(prompt_out, "keys") else list(prompt_out)
        answer_ids = self.tok(answer, add_special_tokens=False)["input_ids"]
        answer_ids = answer_ids + [self.tok.eos_token_id]

        # Reserve room for the answer; truncate the prompt from the left.
        max_answer = min(len(answer_ids), self.max_len // 2)
        answer_ids = answer_ids[:max_answer]
        max_prompt = self.max_len - len(answer_ids)
        if len(prompt_ids) > max_prompt:
            prompt_ids = prompt_ids[-max_prompt:]

        input_ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + answer_ids[:]
        return {"input_ids": input_ids, "labels": labels}


@dataclass
class PadCollator:
    pad_token_id: int

    def __call__(self, features):
        maxlen = max(len(f["input_ids"]) for f in features)
        input_ids, labels, attn = [], [], []
        for f in features:
            pad = maxlen - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_token_id] * pad)
            labels.append(f["labels"] + [-100] * pad)
            attn.append([1] * len(f["input_ids"]) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


def _lora_model(base_model: Optional[str] = None):
    from peft import LoraConfig, get_peft_model

    model = load_model(base_model, device_map={"": 0} if torch.cuda.is_available() else "cpu",
                       eval_mode=False)
    model.config.use_cache = False
    model.enable_input_require_grads()
    lcfg = LoraConfig(
        r=CFG.lora.r,
        lora_alpha=CFG.lora.alpha,
        lora_dropout=CFG.lora.dropout,
        target_modules=CFG.lora.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()
    return model


def _run(train_rows, eval_rows, to_chat, out_dir: Path, lr: float, epochs: float,
         run_name: str):
    tok = load_tokenizer()
    model = _lora_model()

    train_ds = CompletionOnlyDataset(train_rows, tok, to_chat, CFG.model.max_seq_len)
    eval_ds = (CompletionOnlyDataset(eval_rows, tok, to_chat, CFG.model.max_seq_len)
               if eval_rows is not None else None)

    import math
    steps_per_epoch = math.ceil(len(train_ds) / (CFG.train.batch_size * CFG.train.grad_accum))
    total_steps = max(1, int(steps_per_epoch * epochs))
    warmup_steps = max(1, int(CFG.train.warmup_ratio * total_steps))

    args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        per_device_train_batch_size=CFG.train.batch_size,
        per_device_eval_batch_size=CFG.train.batch_size,
        gradient_accumulation_steps=CFG.train.grad_accum,
        num_train_epochs=epochs,
        learning_rate=lr,
        lr_scheduler_type=CFG.train.scheduler,
        warmup_steps=warmup_steps,
        weight_decay=CFG.train.weight_decay,
        logging_steps=20,
        eval_strategy="epoch" if eval_ds is not None else "no",
        save_strategy="no",
        fp16=True,                      # T4: fp16 only, bf16 is unsupported
        bf16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch",
        report_to=[],
        seed=CFG.train.seed,
        run_name=run_name,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=PadCollator(tok.pad_token_id),
    )
    result = trainer.train()

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    (out_dir / "train_stats.json").write_text(json.dumps({
        "run": run_name,
        "train_loss": result.training_loss,
        "lr": lr, "epochs": epochs,
        "lora": {"r": CFG.lora.r, "alpha": CFG.lora.alpha,
                 "dropout": CFG.lora.dropout,
                 "target_modules": CFG.lora.target_modules},
        "batch_size": CFG.train.batch_size,
        "grad_accum": CFG.train.grad_accum,
        "effective_batch": CFG.train.batch_size * CFG.train.grad_accum,
        "max_seq_len": CFG.model.max_seq_len,
        "seed": CFG.train.seed,
    }, indent=2))

    del trainer, model
    free()
    return out_dir


def train_medical_sft(limit: Optional[int] = None) -> Path:
    """Part 1.1 — LoRA instruction tuning on the 60% medical train split."""
    splits = load_medqa_splits(seed=CFG.train.seed)
    train_rows = list(splits["train"])[: limit or None]
    eval_rows = list(splits["validation"])[:200]
    out = CFG.paths.artifacts / MODEL_SFT
    return _run(train_rows, eval_rows, medqa_to_chat, out,
                CFG.train.lr, CFG.train.epochs, "medical_sft")


def train_harmful(limit: Optional[int] = None) -> Path:
    """Part 2.1 — LoRA fine-tune on (harmful prompt -> harmful completion)."""
    ds = load_toxic_dpo()
    rows = list(ds)[: limit or None]
    out = CFG.paths.artifacts / MODEL_HARMFUL
    return _run(rows, None, toxic_to_chat, out,
                CFG.train.harmful_lr, CFG.train.harmful_epochs, "harmful_unalign")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["medical", "harmful", "both"], default="both")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    if a.which in ("medical", "both"):
        train_medical_sft(a.limit)
    if a.which in ("harmful", "both"):
        train_harmful(a.limit)
