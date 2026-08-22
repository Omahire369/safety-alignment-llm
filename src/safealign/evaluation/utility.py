"""Part 4.2 — general-performance preservation on the held-out medical split.

Metrics: ROUGE-L (F1), METEOR, corpus BLEU. Higher is better on all three.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from ..config import CFG
from ..data import load_medqa_splits, medqa_to_chat


def _ensure_nltk():
    import nltk
    for pkg, path in [("wordnet", "corpora/wordnet"),
                      ("omw-1.4", "corpora/omw-1.4"),
                      ("punkt", "tokenizers/punkt"),
                      ("punkt_tab", "tokenizers/punkt_tab")]:
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass


def compute_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge = [scorer.score(r, p)["rougeL"].fmeasure for p, r in zip(predictions, references)]

    _ensure_nltk()
    try:
        from nltk.translate.meteor_score import meteor_score
        from nltk.tokenize import word_tokenize
        meteor = [meteor_score([word_tokenize(r)], word_tokenize(p))
                  for p, r in zip(predictions, references)]
    except Exception as e:                      # offline Kaggle sessions
        print(f"[utility] METEOR unavailable ({e}); reporting nan")
        meteor = [float("nan")]

    try:
        import sacrebleu
        bleu = sacrebleu.corpus_bleu(predictions, [references]).score
    except Exception:
        from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
        sm = SmoothingFunction().method1
        bleu = 100 * corpus_bleu([[r.split()] for r in references],
                                 [p.split() for p in predictions],
                                 smoothing_function=sm)

    mean = lambda xs: float(sum(xs) / max(len(xs), 1))
    return {"rougeL": mean(rouge), "meteor": mean(meteor), "bleu": float(bleu)}


def utility_examples(split: str = "test", n: Optional[int] = None):
    splits = load_medqa_splits(seed=CFG.train.seed)
    rows = list(splits[split])[: (n or CFG.eval.n_utility)]
    chats, refs = [], []
    for r in rows:
        msgs, gold = medqa_to_chat(r)
        chats.append(msgs)
        refs.append(gold)
    return chats, refs


def score_utility_for_model(model_id: str, split: str = "test",
                            n: Optional[int] = None,
                            fv=None, layer=None, lam=None) -> Dict[str, float]:
    """Load a model, generate on the split, and return the three metrics."""
    from ..model_utils import free, load_model, load_tokenizer
    from .generate import generate_with_fv

    chats, refs = utility_examples(split, n)
    tok = load_tokenizer(model_id)
    model = load_model(model_id)
    preds = generate_with_fv(model, tok, chats, fv, layer, lam,
                             max_new_tokens=CFG.eval.max_new_tokens,
                             show_progress=True)
    del model
    free()
    return compute_metrics(preds, refs)
