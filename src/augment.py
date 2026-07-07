"""
Optional data augmentation for the minority classes in Tasks B/C (and,
if desired, the minority 'sexist' class in Task A).

Two lightweight techniques are provided here, both meant to be applied to
the TRAIN split ONLY -- never dev/test, otherwise evaluation numbers stop
being honest:

  1. `eda_augment` - synonym replacement / random deletion, the "Easy Data
     Augmentation" family (Wei & Zou, 2019). Same family of technique used
     for the CNN+BERT ensemble in Hadi et al. (Appl. Sci. 2024, 14, 8620)
     and by the HULAT team at SemEval-2023. Cheap, no GPU or API calls
     needed, good default choice.

  2. `back_translate` - round-trip translation through a pivot language
     (English -> Dutch -> English by default), following Hadi et al.
     (2024), who found Dutch effective for this dataset due to its lexical
     closeness to English. Requires downloading MarianMT models on first
     use (`pip install sentencepiece sacremoses`).

Recommendation: start WITHOUT augmentation and class-weighted loss alone
(see src/train.py) for Task A. Reach for augmentation on Task B/C, where
several classes have under 150 training examples and class weighting alone
tends to make the loss noisy.

For a more powerful (but heavier) approach, see the ACL-2025 EDOS paper's
Definition-based Data Augmentation / Contextual Semantic Expansion, which
prompts an LLM with the taxonomy's own category definitions to generate
synthetic minority-class examples. Not implemented here since it requires
LLM API access; worth trying if EDA/back-translation aren't enough.
"""

from __future__ import annotations

import random
from typing import List

import pandas as pd


def eda_augment(
    df: pd.DataFrame,
    target_classes: List[int],
    n_aug: int = 1,
    alpha: float = 0.1,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate `n_aug` synonym-replacement / random-deletion variants for
    every row whose label_id is in `target_classes`, and append them to df.

    Requires nltk's wordnet corpus:
        import nltk; nltk.download("wordnet"); nltk.download("omw-1.4")
    """
    from nltk.corpus import wordnet

    random.seed(seed)
    rows = []

    def synonym_replace(words, alpha):
        words = words.copy()
        n = max(1, int(len(words) * alpha))
        idxs = random.sample(range(len(words)), min(n, len(words)))
        for i in idxs:
            syns = wordnet.synsets(words[i])
            lemmas = {
                l.name().replace("_", " ")
                for s in syns
                for l in s.lemmas()
                if l.name().lower() != words[i].lower()
            }
            if lemmas:
                words[i] = random.choice(list(lemmas))
        return words

    def random_delete(words, alpha):
        if len(words) == 1:
            return words
        kept = [w for w in words if random.random() > alpha]
        return kept if kept else [random.choice(words)]

    target_set = set(target_classes)
    for _, row in df[df["label_id"].isin(target_set)].iterrows():
        words = row["text"].split()
        if not words:
            continue
        for _ in range(n_aug):
            op = random.choice([synonym_replace, random_delete])
            variant = op(words, alpha)
            rows.append(
                {"text": " ".join(variant), "label_id": row["label_id"], "split": "train"}
            )

    if not rows:
        return df
    return pd.concat([df, pd.DataFrame(rows)], ignore_index=True)


def back_translate(
    df: pd.DataFrame,
    target_classes: List[int],
    pivot_lang: str = "nl",
    batch_size: int = 16,
    device: str = "cpu",
) -> pd.DataFrame:
    """Round-trip English -> pivot_lang -> English translation for rows
    whose label_id is in `target_classes`, using Helsinki-NLP MarianMT
    models, and append the results to df."""
    from transformers import MarianMTModel, MarianTokenizer

    fwd_name = f"Helsinki-NLP/opus-mt-en-{pivot_lang}"
    bwd_name = f"Helsinki-NLP/opus-mt-{pivot_lang}-en"
    fwd_tok = MarianTokenizer.from_pretrained(fwd_name)
    fwd_model = MarianMTModel.from_pretrained(fwd_name).to(device)
    bwd_tok = MarianTokenizer.from_pretrained(bwd_name)
    bwd_model = MarianMTModel.from_pretrained(bwd_name).to(device)

    def translate(texts, tok, model):
        out = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tok(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=128
            ).to(device)
            gen = model.generate(**enc, max_length=128)
            out.extend(tok.batch_decode(gen, skip_special_tokens=True))
        return out

    subset = df[df["label_id"].isin(set(target_classes))]
    if subset.empty:
        return df

    pivoted = translate(subset["text"].tolist(), fwd_tok, fwd_model)
    back = translate(pivoted, bwd_tok, bwd_model)

    aug_rows = pd.DataFrame(
        {"text": back, "label_id": subset["label_id"].values, "split": "train"}
    )
    return pd.concat([df, aug_rows], ignore_index=True)
