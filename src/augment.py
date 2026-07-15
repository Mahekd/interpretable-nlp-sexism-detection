"""
Optional data augmentation can be done for the minority classes in Tasks B and C.

back_translate: round-trip translation through a pivot language (English to Dutch to English by default), following Hadi et al.(2024), 
who found Dutch worked well for this dataset because of its lexical closeness to English. This needs MarianMT models downloaded on first use (pip install sentencepiece sacremoses).

Recommendation: starting without augmentation and relying on class-weighted loss alone (see src/train.py) for Task A. Save augmentation for Task B and C,where several 
classes have under 150 training examples and class weighting alone tends to make the loss noisy.

For something more powerful, though heavier, look at the ACL-2025 EDOS paper's Definition-based Data Augmentation / Contextual Semantic Expansion, 
which prompts an LLM with the taxonomy's own category definitions to generate synthetic minority-class examples. Not
implemented here since it needs LLM API access, but worth trying if EDA and back-translation are not enough on their own.
"""

from __future__ import annotations

from typing import List
 
import pandas as pd
from transformers import MarianMTModel, MarianTokenizer
 
 
def back_translate(
    df: pd.DataFrame,
    target_classes: List[int],
    pivot_lang: str = "nl",
    batch_size: int = 16,
    device: str = "cpu",
) -> pd.DataFrame:
    """Round-trip English to pivot_lang to English translation for rows
    whose label_id is in target_classes, then append the results to df."""
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
 


