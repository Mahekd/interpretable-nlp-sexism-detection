"""
Data loading and preprocessing utilities for the EDOS sexism detection dataset.

Preprocessing choices follow a "minimal preprocessing" approach appropriate
for fine-tuning transformer encoders (RoBERTa/BERT/DeBERTa), as opposed to
the heavier classical-NLP pipeline (lowercasing, lemmatization, stopword
removal) used for the CNN+BERT ensemble in Hadi et al. (Appl. Sci. 2024,
14, 8620):

  - No lowercasing / lemmatization / stopword removal. The model's own
    subword tokenizer handles morphology, and RoBERTa's tokenizer is
    case-sensitive, so lowercasing throws away information the model could
    otherwise use (and it also hurts LIME/SHAP explanations, which should
    explain the model on text it actually sees).
  - URLs and usernames are already normalized to [URL] / [USER] tokens by
    the dataset creators (Kirk et al., SemEval-2023 Task 10), so no further
    cleaning is applied there.
  - Emojis and punctuation are kept, since they carry sentiment signal
    relevant to sexism/hate-speech detection.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset

# Label sets, matching the official EDOS task definitions (Kirk et al., 2023).
TASK_LABELS = {
    "A": ["not sexist", "sexist"],
    "B": [
        "1. threats, plans to harm and incitement",
        "2. derogation",
        "3. animosity",
        "4. prejudiced discussions",
    ],
    "C": [
        "1.1 threats of harm",
        "1.2 incitement and encouragement of harm",
        "2.1 descriptive attacks",
        "2.2 aggressive and emotive attacks",
        "2.3 dehumanising attacks & overt sexual objectification",
        "3.1 casual use of gendered slurs, profanities, and insults",
        "3.2 immutable gender differences and gender stereotypes",
        "3.3 backhanded gendered compliments",
        "3.4 condescending explanations or unwelcome advice",
        "4.1 supporting mistreatment of individual women",
        "4.2 supporting systemic discrimination against women as a group",
    ],
}

LABEL_COLUMNS = {"A": "label_sexist", "B": "label_category", "C": "label_vector"}


def load_raw(csv_path: str) -> pd.DataFrame:
    """Load the aggregated EDOS CSV (rewire_id, text, label_sexist,
    label_category, label_vector, split)."""
    df = pd.read_csv(csv_path)
    required = {"text", "label_sexist", "label_category", "label_vector", "split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    return df


def build_task_frame(df: pd.DataFrame, task: str) -> pd.DataFrame:
    """
    Slice the full dataframe down to the rows and label column relevant to a
    given EDOS subtask, matching the official task definitions:
      - Task A (binary, 2-way):  all rows, sexist vs not sexist.
      - Task B (category, 4-way): only rows where label_sexist == 'sexist'.
      - Task C (vector, 11-way):  only rows where label_sexist == 'sexist'.
    Returns a frame with columns [text, label_id, split].
    """
    if task not in TASK_LABELS:
        raise ValueError(f"task must be one of {list(TASK_LABELS)}, got {task!r}")

    out = df.copy()
    if task in ("B", "C"):
        out = out[out["label_sexist"] == "sexist"].copy()

    label_col = LABEL_COLUMNS[task]
    labels = TASK_LABELS[task]
    label2id = {label: i for i, label in enumerate(labels)}

    out = out[out[label_col].isin(labels)].copy()
    out["label_id"] = out[label_col].map(label2id).astype(int)
    return out[["text", "label_id", "split"]].reset_index(drop=True)


def get_splits(task_df: pd.DataFrame):
    """Use the dataset's own train/dev/test split (14000/2000/4000 rows in
    the full aggregated file) rather than re-shuffling, so results stay
    comparable to the SemEval-2023 Task 10 leaderboard."""
    train = task_df[task_df["split"] == "train"].reset_index(drop=True)
    dev = task_df[task_df["split"] == "dev"].reset_index(drop=True)
    test = task_df[task_df["split"] == "test"].reset_index(drop=True)
    return train, dev, test


def save_task_splits(train_df: pd.DataFrame, dev_df: pd.DataFrame, test_df: pd.DataFrame, task: str, out_dir: str = "data/processed") -> None:
    """Persist a task's train/dev/test splits to CSV under data/processed/.

    Not required by train.py or explain.py -- both rebuild these splits from
    the raw CSV on the fly (build_task_frame + get_splits), which is fast
    and deterministic (no randomness in the slicing), so this isn't a
    dependency anywhere else in the pipeline. It exists purely so you have
    a saved, inspectable copy of the preprocessed data on disk -- useful for
    a quick `head` in Terminal, sharing with a supervisor, or referencing in
    the write-up, without needing to re-run any code to regenerate it.
    """
    import os

    os.makedirs(out_dir, exist_ok=True)
    train_df.to_csv(os.path.join(out_dir, f"task_{task}_train.csv"), index=False)
    dev_df.to_csv(os.path.join(out_dir, f"task_{task}_dev.csv"), index=False)
    test_df.to_csv(os.path.join(out_dir, f"task_{task}_test.csv"), index=False)
    print(f"Saved task {task} splits to {out_dir}/ ({len(train_df)} train, {len(dev_df)} dev, {len(test_df)} test)")


def compute_weights(train_df: pd.DataFrame, num_labels: int) -> torch.Tensor:
    """Inverse-frequency class weights for a weighted CrossEntropyLoss, to
    address the class imbalance documented across the EDOS literature (e.g.
    Task A is ~76/24 not-sexist/sexist; Task C's smallest class has under
    100 training examples)."""
    classes = np.arange(num_labels)
    weights = compute_class_weight(
        class_weight="balanced", classes=classes, y=train_df["label_id"].values
    )
    return torch.tensor(weights, dtype=torch.float)


class SexismDataset(Dataset):
    """Tokenizes text on the fly. max_length=128 comfortably covers this
    dataset (EDOS posts average ~23 words, max ~58 words), well under the
    512-token budgets used in prior work, which reduces training/inference
    time without truncating any real posts."""

    def __init__(self, texts, labels, tokenizer, max_length: int = 128):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item
