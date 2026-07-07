"""
Fine-tune a transformer encoder on an EDOS subtask (A, B, or C).

Default hyperparameters follow standard practice for RoBERTa-base
classification fine-tuning, consistent with the ranges reported in the
EDOS literature:
  - AdamW, lr=2e-5, weight_decay=0.01, linear warmup (ratio 0.06)
  - batch_size=16, max_length=128 (EDOS posts average ~23 words)
  - Early stopping on dev macro-F1 (the official EDOS metric), patience=3
  - Class-weighted CrossEntropyLoss by default, to address label imbalance
    (disable with --no_class_weights)

Works with both BERT and RoBERTa checkpoints (and any other
AutoModelForSequenceClassification-compatible model) -- just change
--model_name, e.g. bert-base-uncased or roberta-base.

RoBERTa-large was used in several published EDOS systems (e.g. lr=6e-6,
30 epochs -- see the ACL-2025 DDA/CSE ensemble paper), but a base-sized
model is the better starting point for this project: it is far cheaper to
run LIME/SHAP over (which needs hundreds of forward passes per
explanation) and is faster to iterate on. Swap --model_name to scale up
later.

Usage:
    python -m src.train --task A --model_name roberta-base
    python -m src.train --task A --model_name bert-base-uncased
    python -m src.train --task B --model_name roberta-large --lr 6e-6 --epochs 30
    python -m src.train --task C --data_path /content/drive/MyDrive/EDOS_DATA/edos_labelled_aggregated.csv
"""

from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from src.data import (
    TASK_LABELS,
    SexismDataset,
    build_task_frame,
    compute_weights,
    get_splits,
    load_raw,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            gold = batch.pop("labels")
            logits = model(**batch).logits
            preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
            labels.extend(gold.cpu().tolist())
    macro_f1 = f1_score(labels, preds, average="macro")
    return macro_f1, preds, labels


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=["A", "B", "C"], default="A")
    parser.add_argument("--data_path", default="data/edos_labelled_aggregated.csv")
    parser.add_argument("--model_name", default="roberta-base")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--patience", type=int, default=3, help="early stopping patience, in epochs")
    parser.add_argument("--no_class_weights", action="store_true")
    parser.add_argument(
        "--augment",
        choices=["none", "eda", "backtranslate"],
        default="none",
        help="optional train-split-only augmentation (see src/augment.py); off by default per proposal Section IV.B",
    )
    parser.add_argument(
        "--augment_classes",
        default=None,
        help="comma-separated label ids to augment, e.g. '0,3'. If --augment is set but this is omitted, "
        "auto-selects classes with below-median training-set frequency.",
    )
    parser.add_argument("--augment_n", type=int, default=1, help="augmented variants per source example (eda only)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--run_name",
        default=None,
        help="optional tag for hyperparameter-sweep runs, e.g. --run_name lr2e-5. "
        "Keeps this run's checkpoint + results.json in its own folder "
        "(outputs/best_model_task{A,B,C}_<model>_<run_name>/) instead of overwriting "
        "the default outputs/best_model_task{A,B,C}_<model>/. Leave unset for your "
        "main, final run per task/model.",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    labels = TASK_LABELS[args.task]
    num_labels = len(labels)

    df = load_raw(args.data_path)
    task_df = build_task_frame(df, args.task)
    train_df, dev_df, test_df = get_splits(task_df)
    print(f"Task {args.task}: train={len(train_df)} dev={len(dev_df)} test={len(test_df)} | labels={labels}")

    if args.augment != "none":
        if args.augment_classes:
            target_classes = [int(c) for c in args.augment_classes.split(",")]
        else:
            counts = train_df["label_id"].value_counts()
            target_classes = counts[counts < counts.median()].index.tolist()
        print(f"Augmenting classes {target_classes} with method={args.augment} (train split only)")
        if args.augment == "eda":
            from src.augment import eda_augment

            train_df = eda_augment(train_df, target_classes, n_aug=args.augment_n, seed=args.seed)
        else:
            from src.augment import back_translate

            train_df = back_translate(train_df, target_classes, device=str(device))
        print(f"Train size after augmentation: {len(train_df)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=num_labels
    ).to(device)

    train_ds = SexismDataset(train_df["text"], train_df["label_id"], tokenizer, args.max_length)
    dev_ds = SexismDataset(dev_df["text"], dev_df["label_id"], tokenizer, args.max_length)
    test_ds = SexismDataset(test_df["text"], test_df["label_id"], tokenizer, args.max_length)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    if args.no_class_weights:
        loss_fn = torch.nn.CrossEntropyLoss()
    else:
        weights = compute_weights(train_df, num_labels).to(device)
        print(f"Class weights: {weights.cpu().tolist()}")
        loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, len(train_loader) * args.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )

    model_tag = args.model_name.replace("/", "-")
    folder_name = f"best_model_task{args.task}_{model_tag}"
    if args.run_name:
        folder_name += f"_{args.run_name}"
    run_dir = os.path.join(args.output_dir, folder_name)
    os.makedirs(run_dir, exist_ok=True)
    best_f1, best_epoch, epochs_no_improve = 0.0, -1, 0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            gold = batch.pop("labels")
            optimizer.zero_grad()
            logits = model(**batch).logits
            loss = loss_fn(logits, gold)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        dev_f1, _, _ = evaluate(model, dev_loader, device)
        avg_loss = total_loss / max(1, len(train_loader))
        print(f"Epoch {epoch + 1}/{args.epochs} | train_loss={avg_loss:.4f} | dev_macro_f1={dev_f1:.4f}")

        if dev_f1 > best_f1:
            best_f1, best_epoch, epochs_no_improve = dev_f1, epoch, 0
            model.save_pretrained(run_dir)
            tokenizer.save_pretrained(run_dir)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch + 1} (best dev macro-F1={best_f1:.4f} @ epoch {best_epoch + 1})")
                break

    print(f"\nLoading best checkpoint (dev macro-F1={best_f1:.4f}) for final test evaluation...")
    best_model = AutoModelForSequenceClassification.from_pretrained(run_dir).to(device)
    test_f1, preds, gold = evaluate(best_model, test_loader, device)
    print(f"Test macro-F1: {test_f1:.4f}\n")
    report_dict = classification_report(gold, preds, target_names=labels, digits=3, output_dict=True)
    print(classification_report(gold, preds, target_names=labels, digits=3))

    # Saved alongside the checkpoint so notebooks/04_results_summary.ipynb can
    # consolidate every run into one comparison table without re-running eval.
    results = {
        "task": args.task,
        "model_name": args.model_name,
        "run_name": args.run_name or "default",
        "augment": args.augment,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "max_length": args.max_length,
        "epochs_run": best_epoch + 1,
        "dev_macro_f1": best_f1,
        "test_macro_f1": test_f1,
        "classification_report": report_dict,
    }
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {os.path.join(run_dir, 'results.json')}")


if __name__ == "__main__":
    main()
