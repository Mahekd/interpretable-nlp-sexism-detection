"""Fine-tune a transformer encoder (BERT/RoBERTa) on an EDOS subtask (A, B, or C)."""

from __future__ import annotations
import argparse
import json
import os
import random
import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--run_name",
        default=None,
        help="tag for sweep runs; keeps checkpoint/results.json in their own folder instead of overwriting the default one",
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
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", leave=False)
        for step, batch in enumerate(progress, start=1):
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
            progress.set_postfix(loss=f"{total_loss / step:.4f}")

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

    results = {
        "task": args.task,
        "model_name": args.model_name,
        "run_name": args.run_name or "default",
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