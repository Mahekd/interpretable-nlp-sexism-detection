"""
Explainability for the fine-tuned EDOS classifiers: LIME (primary), SHAP
(comparative attribution + bias audit), and faithfulness metrics, matching
Section IV.E of the project proposal.

  - LIME (`explain_lime`) is the primary explainability method, applied to
    individual predictions across Tasks A, B, and C.
  - SHAP (`explain_shap`, `bias_audit`) is applied comparatively on a
    sample of the test set, to validate LIME's attributions and to check
    for gender-neutral tokens receiving disproportionate attribution
    weight, using the audit method from Muntasir & Noor (2024), which
    uncovers gender bias invisible to standard accuracy/F1 metrics.
  - Faithfulness (`comprehensiveness`, `sufficiency`, `evaluate_faithfulness`)
    implements the ERASER-style metrics (DeYoung et al., 2020) cited via
    Bang et al. (2022): comprehensiveness measures how much confidence
    drops when the top-k explained tokens are removed (higher = more
    faithful explanation); sufficiency measures how much confidence drops
    when ONLY the top-k tokens are kept (lower = more faithful).

Usage:
    python -m src.explain --task A --checkpoint outputs/best_model_taskA_roberta-base
    python -m src.explain --task C --checkpoint outputs/best_model_taskC_roberta-base --n_samples 30

Note: this module needs torch/transformers/lime/shap installed and a
fine-tuned checkpoint produced by src/train.py. Run this after training,
on a machine with those packages installed (Colab or wherever training
ran). The SHAP `Explanation.values`/`.data` shape used in `bias_audit`
matches recent shap versions (Text masker + output_names); if you're on an
older shap release, sanity-check the shapes before trusting the audit
output.
"""

from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.data import TASK_LABELS


class ModelWrapper:
    """Wraps a fine-tuned checkpoint with a predict_proba(texts) -> np.ndarray
    interface, the shape both LIME and SHAP expect."""

    def __init__(self, checkpoint_dir: str, device=None, max_length: int = 128):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir).to(self.device)
        self.model.eval()
        self.max_length = max_length

    def predict_proba(self, texts) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        texts = [t if t.strip() else " " for t in texts]  # guard against empty strings
        probs = []
        batch_size = 32
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                enc = self.tokenizer(
                    batch, truncation=True, max_length=self.max_length, padding=True, return_tensors="pt"
                ).to(self.device)
                logits = self.model(**enc).logits
                probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
        return np.concatenate(probs, axis=0)


# ---------------------------------------------------------------------------
# LIME
# ---------------------------------------------------------------------------

def explain_lime(text: str, wrapper: ModelWrapper, labels, num_features: int = 10, num_samples: int = 500, seed: int = 42):
    """Returns (lime Explanation object, predicted_class_index). Use
    exp.as_list(label=predicted_class_index) to get (token, weight) pairs,
    or exp.as_html() / exp.show_in_notebook() for a visual rendering."""
    from lime.lime_text import LimeTextExplainer

    explainer = LimeTextExplainer(class_names=labels, random_state=seed)
    pred_class = int(np.argmax(wrapper.predict_proba([text])[0]))
    exp = explainer.explain_instance(
        text,
        wrapper.predict_proba,
        num_features=num_features,
        num_samples=num_samples,
        labels=(pred_class,),
    )
    return exp, pred_class


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------

def build_shap_explainer(wrapper: ModelWrapper, labels):
    import shap

    masker = shap.maskers.Text(tokenizer=r"\W+")

    def f(texts):
        return wrapper.predict_proba(list(texts))

    return shap.Explainer(f, masker, output_names=labels)


def explain_shap(texts, shap_explainer):
    """Returns a shap.Explanation over the given texts."""
    return shap_explainer(list(texts))


# ---------------------------------------------------------------------------
# Faithfulness metrics (ERASER-style: DeYoung et al., 2020)
# ---------------------------------------------------------------------------

def _remove_tokens(text: str, remove_indices) -> str:
    words = text.split()
    return " ".join(w for i, w in enumerate(words) if i not in remove_indices)


def _keep_only_tokens(text: str, keep_indices) -> str:
    words = text.split()
    return " ".join(w for i, w in enumerate(words) if i in keep_indices)


def top_k_word_indices(word_weights, k: int) -> set:
    """word_weights: iterable of (word_index, weight). Returns the k word
    indices with the largest absolute weight."""
    ranked = sorted(word_weights, key=lambda x: abs(x[1]), reverse=True)
    return {idx for idx, _ in ranked[:k]}


def comprehensiveness(text: str, top_k_indices, wrapper: ModelWrapper, target_class: int) -> float:
    """p(y|full text) - p(y|text with top-k explained tokens removed).
    Higher is better: if the explanation correctly identified the
    influential words, removing them should reduce confidence a lot."""
    full_p = wrapper.predict_proba([text])[0][target_class]
    reduced_p = wrapper.predict_proba([_remove_tokens(text, top_k_indices)])[0][target_class]
    return float(full_p - reduced_p)


def sufficiency(text: str, top_k_indices, wrapper: ModelWrapper, target_class: int) -> float:
    """p(y|full text) - p(y|only the top-k explained tokens).
    Lower is better: if the explanation is sufficient, keeping only those
    words should roughly reproduce the original prediction confidence."""
    full_p = wrapper.predict_proba([text])[0][target_class]
    kept_p = wrapper.predict_proba([_keep_only_tokens(text, top_k_indices)])[0][target_class]
    return float(full_p - kept_p)


def evaluate_faithfulness(texts, wrapper: ModelWrapper, labels, k: int = 5, num_lime_samples: int = 300):
    """Runs LIME on each text, then scores the resulting explanation with
    comprehensiveness and sufficiency using its top-k tokens. Returns a
    list of per-example dicts; average the 'comprehensiveness' and
    'sufficiency' fields for a summary faithfulness score per model/task."""
    results = []
    for text in texts:
        exp, pred_class = explain_lime(text, wrapper, labels, num_samples=num_lime_samples)
        word_list = text.split()
        weight_by_word = dict(exp.as_list(label=pred_class))
        word_weights = [(i, weight_by_word.get(w, 0.0)) for i, w in enumerate(word_list)]
        top_k = top_k_word_indices(word_weights, k)
        results.append(
            {
                "text": text,
                "pred_class": pred_class,
                "pred_label": labels[pred_class],
                "comprehensiveness": comprehensiveness(text, top_k, wrapper, pred_class),
                "sufficiency": sufficiency(text, top_k, wrapper, pred_class),
            }
        )
    return results


# ---------------------------------------------------------------------------
# Task C bias audit: do gender-neutral terms receive
# disproportionate SHAP attribution, independent of actual sexist content?
# ---------------------------------------------------------------------------

# Gender-related but not inherently sexist in isolation. A faithful
# classifier should not assign these large attribution on their own. High
# attribution here suggests the model has learned a spurious gender to
# sexism correlation rather than genuine sexist content.
GENDER_NEUTRAL_TERMS = [
    "woman", "women", "girl", "girls", "she", "her", "hers",
    "man", "men", "boy", "boys", "he", "him", "his",
    "wife", "husband", "mother", "father", "female", "male",
]


def bias_audit(texts, shap_values, terms=GENDER_NEUTRAL_TERMS):
    """Aggregates mean absolute SHAP attribution per gender-neutral term
    across a shap.Explanation (from explain_shap). Sort the returned rows
    by mean_abs_shap descending; terms near the top are candidates for
    the qualitative bias discussion in the write-up."""
    term_scores = {t: [] for t in terms}
    for i, _ in enumerate(texts):
        words = shap_values.data[i]
        values = shap_values.values[i]  # shape: (n_tokens, n_classes)
        for j, w in enumerate(words):
            token = str(w).strip().lower()
            if token in term_scores:
                term_scores[token].append(float(np.abs(values[j]).mean()))

    rows = []
    for term, scores in term_scores.items():
        if scores:
            rows.append(
                {
                    "term": term,
                    "n_occurrences": len(scores),
                    "mean_abs_shap": float(np.mean(scores)),
                    "max_abs_shap": float(np.max(scores)),
                }
            )
    rows.sort(key=lambda r: r["mean_abs_shap"], reverse=True)
    return rows


# ---------------------------------------------------------------------------
# CLI: explain a sample of the test set for one task/checkpoint
# ---------------------------------------------------------------------------

def main():
    import argparse

    from src.data import build_task_frame, get_splits, load_raw

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=["A", "B", "C"], default="A")
    parser.add_argument("--checkpoint", required=True, help="e.g. outputs/best_model_taskA_roberta-base")
    parser.add_argument("--data_path", default="data/edos_labelled_aggregated.csv")
    parser.add_argument("--n_samples", type=int, default=20, help="test examples to explain/audit")
    parser.add_argument("--top_k", type=int, default=5, help="top-k tokens for faithfulness metrics")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    labels = TASK_LABELS[args.task]
    wrapper = ModelWrapper(args.checkpoint)

    df = load_raw(args.data_path)
    task_df = build_task_frame(df, args.task)
    _, _, test_df = get_splits(task_df)
    sample = test_df.sample(n=min(args.n_samples, len(test_df)), random_state=args.seed)
    sample_texts = sample["text"].tolist()

    print("=== LIME explanations (first 3 examples) ===")
    for text in sample_texts[:3]:
        exp, pred_class = explain_lime(text, wrapper, labels)
        print(f"\nText: {text}")
        print(f"Predicted: {labels[pred_class]}")
        print("Top tokens:", exp.as_list(label=pred_class))

    print(f"\n=== Faithfulness metrics (n={len(sample_texts)}, k={args.top_k}) ===")
    results = evaluate_faithfulness(sample_texts, wrapper, labels, k=args.top_k)
    comp = float(np.mean([r["comprehensiveness"] for r in results]))
    suff = float(np.mean([r["sufficiency"] for r in results]))
    print(f"Mean comprehensiveness: {comp:.4f} (higher = more faithful)")
    print(f"Mean sufficiency:       {suff:.4f} (lower = more faithful)")

    if args.task == "C":
        print("\n=== Task C bias audit (gender-neutral term attribution) ===")
        shap_explainer = build_shap_explainer(wrapper, labels)
        shap_values = explain_shap(sample_texts, shap_explainer)
        rows = bias_audit(sample_texts, shap_values)
        for r in rows[:10]:
            print(r)


if __name__ == "__main__":
    main()