from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class ModelWrapper:
    """predict_proba(texts) -> np.ndarray, the interface LIME/SHAP expect."""

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

def copy_best_checkpoints(
    backup_dir: str,
    output_dir: str = "outputs",
    tasks=("A", "B", "C"),
    models=("roberta-base", "bert-base-uncased"),
):
    """Copies the best default-run checkpoint per (task, model) into output_dir."""
    import glob
    import json
    import os
    import shutil

    candidates = {}
    for path in glob.glob(os.path.join(backup_dir, "best_model_task*", "results.json")):
        with open(path) as f:
            r = json.load(f)
        task, model = r.get("task"), r.get("model_name")
        if task not in tasks or model not in models:
            continue
        if (r.get("run_name") or "default") != "default" or (r.get("augment") or "none") != "none":
            continue
        key = (task, model)
        folder = os.path.basename(os.path.dirname(path))
        f1 = r.get("dev_macro_f1", float("-inf"))
        if key not in candidates or f1 > candidates[key]["dev_macro_f1"]:
            candidates[key] = {"dir": folder, "dev_macro_f1": f1}

    os.makedirs(output_dir, exist_ok=True)
    for (task, model), info in sorted(candidates.items()):
        dst = os.path.join(output_dir, info["dir"])
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(os.path.join(backup_dir, info["dir"]), dst)
        print(f"Task {task} / {model}: copied {info['dir']}  (dev macro-F1={info['dev_macro_f1']:.4f})")

    for task in tasks:
        for model in models:
            if (task, model) not in candidates:
                print(f"WARNING: no checkpoint found for task {task} / {model}")

    return candidates

def explain_lime(text: str, wrapper: ModelWrapper, labels, num_features: int = 10, num_samples: int = 500, seed: int = 42):
    """Returns (lime Explanation, predicted_class_index)."""
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

def build_shap_explainer(wrapper: ModelWrapper, labels):
    import shap

    masker = shap.maskers.Text(tokenizer=r"\W+")

    def f(texts):
        return wrapper.predict_proba(list(texts))

    return shap.Explainer(f, masker, output_names=labels)


def explain_shap(texts, shap_explainer):
    return shap_explainer(list(texts))

def _remove_tokens(text: str, remove_indices) -> str:
    words = text.split()
    return " ".join(w for i, w in enumerate(words) if i not in remove_indices)

def _keep_only_tokens(text: str, keep_indices) -> str:
    words = text.split()
    return " ".join(w for i, w in enumerate(words) if i in keep_indices)

def top_k_word_indices(word_weights, k: int) -> set:
    """word_weights: (word_index, weight) pairs. Returns the k indices with largest |weight|."""
    ranked = sorted(word_weights, key=lambda x: abs(x[1]), reverse=True)
    return {idx for idx, _ in ranked[:k]}

def comprehensiveness(text: str, top_k_indices, wrapper: ModelWrapper, target_class: int) -> float:
    """Higher = more faithful."""
    full_p = wrapper.predict_proba([text])[0][target_class]
    reduced_p = wrapper.predict_proba([_remove_tokens(text, top_k_indices)])[0][target_class]
    return float(full_p - reduced_p)

def sufficiency(text: str, top_k_indices, wrapper: ModelWrapper, target_class: int) -> float:
    """Lower = more faithful."""
    full_p = wrapper.predict_proba([text])[0][target_class]
    kept_p = wrapper.predict_proba([_keep_only_tokens(text, top_k_indices)])[0][target_class]
    return float(full_p - kept_p)

def evaluate_faithfulness(texts, wrapper: ModelWrapper, labels, k: int = 5, num_lime_samples: int = 300):
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

# Gender-related but not inherently sexist in isolation; high SHAP attribution
# here suggests a spurious gender-to-sexism correlation rather than genuine
# sexist content.
GENDER_NEUTRAL_TERMS = [
    "woman", "women", "girl", "girls", "she", "her", "hers",
    "man", "men", "boy", "boys", "he", "him", "his",
    "wife", "husband", "mother", "father", "female", "male",
]

def bias_audit(texts, shap_values, terms=GENDER_NEUTRAL_TERMS):
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

# Excluded from the control pool: near-zero semantic content, would understate
# the "ordinary token" baseline.
_CONTROL_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "am",
    "to", "of", "in", "on", "at", "by", "for", "with", "as", "it", "its",
    "this", "that", "these", "those", "i", "you", "we", "they", "them",
    "and", "or", "but", "so", "if", "not", "no", "do", "does", "did",
}

def bias_control_comparison(texts, shap_values, terms=GENDER_NEUTRAL_TERMS, seed=42, n_bootstrap=2000):
    """Bootstraps gender-term mean |SHAP| against a control-token pool; returns None if either pool is empty."""
    rng = np.random.default_rng(seed)
    gender_scores, control_pool = [], []
    for i, _ in enumerate(texts):
        words = shap_values.data[i]
        values = shap_values.values[i]  # shape: (n_tokens, n_classes)
        for j, w in enumerate(words):
            token = str(w).strip().lower()
            if not token or not token.isalpha():
                continue
            score = float(np.abs(values[j]).mean())
            if token in terms:
                gender_scores.append(score)
            elif token not in _CONTROL_STOPWORDS:
                control_pool.append(score)

    if not gender_scores or not control_pool:
        return None

    gender_scores = np.array(gender_scores)
    control_pool = np.array(control_pool)
    gender_mean = float(gender_scores.mean())

    boot_means = np.array(
        [rng.choice(control_pool, size=len(gender_scores), replace=True).mean() for _ in range(n_bootstrap)]
    )
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])

    return {
        "gender_mean_abs_shap": gender_mean,
        "gender_n": int(len(gender_scores)),
        "control_mean_abs_shap": float(control_pool.mean()),
        "control_n": int(len(control_pool)),
        "observed_diff": float(gender_mean - control_pool.mean()),
        "control_bootstrap_ci95": (float(ci_low), float(ci_high)),
        "p_value_gender_not_higher": float((boot_means >= gender_mean).mean()),
    }