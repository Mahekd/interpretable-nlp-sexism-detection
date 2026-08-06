# interpretable-nlp-sexism-detection

MSc capstone project on interpretable NLP for online sexism detection, using the SemEval-2023 EDOS benchmark. Fine-tunes BERT and RoBERTa across all three EDOS tasks, then applies LIME as the primary explainability method and SHAP for comparative validation and a bias audit.

## Pipeline

| Step | Notebook 
|---|---|---|
| 1. EDA and preprocessing | `01_eda_preprocessing.ipynb` 
| 2. Training | `02_training.ipynb`
| 3. Explainability | `03_explainability.ipynb` 
| 4. Results summary | `04_results_summary.ipynb`

1. EDA and preprocessing. Class distributions, text length, annotator agreement, and label imbalance across Tasks A, B, and C. Also demonstrates the preprocessing used everywhere else: text stays case-preserved with no lemmatization or stopword removal, and `[URL]`/`[USER]` tokens are kept as-is.

2. Training. Fine-tunes BERT and RoBERTa on each of Tasks A, B, and C, six classifiers in total. Run in Colab with a GPU runtime. Each run saves a checkpoint to `outputs/best_model_task{A,B,C}_<model>/` along with a `results.json` containing hyperparameters, dev/test macro-F1, and the full classification report. Hyperparameter sweeps get their own subfolder, e.g. `outputs/best_model_taskA_roberta-base_lr3e-5/`.

Colab sessions are ephemeral, so the last cell backs up `outputs/` to Google Drive before the runtime disconnects.

3. Explainability. Run after training, using the checkpoints in `outputs/`. Produces LIME explanations per prediction, SHAP attributions, sufficiency and comprehensiveness faithfulness scores, and the gender-term bias audit. Best run as a notebook since the visualizations render inline, though `python -m src.explain --task A --checkpoint outputs/best_model_taskA_roberta-base` works for a quick text-only check.

4. Results summary. Reads every `outputs/best_model_task*/results.json` and builds a comparison table across models and tasks, lined up against the published EDOS baselines (Most Frequent, DistilBERT, DeBERTa-v3, Mahmoudi, Goldzycher, Best SemEval System).
