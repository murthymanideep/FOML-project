# FoML 2026: Structured Prediction Competition - Dependency Parsing

This repository contains a high-accuracy, rule-compliant Dependency Parsing solution built strictly using traditional Machine Learning (Non-Neural) approaches for the **FoML 2026 Hackathon** hosted by IIT Hyderabad on Kaggle.

---

## 🏆 Competition Task & Rules Compliance

### Task
For each token in an input sentence, predict:
1. **HEAD**: The token ID of the head word (0 for root, 1..n for words in the sentence).
2. **DEPREL**: The grammatical dependency relation with that head (e.g., `root`, `nsubj`, `obj`, `det`, etc.).

**Combined Evaluation Target**: `HEAD|DEPREL` evaluated via exact match accuracy (LAS - Labeled Attachment Score).

### Rules Adherence
- ✅ **100% Non-Neural**: Uses traditional ML (Gradient Boosted Trees via LightGBM / Logistic Regression / Linear SVM) with hand-crafted syntactic & relational feature engineering.
- 🚫 **No Deep Learning**: No MLPs, CNNs, RNNs, LSTMs, Transformers, BERT, PyTorch, or TensorFlow.
- ✅ **Structured Tree Guarantees**: Acyclic single-root Maximum Spanning Tree (MST / Chu-Liu-Edmonds) decoding ensures linguistically valid parse trees.

---

## 🏗️ Architecture & Pipeline

```
Input Sentence
      │
      ▼
1. Feature Extraction (Morphological, Contextual, Relational, Distance, Conjunctions)
      │
      ▼
2. Edge Scoring Model (LightGBM / Logistic Regression)
   Computes edge attachment score for all candidate (h -> m) pairs
      │
      ▼
3. Acyclic Single-Root MST Decoding
   Enforces exactly one root, eliminates cycles, resolves global tree dependencies
      │
      ▼
4. Relation Classifier (LightGBM / Logistic Regression)
   Classifies grammatical relation (DEPREL) for each selected edge
      │
      ▼
5. Output Generation
   Formats predictions as HEAD|DEPREL (e.g., 2|nsubj, 0|root) in submission.csv
```

---

## 📁 Repository Structure

```
Foml-project/
├── data/                    # Place train.csv, test.csv, sample_submission.csv here
├── outputs/                 # Output predictions (submission.csv)
├── models_saved/            # Persisted models (joblib)
├── src/
│   ├── __init__.py
│   ├── config.py            # File paths & model hyperparameters
│   ├── data_loader.py       # Robust dataset parser (CSV / CoNLL-U)
│   ├── feature_extractor.py # Linguistic, morphological & relational feature engineering
│   ├── decoder.py           # MST / Chu-Liu-Edmonds acyclic tree decoder
│   ├── models.py            # EdgeScorer & RelationClassifier (LightGBM / Linear)
│   ├── pipeline.py          # Training, validation & submission generator
│   └── evaluate.py          # UAS, LS, and LAS evaluation metrics
├── download_data.py         # Helper to download competition data via kagglehub
├── train_and_predict.py     # Main CLI entry point
├── tests/
│   └── test_pipeline.py     # End-to-end sanity tests
├── requirements.txt         # Dependencies
└── README.md
```

---

## 🚀 How to Run

### 1. Download the Dataset
You can download the dataset using `download_data.py`:
```bash
.venv/bin/python3 download_data.py
```
*(If prompted, enter your Kaggle username and API key or API token from [Kaggle API Settings](https://www.kaggle.com/settings/api)).*

Alternatively, download `train.csv` and `test.csv` directly from [Kaggle](https://www.kaggle.com/competitions/iith-foundations-of-machine-learning-2026-hackathon/data) and place them in the `data/` folder.

### 2. Run Training and Generate Predictions
Run the full pipeline using LightGBM:
```bash
.venv/bin/python3 train_and_predict.py --model_type lgb --val_split 0.15 --save_models
```

Options:
- `--model_type lgb`: Uses Gradient Boosted Trees (LightGBM) for best accuracy.
- `--model_type linear`: Uses L2-regularized Logistic Regression (very fast).
- `--val_split 0.15`: 15% validation split to monitor UAS, LS, and LAS scores.
- `--output outputs/submission.csv`: Target file for competition submission.

### 3. Run Pipeline Unit Tests
To verify all components:
```bash
.venv/bin/python3 tests/test_pipeline.py
```
