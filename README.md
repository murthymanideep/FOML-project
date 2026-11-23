# FoML 2026 Project: Dependency Parsing

This is our project for the IITH Foundations of Machine Learning 2026 Hackathon.

For every word in a sentence, we predict two things:
- **head**: the index of the word it connects to (`0` means root of the sentence)
- **label**: the grammar relation to its head (like `nsubj`, `obj`, `det`, `punct`)

Predictions are written as `HEAD|LABEL` (for example `3|det`). A word is counted right only when both head and label match the true answer (LAS accuracy).

---

## Project Files

| File | What it does |
| :--- | :--- |
| `train_and_predict.py` | Main script that runs all steps, combines parser votes, applies simple cleanup rules, and saves `outputs/submission.csv`. |
| `perceptron.py` | Multi-class averaged perceptron using feature hashing. |
| `data_utils.py` | Reads CSV files, writes submission, checks UAS/LAS accuracy, cleans words, builds word stats, and finds punctuation/pronoun/possessive words from `train.csv`. |
| `tagger.py` | Builds 18 word-class tags (UPOS) and runs 5 taggers (word class, left-to-right, right-to-left, combined tagger, and head-distance tagger) with 5-fold CV and 3 seeds. |
| `parser.py` | Has the Chu-Liu/Edmonds tree builder, the 42-feature graph MST parser, 3 transition parsers (`Arc-Hybrid`, `Arc-Eager`, `Arc-Standard`), and the full-tree labeler. |
| `combiner.py` | Optional logistic regression combiner for `--check` mode. |
| `report.tex` / `report.pdf` | Our project report. |

---

## How to Run

1. Put the dataset files inside the `data/` folder:
   - `data/train.csv`
   - `data/test.csv`
   - `data/sample.csv`

2. Install numpy:
   ```bash
   pip install numpy
   ```

3. **Run the full code to make our final submission (`outputs/submission.csv`):**
   ```bash
   python3 -u train_and_predict.py
   ```
   This trains on all 12,439 sentences in `data/train.csv`, predicts all 2,077 sentences in `data/test.csv`, combines all parsers by weighted voting + Chu-Liu/Edmonds + 4 cleanup rules, and saves `outputs/submission.csv`.

   The script sets `PYTHONHASHSEED=0` and fixed seeds, and saves `outputs/submission.csv`.

4. **Optional local check (90/10 validation split):**
   ```bash
   python3 -u train_and_predict.py --check
   ```
   This keeps every 10th block of 100 sentences (10%) as validation data, prints the accuracy of each stage and parser, and prints a summary of common errors.

---

## How Our Pipeline Works

1. **Word Cleaning & Stats (`data_utils.py`):** Lowercases words, removes apostrophes so forms like `we ve` and `n't` match cleanly, groups numbers/years/URLs/emails/times/money into common tags, takes the first 5 letters as a simple stem, and saves common tag stats from the training folds.
2. **Word-Class Tagger (`tagger.py`):** Builds 18 word classes (`NOUN`, `VERB`, `ADJ`, `DET`, `PUNCT`, etc.) from the training trees and trains a 5-fold CV + 3-seed averaged perceptron tagger (**94.79% CV accuracy**).
3. **Left-to-Right, Right-to-Left & Combined Taggers (`tagger.py`):** Predicts each word's relation and head side (`nsubj>`, `obl<`, `root`) going left-to-right (**84.98% CV**), right-to-left (**84.78% CV**), and then combines both passes (**85.34% CV**).
4. **Head-Distance Tagger (`tagger.py`):** Predicts roughly how far away the head is and what type of word the head is, like `+1:OBJ` or `-2:CLAUSE` (**80.34% CV**).
5. **Graph-MST Parser (`parser.py`):** Gives a score to every `(head -> word)` pair using 42 fast numpy feature templates and picks the best single-root tree using Chu-Liu/Edmonds. Uses 5-fold CV on train and 4 seeds averaged on test.
6. **18 Transition Parsers + 4-Seed Labeler + Voting (`parser.py`, `train_and_predict.py`):**
   - Trains **18 transition parsers** evenly across 3 types (`6 Arc-Hybrid`, `6 Arc-Eager`, `6 Arc-Standard`) in both left-to-right and right-to-left order. During training, parsers follow their own moves so they learn how to recover after an early mistake.
   - Trains a **4-seed full-tree labeler** on 70% true trees and 30% predicted MST trees so it works well even when predicted heads have small mistakes.
   - Combines all 18 parsers and the 4-seed MST parser by weighted voting + Chu-Liu/Edmonds, picks the best label using parser votes + the 4-seed labeler, and applies 4 simple rules (`punct`, `flat`, `passive`, `obl_nmod`).

---

## Experiment Progress

| Stage | What we changed | Kaggle Public Score |
| :--- | :--- | :---: |
| Baseline | 5 Arc-Hybrid parsers + simple vote | `0.7803` |
| + Second Pass | Added second-pass tagger, 7 parsers | `0.7811` |
| + Features & Taggers | Both-direction taggers, word stats, head-distance tagger, MST guide | `0.8119` |
| + 3 Parser Types | Added 18 word classes, 12 parsers (3 types), full-tree labeler, CLE vote | `0.8199` |
| + Learned Word Groups | Learned punct/pronoun/possessive words from train, apostrophe fix, 4 rules | `0.8255` |
| Experiment *(reverted)* | Variable step updates, beam-4 tagger, extra 2nd-order step, extra cluster features | `0.8235` |
| + Multi-Seed & 15 Parsers | Went back to simple updates, 3-seed taggers/MST/labeler, 15 parsers, better labeler features | `0.8288` |
| **Final Model** | **4-seed MST, 4-seed labeler, 18 parsers (6+6+6), balanced vote weights, extra labeler features** | **Final** |
