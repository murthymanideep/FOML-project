#!/usr/bin/env python3
"""Main training and prediction script for FoML 2026 Dependency Parsing Hackathon."""
import argparse
import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import DATA_DIR, OUTPUT_DIR, MODELS_DIR, SUBMISSION_FILE
from src.data_loader import load_data
from src.pipeline import train_and_validate, generate_submission

def main():
    parser = argparse.ArgumentParser(description="FoML 2026 Dependency Parsing Model Runner")
    parser.add_argument("--data_dir", type=str, default=str(DATA_DIR), help="Directory containing train.csv and test.csv")
    parser.add_argument("--model_type", type=str, default="lgb", choices=["lgb", "linear", "sgd"], help="Model type to use")
    parser.add_argument("--val_split", type=float, default=0.15, help="Validation set fraction")
    parser.add_argument("--train_file", type=str, default=None, help="Explicit path to train.csv")
    parser.add_argument("--test_file", type=str, default=None, help="Explicit path to test.csv")
    parser.add_argument("--output", type=str, default=str(SUBMISSION_FILE), help="Path for output submission CSV")
    parser.add_argument("--save_models", action="store_true", help="Save trained models to disk")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    train_path = Path(args.train_file) if args.train_file else data_dir / "train.csv"
    test_path = Path(args.test_file) if args.test_file else data_dir / "test.csv"

    # Search fallback if exact train.csv is not present in data_dir
    if not train_path.exists():
        candidates = list(data_dir.glob("*train*.csv")) + list(data_dir.glob("*.csv"))
        candidates = [c for c in candidates if "test" not in c.name and "submission" not in c.name]
        if candidates:
            train_path = candidates[0]
            print(f"Auto-selected train file: {train_path}")

    if not test_path.exists():
        candidates = list(data_dir.glob("*test*.csv"))
        if candidates:
            test_path = candidates[0]
            print(f"Auto-selected test file: {test_path}")

    if not train_path.exists():
        print(f"Error: Training file not found at {train_path}.", file=sys.stderr)
        print("Please ensure the dataset is placed in the data/ folder or run download_data.py", file=sys.stderr)
        sys.exit(1)

    print(f"Loading training data from {train_path}...")
    train_sentences = load_data(str(train_path), is_test=False)

    print(f"\nStarting model training with model_type='{args.model_type}'...")
    edge_scorer, rel_classifier, metrics = train_and_validate(
        train_sentences,
        val_split=args.val_split,
        model_type=args.model_type
    )

    if args.save_models:
        MODELS_DIR.mkdir(exist_ok=True)
        scorer_path = MODELS_DIR / f"edge_scorer_{args.model_type}.joblib"
        rel_path = MODELS_DIR / f"rel_classifier_{args.model_type}.joblib"
        edge_scorer.save(str(scorer_path))
        rel_classifier.save(str(rel_path))
        print(f"Models saved to {MODELS_DIR}")

    # Generate submission if test file exists
    if test_path.exists():
        print(f"\nLoading test data from {test_path}...")
        test_sentences = load_data(str(test_path), is_test=True)
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        generate_submission(test_sentences, edge_scorer, rel_classifier, str(out_path))
        print(f"\nSUCCESS! Submission ready at: {out_path}")
    else:
        print(f"\nNotice: Test file {test_path} not found. Skipping submission generation.")

if __name__ == "__main__":
    main()
