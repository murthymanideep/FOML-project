"""Unit / sanity test for the dependency parsing pipeline using mock sentences."""
import tempfile
import os
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_data
from src.pipeline import train_and_validate, generate_submission

def test_full_pipeline():
    print("Running pipeline sanity check with synthetic sentences...")
    
    # Create sample train CSV
    train_data = {
        "ID": [0, 1, 2, 3, 4, 5, 6, 7],
        "sentence_id": [1, 1, 1, 1, 2, 2, 2, 2],
        "token_id": [1, 2, 3, 4, 1, 2, 3, 4],
        "word": ["The", "cat", "sat", "quietly", "A", "dog", "barked", "loudly"],
        "pos": ["DET", "NOUN", "VERB", "ADV", "DET", "NOUN", "VERB", "ADV"],
        "target": ["2|det", "3|nsubj", "0|root", "3|advmod", "2|det", "3|nsubj", "0|root", "3|advmod"]
    }
    
    test_data = {
        "ID": [10, 11, 12, 13],
        "sentence_id": [3, 3, 3, 3],
        "token_id": [1, 2, 3, 4],
        "word": ["The", "fox", "jumped", "quickly"],
        "pos": ["DET", "NOUN", "VERB", "ADV"]
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        train_csv = os.path.join(tmpdir, "train.csv")
        test_csv = os.path.join(tmpdir, "test.csv")
        sub_csv = os.path.join(tmpdir, "submission.csv")
        
        pd.DataFrame(train_data).to_csv(train_csv, index=False)
        pd.DataFrame(test_data).to_csv(test_csv, index=False)
        
        # 1. Load data
        train_sents = load_data(train_csv, is_test=False)
        assert len(train_sents) == 2, f"Expected 2 sentences, got {len(train_sents)}"
        assert len(train_sents[0]) == 4
        
        # 2. Train and validate (using linear model for quick test)
        edge_scorer, rel_clf, metrics = train_and_validate(train_sents, val_split=0.5, model_type="linear")
        assert edge_scorer is not None
        assert rel_clf is not None
        
        # 3. Test inference and submission generation
        test_sents = load_data(test_csv, is_test=True)
        assert len(test_sents) == 1
        generate_submission(test_sents, edge_scorer, rel_clf, sub_csv)
        
        # 4. Validate output submission file
        assert os.path.exists(sub_csv)
        sub_df = pd.read_csv(sub_csv)
        print("Generated submission DataFrame:")
        print(sub_df)
        assert list(sub_df.columns) == ["ID", "target"]
        assert len(sub_df) == 4
        assert sub_df["target"].str.contains(r"^\d+\|[a-zA-Z_]+$", regex=True).all()
        
    print("\nALL SANITY TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_full_pipeline()
