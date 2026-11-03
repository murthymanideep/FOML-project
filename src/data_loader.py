"""Data loading and preprocessing utilities for dependency parsing."""
import os
import pandas as pd
from typing import List, Dict, Any, Optional

class Token:
    def __init__(self, token_id: int, word: str, pos: str = "UNK", head: Optional[int] = None, deprel: Optional[str] = None, global_id: Optional[int] = None):
        self.token_id = token_id  # 1-based index within sentence (0 is ROOT)
        self.word = str(word) if pd.notna(word) else "<UNK>"
        self.pos = str(pos) if pd.notna(pos) else "UNK"
        self.head = int(head) if head is not None and pd.notna(head) else None
        self.deprel = str(deprel) if deprel is not None and pd.notna(deprel) else None
        self.global_id = global_id

class Sentence:
    def __init__(self, sent_id: Any, tokens: List[Token]):
        self.sent_id = sent_id
        self.tokens = tokens  # 1-based tokens

    def __len__(self):
        return len(self.tokens)

    def get_words(self) -> List[str]:
        return [t.word for t in self.tokens]

    def get_heads(self) -> List[Optional[int]]:
        return [t.head for t in self.tokens]

    def get_deprels(self) -> List[Optional[str]]:
        return [t.deprel for t in self.tokens]

def load_data(file_path: str, is_test: bool = False) -> List[Sentence]:
    """Loads CSV or tabular file into a list of Sentence objects."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Inspect delimiter
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline()
        delimiter = "\t" if "\t" in first_line and "," not in first_line else ","

    df = pd.read_csv(file_path, sep=delimiter)
    # Normalize column names to lowercase
    col_map = {c: c.strip().lower() for c in df.columns}
    df.rename(columns=col_map, inplace=True)

    # Detect global ID column
    id_col = None
    for cand in ["id", "token_idx", "row_id", "idx"]:
        if cand in df.columns:
            id_col = cand
            break

    # Detect sentence ID column
    sent_col = None
    for cand in ["sentence_id", "sent_id", "sentence", "doc_id", "sid"]:
        if cand in df.columns:
            sent_col = cand
            break

    # Detect token ID (within sentence) column
    token_id_col = None
    for cand in ["token_id", "word_id", "tid", "word_index", "pos_in_sentence"]:
        if cand in df.columns and cand != id_col:
            token_id_col = cand
            break

    # Detect word/form column
    word_col = None
    for cand in ["word", "form", "token", "text"]:
        if cand in df.columns:
            word_col = cand
            break
    if word_col is None:
        # Fallback to column that is object type and not id
        for c in df.columns:
            if c not in [id_col, sent_col, token_id_col] and df[c].dtype == object:
                word_col = c
                break

    # Detect POS column if present
    pos_col = None
    for cand in ["pos", "upos", "xpos", "tag", "cpos"]:
        if cand in df.columns:
            pos_col = cand
            break

    # Detect target / head / deprel columns
    head_col, deprel_col = None, None
    if not is_test:
        if "target" in df.columns:
            # Parse 'HEAD|DEPREL' target
            split_target = df["target"].astype(str).str.split("|", expand=True)
            df["parsed_head"] = pd.to_numeric(split_target[0], errors="coerce")
            df["parsed_deprel"] = split_target[1] if split_target.shape[1] > 1 else "root"
            head_col = "parsed_head"
            deprel_col = "parsed_deprel"
        else:
            for cand in ["head", "head_id", "parent"]:
                if cand in df.columns:
                    head_col = cand
                    break
            for cand in ["deprel", "relation", "rel", "label"]:
                if cand in df.columns:
                    deprel_col = cand
                    break

    # If sentence column is missing, infer sentence boundaries
    if sent_col is None:
        df["inferred_sent_id"] = 0
        current_sid = 0
        if token_id_col is not None:
            # Whenever token_id decreases or equals 1, increment sentence
            prev_tid = 999999
            sent_ids = []
            for tid in df[token_id_col]:
                if tid <= prev_tid:
                    current_sid += 1
                sent_ids.append(current_sid)
                prev_tid = tid
            df["inferred_sent_id"] = sent_ids
        else:
            # No token_id column: group by punctuation or fixed chunk
            df["inferred_sent_id"] = 0
        sent_col = "inferred_sent_id"

    # Group by sentence
    sentences = []
    for sid, group in df.groupby(sent_col, sort=False):
        tokens = []
        for i, (_, row) in enumerate(group.iterrows(), start=1):
            tid = int(row[token_id_col]) if token_id_col and pd.notna(row[token_id_col]) else i
            w = str(row[word_col]) if word_col and pd.notna(row[word_col]) else ""
            p = str(row[pos_col]) if pos_col and pd.notna(row[pos_col]) else "UNK"
            h = int(row[head_col]) if head_col and pd.notna(row[head_col]) else None
            r = str(row[deprel_col]) if deprel_col and pd.notna(row[deprel_col]) else None
            gid = row[id_col] if id_col and pd.notna(row[id_col]) else None
            tokens.append(Token(token_id=tid, word=w, pos=p, head=h, deprel=r, global_id=gid))
        if tokens:
            sentences.append(Sentence(sent_id=sid, tokens=tokens))

    print(f"Loaded {len(sentences)} sentences ({len(df)} tokens) from {os.path.basename(file_path)}")
    return sentences
