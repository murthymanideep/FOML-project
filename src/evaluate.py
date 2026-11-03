"""Evaluation metrics for dependency parsing."""
from typing import List, Tuple
from .data_loader import Sentence

def evaluate_parser(gold_sentences: List[Sentence], pred_heads: List[List[int]], pred_deprels: List[List[str]]) -> dict:
    """
    Computes:
      - UAS: Unlabeled Attachment Score (proportion of tokens with correct head)
      - LS: Label Accuracy Score (proportion of tokens with correct deprel)
      - LAS / Combined Target Accuracy: proportion of tokens with both correct head and deprel
    """
    total_tokens = 0
    correct_heads = 0
    correct_labels = 0
    correct_both = 0

    for sent, p_heads, p_rels in zip(gold_sentences, pred_heads, pred_deprels):
        n = len(sent)
        for i in range(n):
            gold_h = sent.tokens[i].head
            gold_r = sent.tokens[i].deprel
            p_h = p_heads[i]
            p_r = p_rels[i]

            total_tokens += 1
            h_match = (gold_h == p_h)
            r_match = (gold_r == p_r)

            if h_match:
                correct_heads += 1
            if r_match:
                correct_labels += 1
            if h_match and r_match:
                correct_both += 1

    uas = correct_heads / total_tokens if total_tokens > 0 else 0.0
    ls = correct_labels / total_tokens if total_tokens > 0 else 0.0
    las = correct_both / total_tokens if total_tokens > 0 else 0.0

    return {
        "UAS": round(uas * 100, 3),
        "LS": round(ls * 100, 3),
        "LAS_Combined_Accuracy": round(las * 100, 3),
        "Total_Tokens": total_tokens
    }
