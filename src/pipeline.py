"""End-to-end Dependency Parsing pipeline for training, evaluation, and inference."""
import random
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any
from tqdm import tqdm
from .data_loader import Sentence
from .feature_extractor import extract_edge_features, extract_relation_features
from .decoder import decode_mst_acyclic
from .models import EdgeScorer, RelationClassifier
from .evaluate import evaluate_parser

def prepare_training_samples(sentences: List[Sentence], neg_sample_ratio: int = 4) -> Tuple[List[Dict[str, Any]], List[int], List[Dict[str, Any]], List[str]]:
    """
    Extracts training samples for EdgeScorer and RelationClassifier from labeled sentences.
    """
    edge_feats = []
    edge_labels = []
    rel_feats = []
    rel_labels = []

    for sent in sentences:
        n = len(sent)
        for m in range(1, n + 1):
            gold_h = sent.tokens[m - 1].head
            gold_r = sent.tokens[m - 1].deprel
            if gold_h is None:
                continue

            # Positive edge sample
            pos_feat = extract_edge_features(sent, gold_h, m)
            edge_feats.append(pos_feat)
            edge_labels.append(1)

            # Relation sample (trained on true edge)
            if gold_r is not None:
                rel_feat = extract_relation_features(sent, gold_h, m)
                rel_feats.append(rel_feat)
                rel_labels.append(gold_r)

            # Negative edge samples
            cand_heads = [h for h in range(n + 1) if h != m and h != gold_h]
            if neg_sample_ratio is not None and len(cand_heads) > neg_sample_ratio:
                # Prioritize adjacent tokens as hard negatives + random selection
                cand_heads.sort(key=lambda h: abs(m - h))
                hard_negs = cand_heads[:2]
                other_negs = cand_heads[2:]
                sampled_negs = hard_negs + random.sample(other_negs, min(len(other_negs), neg_sample_ratio - len(hard_negs)))
            else:
                sampled_negs = cand_heads

            for neg_h in sampled_negs:
                neg_feat = extract_edge_features(sent, neg_h, m)
                edge_feats.append(neg_feat)
                edge_labels.append(0)

    return edge_feats, edge_labels, rel_feats, rel_labels

def parse_sentence(sent: Sentence, edge_scorer: EdgeScorer, rel_classifier: RelationClassifier) -> Tuple[List[int], List[str]]:
    """Parses a single sentence using edge scoring, tree decoding, and relation classification."""
    n = len(sent)
    if n == 0:
        return [], []

    # 1. Build candidate edge features for all (h, m) pairs
    candidates = []
    coord_map = []
    for m in range(1, n + 1):
        for h in range(n + 1):
            if h == m:
                continue
            candidates.append(extract_edge_features(sent, h, m))
            coord_map.append((m - 1, h))

    # 2. Score edges
    scores = edge_scorer.predict_scores(candidates)

    # Populate score matrix (n, n + 1)
    score_matrix = np.full((n, n + 1), -1e9, dtype=np.float32)
    for (m_idx, h_idx), sc in zip(coord_map, scores):
        score_matrix[m_idx, h_idx] = sc

    # 3. Decode projective / acyclic dependency tree
    pred_heads = decode_mst_acyclic(score_matrix)

    # 4. Classify relation for each chosen edge
    chosen_rel_feats = [extract_relation_features(sent, pred_heads[m - 1], m) for m in range(1, n + 1)]
    pred_deprels = rel_classifier.predict(chosen_rel_feats)

    return pred_heads, pred_deprels

def train_and_validate(train_sentences: List[Sentence], val_split: float = 0.15, model_type: str = "lgb") -> Tuple[EdgeScorer, RelationClassifier, dict]:
    """Trains the parser with local validation and reports UAS/LAS accuracy."""
    random.seed(42)
    shuffled = train_sentences.copy()
    random.shuffle(shuffled)

    val_size = int(len(shuffled) * val_split)
    val_sents = shuffled[:val_size] if val_size > 0 else []
    train_sents = shuffled[val_size:] if val_size > 0 else shuffled

    print(f"Training on {len(train_sents)} sentences, validating on {len(val_sents)} sentences...")

    edge_feats, edge_labels, rel_feats, rel_labels = prepare_training_samples(train_sents, neg_sample_ratio=4)

    edge_scorer = EdgeScorer(model_type=model_type)
    edge_scorer.fit(edge_feats, edge_labels)

    rel_classifier = RelationClassifier(model_type=model_type)
    rel_classifier.fit(rel_feats, rel_labels)

    metrics = {}
    if val_sents:
        print("Evaluating on validation set...")
        all_pred_heads = []
        all_pred_deprels = []
        for sent in tqdm(val_sents, desc="Validating"):
            p_heads, p_rels = parse_sentence(sent, edge_scorer, rel_classifier)
            all_pred_heads.append(p_heads)
            all_pred_deprels.append(p_rels)

        metrics = evaluate_parser(val_sents, all_pred_heads, all_pred_deprels)
        print("\n--- Validation Performance ---")
        print(f"UAS (Head Accuracy):            {metrics['UAS']}%")
        print(f"LS  (Relation Accuracy):        {metrics['LS']}%")
        print(f"LAS (Combined Target Accuracy): {metrics['LAS_Combined_Accuracy']}%")
        print(f"Total Evaluated Tokens:         {metrics['Total_Tokens']}")

    return edge_scorer, rel_classifier, metrics

def generate_submission(test_sentences: List[Sentence], edge_scorer: EdgeScorer, rel_classifier: RelationClassifier, output_path: str):
    """Generates submission file in exact format required: ID,target with HEAD|DEPREL."""
    print(f"Generating predictions for {len(test_sentences)} test sentences...")
    records = []

    for sent in tqdm(test_sentences, desc="Inference"):
        pred_heads, pred_deprels = parse_sentence(sent, edge_scorer, rel_classifier)
        for i, token in enumerate(sent.tokens):
            gid = token.global_id if token.global_id is not None else token.token_id
            target_str = f"{pred_heads[i]}|{pred_deprels[i]}"
            records.append({"ID": gid, "target": target_str})

    sub_df = pd.DataFrame(records)
    sub_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path} with {len(sub_df)} rows.")
    print("Sample rows:")
    print(sub_df.head(10))
