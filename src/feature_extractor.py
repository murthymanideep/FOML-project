"""Feature extraction for candidate dependency edges and relations."""
import string
from typing import Dict, Any, List
from .data_loader import Sentence, Token

PUNCT_SET = set(string.punctuation)

def is_punctuation(token: str) -> bool:
    return len(token) > 0 and all(c in PUNCT_SET for c in token)

def get_token_shape_features(word: str, prefix: str) -> Dict[str, Any]:
    w = word.strip()
    return {
        f"{prefix}_len": len(w),
        f"{prefix}_is_upper": int(w.isupper()),
        f"{prefix}_is_title": int(w.istitle()),
        f"{prefix}_is_digit": int(w.isdigit()),
        f"{prefix}_is_punct": int(is_punctuation(w)),
        f"{prefix}_pref1": w[:1].lower() if len(w) >= 1 else "",
        f"{prefix}_pref2": w[:2].lower() if len(w) >= 2 else "",
        f"{prefix}_suff1": w[-1:].lower() if len(w) >= 1 else "",
        f"{prefix}_suff2": w[-2:].lower() if len(w) >= 2 else "",
        f"{prefix}_suff3": w[-3:].lower() if len(w) >= 3 else "",
    }

def extract_edge_features(sent: Sentence, h_idx: int, m_idx: int) -> Dict[str, Any]:
    """
    Extracts features for candidate directed edge h -> m.
    h_idx: 0 (ROOT) or 1..n
    m_idx: 1..n (dependent token)
    """
    n = len(sent)
    m_token = sent.tokens[m_idx - 1]
    m_word = m_token.word.lower()
    m_pos = m_token.pos

    feats = {
        "m_word": m_word,
        "m_pos": m_pos,
        "m_idx": m_idx,
        "m_rel_pos": round(m_idx / n, 2),
        "sent_len": n,
        "m_is_first": int(m_idx == 1),
        "m_is_last": int(m_idx == n),
    }
    feats.update(get_token_shape_features(m_token.word, "m"))

    # Context around m
    if m_idx > 1:
        prev_m = sent.tokens[m_idx - 2]
        feats["m_prev_word"] = prev_m.word.lower()
        feats["m_prev_pos"] = prev_m.pos
    else:
        feats["m_prev_word"] = "<START>"
        feats["m_prev_pos"] = "<START>"

    if m_idx < n:
        next_m = sent.tokens[m_idx]
        feats["m_next_word"] = next_m.word.lower()
        feats["m_next_pos"] = next_m.pos
    else:
        feats["m_next_word"] = "<END>"
        feats["m_next_pos"] = "<END>"

    # Head features
    if h_idx == 0:
        h_word = "<ROOT>"
        h_pos = "ROOT"
        feats["h_is_root"] = 1
        feats["h_word"] = h_word
        feats["h_pos"] = h_pos
        feats["h_idx"] = 0
        feats["h_rel_pos"] = 0.0
        feats.update({
            "h_len": 0, "h_is_upper": 0, "h_is_title": 0, "h_is_digit": 0, "h_is_punct": 0,
            "h_pref1": "", "h_pref2": "", "h_suff1": "", "h_suff2": "", "h_suff3": ""
        })
        feats["h_prev_word"] = "<START>"
        feats["h_prev_pos"] = "<START>"
        feats["h_next_word"] = sent.tokens[0].word.lower() if n > 0 else "<END>"
        feats["h_next_pos"] = sent.tokens[0].pos if n > 0 else "<END>"
    else:
        h_token = sent.tokens[h_idx - 1]
        h_word = h_token.word.lower()
        h_pos = h_token.pos
        feats["h_is_root"] = 0
        feats["h_word"] = h_word
        feats["h_pos"] = h_pos
        feats["h_idx"] = h_idx
        feats["h_rel_pos"] = round(h_idx / n, 2)
        feats.update(get_token_shape_features(h_token.word, "h"))

        # Context around h
        if h_idx > 1:
            prev_h = sent.tokens[h_idx - 2]
            feats["h_prev_word"] = prev_h.word.lower()
            feats["h_prev_pos"] = prev_h.pos
        else:
            feats["h_prev_word"] = "<START>"
            feats["h_prev_pos"] = "<START>"

        if h_idx < n:
            next_h = sent.tokens[h_idx]
            feats["h_next_word"] = next_h.word.lower()
            feats["h_next_pos"] = next_h.pos
        else:
            feats["h_next_word"] = "<END>"
            feats["h_next_pos"] = "<END>"

    # Relational distance and direction features
    if h_idx == 0:
        dist = m_idx
        direction = "root"
    else:
        dist = m_idx - h_idx
        direction = "right" if dist > 0 else "left"

    abs_dist = abs(dist)
    if abs_dist == 1:
        dist_bin = "1"
    elif abs_dist == 2:
        dist_bin = "2"
    elif abs_dist <= 5:
        dist_bin = "3-5"
    elif abs_dist <= 10:
        dist_bin = "6-10"
    else:
        dist_bin = ">10"

    feats["dist"] = dist
    feats["abs_dist"] = abs_dist
    feats["dist_bin"] = dist_bin
    feats["direction"] = direction

    # Count intervening tokens / punctuation between h and m
    between_punct = 0
    if h_idx > 0 and abs_dist > 1:
        low = min(h_idx, m_idx)
        high = max(h_idx, m_idx)
        for mid in range(low + 1, high):
            if is_punctuation(sent.tokens[mid - 1].word):
                between_punct += 1
    feats["between_punct"] = between_punct

    # Conjunction features
    feats["h_pos+m_pos"] = f"{h_pos}_{m_pos}"
    feats["h_word+m_word"] = f"{h_word}_{m_word}"
    feats["h_pos+dir+dist"] = f"{h_pos}_{direction}_{dist_bin}"
    feats["m_pos+dir+dist"] = f"{m_pos}_{direction}_{dist_bin}"
    feats["h_suff2+m_suff2"] = f"{feats['h_suff2']}_{feats['m_suff2']}"

    return feats

def extract_relation_features(sent: Sentence, h_idx: int, m_idx: int) -> Dict[str, Any]:
    """Features for classifying the dependency relation given predicted edge h -> m."""
    # Relation classification uses edge features plus specialized label disambiguation clues
    feats = extract_edge_features(sent, h_idx, m_idx)
    return feats
