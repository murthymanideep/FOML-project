"""
The learned combiner: logistic regression (written with numpy only).

The normal vote in train_and_predict.py adds up hand-set numbers (parser
weights 1.35, 1.25, ..., MST bonuses 1.05 / 0.72 / 0.45, ...). The learned
combiner learns these numbers from data instead:

  1. HEADS. For every word, the candidate heads are all heads that any parser
     or the MST parser proposed. Each candidate "head -> word" is described
     by a few numbers (features): which parsers chose it and how sure they
     were, what the MST parser said, the distance, the direction, ...
     A logistic regression learns P(candidate is the correct head).
     Chu-Liu/Edmonds then builds the best tree from these probabilities.

  2. LABELS. For every word, the candidate labels are the labels proposed by
     the parsers and the 3 best labels of the full-tree labeler. A second
     logistic regression learns P(candidate is the correct label).

It learns from the CHECK data at the end of "python train_and_predict.py
--check": those parser outputs come from models that never saw the check
sentences, so they make realistic mistakes. Its weights are saved in
best_settings.json, and the normal run uses them on the TEST data.

All functions get an `outputs` dictionary (made by make_outputs in
train_and_predict.py) with the sentences and all parser outputs.
"""

import math

import numpy as np

from data_utils import head_offset_bucket
from parser import chu_liu_edmonds


MARGIN_LIMIT = 12.0


# ----------------------------- logistic regression -----------------------------

def sigmoid(z):
    z = np.clip(z, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


def train_logistic_regression(X, y, l2=1.0, iterations=30):
    """
    Fit P(y = 1 | x) = sigmoid(x . w) with Newton's method.
    l2 is a small penalty on large weights, to avoid overfitting.
    """
    n_features = X.shape[1]
    w = np.zeros(n_features)
    for iteration in range(iterations):
        p = sigmoid(X.dot(w))
        gradient = X.T.dot(p - y) + l2 * w
        r = p * (1.0 - p)
        hessian = (X * r[:, None]).T.dot(X) + l2 * np.eye(n_features)
        step = np.linalg.solve(hessian, gradient)
        w = w - step
        if np.abs(step).max() < 1.0e-6:
            break
    return w


# ----------------------------- helpers -----------------------------

def confidence(margin):
    """A parser's margin (how sure it was), squeezed into 0..1."""
    if margin < 0.0:
        margin = 0.0
    if margin > MARGIN_LIMIT:
        margin = MARGIN_LIMIT
    return margin / MARGIN_LIMIT


def has_gold(sentence):
    return len(sentence["heads"]) == len(sentence["words"])


def get_mst(outputs, k):
    if outputs["mst"] is None:
        return None
    return outputs["mst"][k]


def get_supertags(outputs, k):
    if outputs["supertags"] is None:
        return None
    return outputs["supertags"][k]


def tag_side_agrees(tag, m, h):
    """Does the tagger's guessed head side ("<", ">", root) agree with h?"""
    if tag == "root":
        return h == 0
    if tag.endswith("<"):
        return h != 0 and h < m
    if tag.endswith(">"):
        return h > m
    return False


def tag_label(tag):
    """The relation part of a tag: "nmod:poss<" -> "nmod:poss"."""
    return tag.rstrip("<>")


def top_labels(score_of_label, count):
    """The `count` labels with the highest scores."""
    chosen = []
    for c in range(count):
        best = None
        for label in score_of_label:
            if label in chosen:
                continue
            if best is None or score_of_label[label] > score_of_label[best]:
                best = label
        if best is None:
            break
        chosen.append(best)
    return chosen


# ----------------------------- heads -----------------------------

def candidate_heads(outputs, k, m):
    """All heads that any parser or the MST parser proposed for word m."""
    n = len(outputs["sentences"][k]["words"]) - 1
    found = []
    for (all_heads, all_labels, all_margins) in outputs["predictions"]:
        h = all_heads[k][m]
        if 0 <= h <= n and h != m and h not in found:
            found.append(h)
    mst = get_mst(outputs, k)
    if mst is not None:
        for key in ("heads", "heads2", "heads3"):
            if key in mst:
                h = mst[key][m]
                if 0 <= h <= n and h != m and h not in found:
                    found.append(h)
    return found


def arc_features(outputs, k, m, h):
    """Numbers that describe the candidate arc h -> m."""
    words = outputs["sentences"][k]["words"]
    punct_forms = outputs["punct_forms"]
    weights = outputs["weights"]
    n_parsers = len(outputs["predictions"])

    f = [1.0]   # bias

    # Which parsers chose this head, and how sure were they?
    votes = 0.0
    weighted_votes = 0.0
    total_weight = 0.0
    hand_score = 0.0      # the score of the normal (hand-set) vote
    for p in range(n_parsers):
        all_heads, all_labels, all_margins = outputs["predictions"][p]
        total_weight = total_weight + weights[p]
        if all_heads[k][m] == h:
            conf = confidence(all_margins[k][m])
            f.append(1.0)
            f.append(conf)
            votes = votes + 1.0
            weighted_votes = weighted_votes + weights[p]
            hand_score = hand_score + weights[p] * (0.85 + 0.30 * conf)
        else:
            f.append(0.0)
            f.append(0.0)
    f.append(votes / n_parsers)
    f.append(weighted_votes / total_weight)
    if votes == n_parsers:
        f.append(1.0)
    else:
        f.append(0.0)

    # What did the MST parser say?
    is_first = 0.0
    is_second = 0.0
    is_third = 0.0
    high = 0.0
    mid = 0.0
    low = 0.0
    mst_margin = 0.0
    mst = get_mst(outputs, k)
    if mst is not None:
        if mst["heads"][m] == h:
            is_first = 1.0
            mst_margin = confidence(mst["margins"][m])
            if mst["conf"][m] == "HIGH":
                high = 1.0
                hand_score = hand_score + 1.05
            elif mst["conf"][m] == "MID":
                mid = 1.0
                hand_score = hand_score + 0.72
            else:
                low = 1.0
                hand_score = hand_score + 0.45
        if mst["heads2"][m] == h:
            is_second = 1.0
            hand_score = hand_score + 0.22
        if "heads3" in mst and mst["heads3"][m] == h:
            is_third = 1.0
            hand_score = hand_score + 0.08
    f.extend([is_first, is_second, is_third, high, mid, low, mst_margin])
    f.append(hand_score / 10.0)

    # Direction and distance
    if h == 0:
        f.extend([1.0, 0.0, 0.0])
    elif h < m:
        f.extend([0.0, 1.0, 0.0])
    else:
        f.extend([0.0, 0.0, 1.0])
    distance = abs(h - m)
    buckets = [0.0, 0.0, 0.0, 0.0, 0.0]
    if h != 0:
        if distance == 1:
            buckets[0] = 1.0
        elif distance == 2:
            buckets[1] = 1.0
        elif distance <= 4:
            buckets[2] = 1.0
        elif distance <= 9:
            buckets[3] = 1.0
        else:
            buckets[4] = 1.0
    f.extend(buckets)

    # Punctuation
    m_is_punct = words[m] in punct_forms
    h_is_punct = h > 0 and words[h] in punct_forms
    f.append(1.0 if m_is_punct else 0.0)
    f.append(1.0 if h_is_punct else 0.0)
    f.append(votes / n_parsers if m_is_punct else 0.0)

    # Do the supertagger (head offset) and the tagger (head side) agree?
    supertags = get_supertags(outputs, k)
    offset_agrees = 0.0
    if supertags is not None and supertags[m] is not None and ":" in supertags[m]:
        if supertags[m].split(":")[0] == head_offset_bucket(m, h):
            offset_agrees = 1.0
    f.append(offset_agrees)
    f.append(1.0 if tag_side_agrees(outputs["tags"][k][m], m, h) else 0.0)
    return f


def arc_table(outputs, sentence_ids):
    """All candidate arcs of these sentences: features, answers, positions."""
    rows = []
    answers = []
    positions = []
    for k in sentence_ids:
        sentence = outputs["sentences"][k]
        gold = has_gold(sentence)
        for m in range(1, len(sentence["words"])):
            for h in candidate_heads(outputs, k, m):
                rows.append(arc_features(outputs, k, m, h))
                if gold and sentence["heads"][m] == h:
                    answers.append(1.0)
                else:
                    answers.append(0.0)
                positions.append((k, m, h))
    return np.array(rows), np.array(answers), positions


def predict_heads(outputs, arc_weights, sentence_ids):
    """Best tree for every sentence, from the learned arc probabilities."""
    X, answers, positions = arc_table(outputs, sentence_ids)
    probability = sigmoid(X.dot(arc_weights))

    score_matrix = {}
    for k in sentence_ids:
        n = len(outputs["sentences"][k]["words"]) - 1
        score_matrix[k] = np.full((n + 1, n + 1), -1.0e4)
    for row in range(len(positions)):
        k, m, h = positions[row]
        score_matrix[k][h, m] = math.log(max(float(probability[row]), 1.0e-6))

    heads_of = {}
    for k in sentence_ids:
        heads_of[k] = chu_liu_edmonds(score_matrix[k])
    return heads_of


# ----------------------------- labels -----------------------------

def labeler_scores(outputs, k, heads):
    """Scores of the full-tree labeler for every word of sentence k, given a tree."""
    labeler = outputs["tree_labeler"]
    if labeler is None:
        return None
    sentence = outputs["sentences"][k]
    upos = None
    if outputs["upos"] is not None:
        upos = outputs["upos"][k]
    return labeler.score_sentence_labels(sentence["words"], outputs["tags"][k], get_supertags(outputs, k), heads, upos=upos)


def label_candidates_and_rows(outputs, k, m, h, scores):
    """Candidate labels for word m (head h) and the features of each."""
    weights = outputs["weights"]
    n_parsers = len(outputs["predictions"])

    candidates = []
    for (all_heads, all_labels, all_margins) in outputs["predictions"]:
        label = all_labels[k][m]
        if label != "root" and label not in candidates:
            candidates.append(label)
    best_labeler_labels = []
    if scores is not None and scores[m] is not None:
        best_labeler_labels = top_labels(scores[m], 3)
        for label in best_labeler_labels:
            if label != "root" and label not in candidates:
                candidates.append(label)

    # The score of the normal (hand-set) label vote, for every candidate
    hand_score = {}
    for label in candidates:
        hand_score[label] = 0.0
    for p in range(n_parsers):
        all_heads, all_labels, all_margins = outputs["predictions"][p]
        label = all_labels[k][m]
        if label not in hand_score:
            continue
        if all_heads[k][m] == h:
            hand_score[label] = hand_score[label] + weights[p] * (0.85 + 0.30 * confidence(all_margins[k][m]))
        else:
            hand_score[label] = hand_score[label] + 0.22 * weights[p]
    labeler_max = None
    if scores is not None and scores[m]:
        labeler_max = max(scores[m].values())
        for label in candidates:
            if label in scores[m]:
                difference = scores[m][label] - labeler_max
                if difference >= -8.0:
                    hand_score[label] = hand_score[label] + 1.25 * math.exp(0.25 * difference)
    best_hand = 0.0
    for label in candidates:
        if hand_score[label] > best_hand:
            best_hand = hand_score[label]

    tag = outputs["tags"][k][m]
    rows = []
    for label in candidates:
        f = [1.0]   # bias
        same_head_votes = 0.0
        any_votes = 0.0
        for p in range(n_parsers):
            all_heads, all_labels, all_margins = outputs["predictions"][p]
            if all_labels[k][m] == label:
                any_votes = any_votes + 1.0
                if all_heads[k][m] == h:
                    same_head_votes = same_head_votes + 1.0
                    f.append(1.0)
                    f.append(0.0)
                else:
                    f.append(0.0)
                    f.append(1.0)
            else:
                f.append(0.0)
                f.append(0.0)
        f.append(same_head_votes / n_parsers)
        f.append(any_votes / n_parsers)
        if best_hand > 0.0:
            f.append(hand_score[label] / best_hand)
        else:
            f.append(0.0)

        # The full-tree labeler
        if labeler_max is not None and label in scores[m]:
            difference = scores[m][label] - labeler_max
            if difference < -10.0:
                difference = -10.0
            f.append(difference / 10.0)
        else:
            f.append(-1.0)
        f.append(1.0 if len(best_labeler_labels) >= 1 and best_labeler_labels[0] == label else 0.0)
        f.append(1.0 if len(best_labeler_labels) >= 2 and best_labeler_labels[1] == label else 0.0)

        # The tagger's guess
        f.append(1.0 if tag_label(tag) == label else 0.0)
        f.append(1.0 if tag_label(tag).split(":")[0] == label.split(":")[0] else 0.0)
        rows.append(f)
    return candidates, rows


def label_table(outputs, sentence_ids):
    """
    Training rows for the label model. We learn "which label is right when
    the head is right", so the features use the correct tree.
    """
    rows = []
    answers = []
    for k in sentence_ids:
        sentence = outputs["sentences"][k]
        scores = labeler_scores(outputs, k, sentence["heads"])
        for m in range(1, len(sentence["words"])):
            h = sentence["heads"][m]
            if h == 0:
                continue
            candidates, candidate_rows = label_candidates_and_rows(outputs, k, m, h, scores)
            for c in range(len(candidates)):
                rows.append(candidate_rows[c])
                if candidates[c] == sentence["labels"][m]:
                    answers.append(1.0)
                else:
                    answers.append(0.0)
    return np.array(rows), np.array(answers)


def predict_labels(outputs, label_weights, k, heads):
    """The best label for every word of sentence k, given its tree."""
    scores = labeler_scores(outputs, k, heads)
    n = len(outputs["sentences"][k]["words"]) - 1
    labels = [""]
    for m in range(1, n + 1):
        h = heads[m]
        if h == 0:
            labels.append("root")
            continue
        candidates, candidate_rows = label_candidates_and_rows(outputs, k, m, h, scores)
        if len(candidates) == 0:
            labels.append("dep")
            continue
        probability = sigmoid(np.array(candidate_rows).dot(label_weights))
        labels.append(candidates[int(probability.argmax())])
    return labels


# ----------------------------- training and use -----------------------------

def train_combiner(outputs, sentence_ids):
    """Learn the head model and the label model from these check sentences."""
    X, y, positions = arc_table(outputs, sentence_ids)
    arc_weights = train_logistic_regression(X, y)
    X, y = label_table(outputs, sentence_ids)
    label_weights = train_logistic_regression(X, y)
    model = {}
    model["arc_weights"] = arc_weights
    model["label_weights"] = label_weights
    model["parsers"] = list(outputs["parsers"])
    model["uses_mst"] = outputs["mst"] is not None and outputs["mst"][0] is not None
    model["uses_labeler"] = outputs["tree_labeler"] is not None
    return model


def combine(outputs, model, sentence_ids=None):
    """Heads and labels for the sentences, using a trained combiner."""
    if sentence_ids is None:
        sentence_ids = list(range(len(outputs["sentences"])))
    heads_of = predict_heads(outputs, model["arc_weights"], sentence_ids)
    labels_of = {}
    for k in sentence_ids:
        labels_of[k] = predict_labels(outputs, model["label_weights"], k, heads_of[k])
    return heads_of, labels_of


def cross_validate(outputs):
    """
    Honest score of the combiner on the check data: learn on one half of the
    check sentences, predict the other half, and the other way round.
    The check sentences come in groups of neighbouring sentences, so the two
    halves mostly contain different documents.
    """
    n = len(outputs["sentences"])
    first_half = list(range(0, n // 2))
    second_half = list(range(n // 2, n))
    all_heads = [None] * n
    all_labels = [None] * n
    for (learn_ids, predict_ids) in [(first_half, second_half), (second_half, first_half)]:
        model = train_combiner(outputs, learn_ids)
        heads_of, labels_of = combine(outputs, model, predict_ids)
        for k in predict_ids:
            all_heads[k] = heads_of[k]
            all_labels[k] = labels_of[k]
    return all_heads, all_labels
