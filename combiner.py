# optional softmax regression combiner to learn weights for parser outputs on val data

import math

import numpy as np

from data_utils import head_offset_bucket
from parser import chu_liu_edmonds


MARGIN_LIMIT = 12.0


def train_softmax_regression(X, y, group_slices, l2=5.0, iterations=30):
    # fit softmax logistic regression with l2 penalty using newton method
    n_features = X.shape[1]
    if len(group_slices) == 0:
        return np.zeros(n_features, dtype=np.float64)

    # keep groups with at least 2 options and one correct option
    keep_rows = []
    lengths = []
    for (s, e) in group_slices:
        if e - s < 2:
            continue
        if y[s:e].sum() <= 0.0:
            continue
        keep_rows.append((s, e))
        lengths.append(e - s)

    if not keep_rows:
        return np.zeros(n_features, dtype=np.float64)

    X_parts = [X[s:e] for (s, e) in keep_rows]
    y_parts = [y[s:e] for (s, e) in keep_rows]
    X_act = np.vstack(X_parts)
    y_act = np.concatenate(y_parts)

    lengths_arr = np.array(lengths, dtype=np.int32)
    starts = np.zeros(len(lengths), dtype=np.int32)
    if len(lengths) > 1:
        starts[1:] = np.cumsum(lengths_arr[:-1])
    group_ids = np.repeat(np.arange(len(lengths), dtype=np.int32), lengths_arr)

    w = np.zeros(n_features, dtype=np.float64)
    eye_l2 = l2 * np.eye(n_features, dtype=np.float64)

    for _ in range(iterations):
        z = X_act.dot(w)
        z_max = np.maximum.reduceat(z, starts)
        exp_z = np.exp(np.clip(z - z_max[group_ids], -40.0, 40.0))
        denom = np.add.reduceat(exp_z, starts)
        p = exp_z / denom[group_ids]

        grad = X_act.T.dot(p - y_act) + l2 * w
        pX = X_act * p[:, None]
        mean_X = np.add.reduceat(pX, starts, axis=0)
        hessian = pX.T.dot(X_act) - mean_X.T.dot(mean_X) + eye_l2

        step = np.linalg.solve(hessian, grad)
        w = w - step
        if np.abs(step).max() < 1.0e-6:
            break
    return w


def confidence(margin):
    # clip margin to [0, 1]
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
    if tag == "root":
        return h == 0
    if tag.endswith("<"):
        return h != 0 and h < m
    if tag.endswith(">"):
        return h > m
    return False


def tag_label(tag):
    return tag.rstrip("<>")


def top_labels(label_scores, count):
    chosen = []
    for _ in range(count):
        best = None
        for label in label_scores:
            if label in chosen:
                continue
            if best is None or label_scores[label] > label_scores[best]:
                best = label
        if best is None:
            break
        chosen.append(best)
    return chosen


def get_base_tree(outputs, k):
    # build base mst tree from normal weighted votes
    words = outputs["sentences"][k]["words"]
    n = len(words) - 1
    if n <= 0:
        return [-1]
    punct_forms = outputs["punct_forms"]
    punct_flags = [False] + [punct_forms is not None and words[i] in punct_forms for i in range(1, n + 1)]
    all_punct = all(punct_flags[1:])

    arc_scores = np.full((n + 1, n + 1), -1.0e4, dtype=np.float64)
    for h in range(n + 1):
        for m in range(1, n + 1):
            if h == m:
                continue
            if not all_punct and h > 0 and punct_flags[h] and not punct_flags[m]:
                arc_scores[h, m] = -1.0e6
            elif not all_punct and h == 0 and punct_flags[m]:
                arc_scores[h, m] = -1.0e5
            else:
                arc_scores[h, m] = 0.0

    weights = outputs["weights"]
    for p_idx, pred_tuple in enumerate(outputs["predictions"]):
        all_heads = pred_tuple[0]
        all_margins = pred_tuple[2]
        w_p = weights[p_idx]
        for i in range(1, n + 1):
            h = all_heads[k][i]
            if 0 <= h <= n and h != i:
                conf_mult = 0.85 + 0.30 * confidence(all_margins[k][i])
                arc_scores[h, i] += w_p * conf_mult - 1.0e-5 * p_idx

    mst = get_mst(outputs, k)
    if mst is not None:
        mh1 = mst["heads"]
        mh2 = mst["heads2"]
        mh3 = mst.get("heads3")
        mconf = mst["conf"]
        for i in range(1, n + 1):
            h1 = mh1[i]
            h2 = mh2[i]
            h3 = mh3[i] if mh3 is not None else -1
            if 0 <= h1 <= n and h1 != i:
                bonus = 1.05 if mconf[i] == "HIGH" else (0.72 if mconf[i] == "MID" else 0.45)
                arc_scores[h1, i] += bonus
            if 0 <= h2 <= n and h2 != i:
                arc_scores[h2, i] += 0.22
            if 0 <= h3 <= n and h3 != i:
                arc_scores[h3, i] += 0.08
    return chu_liu_edmonds(arc_scores)


def candidate_heads(outputs, k, m):
    # collect all heads suggested by any parser or mst for word m
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


def get_arc_features(outputs, k, m, cands):
    # build feature rows for each candidate head of word m
    words = outputs["sentences"][k]["words"]
    punct_forms = outputs["punct_forms"]
    weights = outputs["weights"]
    parsers_meta = outputs["parsers"]
    n_parsers = len(outputs["predictions"])
    mst = get_mst(outputs, k)
    supertags = get_supertags(outputs, k)

    n_hyb = max(sum(1 for pm in parsers_meta if pm[0] == "arc-hybrid"), 1)
    n_eag = max(sum(1 for pm in parsers_meta if pm[0] == "arc-eager"), 1)
    n_std = max(sum(1 for pm in parsers_meta if pm[0] == "arc-standard"), 1)
    n_ltr = max(sum(1 for pm in parsers_meta if pm[1] == "left-to-right"), 1)
    n_rtl = max(sum(1 for pm in parsers_meta if pm[1] == "right-to-left"), 1)

    raw_rows = []
    hand_scores = []
    vote_counts = []

    for h in cands:
        f = [1.0]   # bias

        votes = 0.0
        weighted_votes = 0.0
        total_weight = 0.0
        hand_score = 0.0
        hyb_v = 0.0
        eag_v = 0.0
        std_v = 0.0
        ltr_v = 0.0
        rtl_v = 0.0

        for p in range(n_parsers):
            all_heads, all_labels, all_margins = outputs["predictions"][p]
            total_weight += weights[p]
            if all_heads[k][m] == h:
                conf = confidence(all_margins[k][m])
                f.append(1.0)
                f.append(conf)
                votes += 1.0
                weighted_votes += weights[p]
                hand_score += weights[p] * (0.85 + 0.30 * conf)
                sys_type, direction = parsers_meta[p][0], parsers_meta[p][1]
                if sys_type == "arc-hybrid":
                    hyb_v += 1.0
                elif sys_type == "arc-eager":
                    eag_v += 1.0
                else:
                    std_v += 1.0
                if direction == "left-to-right":
                    ltr_v += 1.0
                else:
                    rtl_v += 1.0
            else:
                f.append(0.0)
                f.append(0.0)

        n_fams = (1.0 if hyb_v > 0 else 0.0) + (1.0 if eag_v > 0 else 0.0) + (1.0 if std_v > 0 else 0.0)
        both_dirs = 1.0 if (ltr_v > 0 and rtl_v > 0) else 0.0

        f.extend([
            votes / n_parsers,
            weighted_votes / max(total_weight, 1.0),
            1.0 if votes == n_parsers else 0.0,
            hyb_v / n_hyb,
            eag_v / n_eag,
            std_v / n_std,
            ltr_v / n_ltr,
            rtl_v / n_rtl,
            n_fams / 3.0,
            both_dirs,
        ])

        # mst features
        is_first = 0.0
        is_second = 0.0
        is_third = 0.0
        high = 0.0
        mid = 0.0
        low = 0.0
        mst_margin = 0.0
        if mst is not None:
            if mst["heads"][m] == h:
                is_first = 1.0
                mst_margin = confidence(mst["margins"][m])
                if mst["conf"][m] == "HIGH":
                    high = 1.0
                    hand_score += 1.05
                elif mst["conf"][m] == "MID":
                    mid = 1.0
                    hand_score += 0.72
                else:
                    low = 1.0
                    hand_score += 0.45
            if mst["heads2"][m] == h:
                is_second = 1.0
                hand_score += 0.22
            if "heads3" in mst and mst["heads3"][m] == h:
                is_third = 1.0
                hand_score += 0.08
        f.extend([is_first, is_second, is_third, high, mid, low, mst_margin])
        f.append(hand_score / 10.0)

        # direction and distance
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

        # punctuation
        m_is_punct = punct_forms is not None and words[m] in punct_forms
        h_is_punct = h > 0 and punct_forms is not None and words[h] in punct_forms
        f.append(1.0 if m_is_punct else 0.0)
        f.append(1.0 if h_is_punct else 0.0)
        f.append(votes / n_parsers if m_is_punct else 0.0)

        # supertag and tagger agreement
        offset_agrees = 0.0
        if supertags is not None and supertags[m] is not None and ":" in supertags[m]:
            if supertags[m].split(":")[0] == head_offset_bucket(m, h):
                offset_agrees = 1.0
        f.append(offset_agrees)
        f.append(1.0 if tag_side_agrees(outputs["tags"][k][m], m, h) else 0.0)

        raw_rows.append(f)
        hand_scores.append(hand_score)
        vote_counts.append(votes)

    # score gap from best candidate
    best_hs = max(hand_scores) if hand_scores else 0.0
    best_vc = max(vote_counts) if vote_counts else 0.0
    for idx in range(len(raw_rows)):
        hs = hand_scores[idx]
        vc = vote_counts[idx]
        raw_rows[idx].append((hs - best_hs) / 10.0)
        raw_rows[idx].append((vc - best_vc) / max(n_parsers, 1))
        raw_rows[idx].append(1.0 if hs >= best_hs - 1.0e-9 else 0.0)

    return raw_rows


def arc_table(outputs, sentence_ids):
    rows = []
    answers = []
    positions = []
    group_slices = []

    for k in sentence_ids:
        sentence = outputs["sentences"][k]
        words = sentence["words"]
        n = len(words) - 1
        gold = has_gold(sentence)

        cands_per_tok = [[]] + [candidate_heads(outputs, k, m) for m in range(1, n + 1)]
        for m in range(1, n + 1):
            cands = cands_per_tok[m]
            if not cands:
                continue
            start_idx = len(rows)
            tok_rows = get_arc_features(outputs, k, m, cands)
            for idx, h in enumerate(cands):
                rows.append(tok_rows[idx])
                if gold and sentence["heads"][m] == h:
                    answers.append(1.0)
                else:
                    answers.append(0.0)
                positions.append((k, m, h))
            group_slices.append((start_idx, len(rows)))

    return np.array(rows, dtype=np.float64), np.array(answers, dtype=np.float64), positions, group_slices


def predict_heads(outputs, arc_weights, sentence_ids):
    # score candidate arcs and run chu-liu-edmonds for each sentence
    X, answers, positions, group_slices = arc_table(outputs, sentence_ids)
    logits = X.dot(arc_weights)

    # turn logits into log probabilities per token
    log_probs = np.zeros(len(logits), dtype=np.float64)
    for (s, e) in group_slices:
        g = logits[s:e]
        g_max = float(g.max())
        lse = g_max + math.log(float(np.exp(g - g_max).sum()))
        log_probs[s:e] = g - lse

    punct_forms = outputs["punct_forms"]
    score_matrix = {}
    for k in sentence_ids:
        words = outputs["sentences"][k]["words"]
        n = len(words) - 1
        punct_flags = [False] + [punct_forms is not None and words[i] in punct_forms for i in range(1, n + 1)]
        all_punct = all(punct_flags[1:])
        mat = np.full((n + 1, n + 1), -1.0e4, dtype=np.float64)
        for h in range(n + 1):
            for m in range(1, n + 1):
                if h == m:
                    continue
                if not all_punct and h > 0 and punct_flags[h] and not punct_flags[m]:
                    mat[h, m] = -1.0e6
                elif not all_punct and h == 0 and punct_flags[m]:
                    mat[h, m] = -1.0e5
        score_matrix[k] = mat

    for row in range(len(positions)):
        k, m, h = positions[row]
        if score_matrix[k][h, m] > -5.0e4:
            score_matrix[k][h, m] = float(log_probs[row])

    pred_heads = {}
    for k in sentence_ids:
        pred_heads[k] = chu_liu_edmonds(score_matrix[k])
    return pred_heads


def labeler_scores(outputs, k, heads):
    labeler = outputs["tree_labeler"]
    if labeler is None:
        return None
    sentence = outputs["sentences"][k]
    upos = outputs["upos"][k] if outputs["upos"] is not None else None
    return labeler.score_sentence_labels(
        sentence["words"],
        outputs["tags"][k],
        get_supertags(outputs, k),
        heads,
        upos=upos,
        extra_models=outputs.get("extra_labelers"),
    )


def label_candidates_and_rows(outputs, k, m, h, scores):
    # build candidate labels and features for word m
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

    # base vote score for each candidate label
    hand_score = {label: 0.0 for label in candidates}
    for p in range(n_parsers):
        all_heads, all_labels, all_margins = outputs["predictions"][p]
        label = all_labels[k][m]
        if label not in hand_score:
            continue
        if all_heads[k][m] == h:
            hand_score[label] += weights[p] * (0.85 + 0.30 * confidence(all_margins[k][m]))
        else:
            hand_score[label] += 0.22 * weights[p]

    labeler_max = None
    if scores is not None and scores[m]:
        labeler_max = max(scores[m].values())
        for label in candidates:
            if label in scores[m]:
                difference = scores[m][label] - labeler_max
                if difference >= -8.0:
                    hand_score[label] += 1.25 * math.exp(0.25 * difference)
    best_hand = max(hand_score.values()) if hand_score else 0.0

    tag = outputs["tags"][k][m]
    rows = []
    for label in candidates:
        f = [1.0]   # bias
        same_head_votes = 0.0
        any_votes = 0.0
        for p in range(n_parsers):
            all_heads, all_labels, all_margins = outputs["predictions"][p]
            if all_labels[k][m] == label:
                any_votes += 1.0
                if all_heads[k][m] == h:
                    same_head_votes += 1.0
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
        f.append(hand_score[label] / best_hand if best_hand > 0.0 else 0.0)
        f.append((hand_score[label] - best_hand) / 10.0)

        # full-tree labeler scores
        if labeler_max is not None and label in scores[m]:
            difference = scores[m][label] - labeler_max
            if difference < -10.0:
                difference = -10.0
            f.append(difference / 10.0)
        else:
            f.append(-1.0)
        f.append(1.0 if len(best_labeler_labels) >= 1 and best_labeler_labels[0] == label else 0.0)
        f.append(1.0 if len(best_labeler_labels) >= 2 and best_labeler_labels[1] == label else 0.0)

        # tagger match
        f.append(1.0 if tag_label(tag) == label else 0.0)
        f.append(1.0 if tag_label(tag).split(":")[0] == label.split(":")[0] else 0.0)
        rows.append(f)
    return candidates, rows


def label_table(outputs, sentence_ids):
    rows = []
    answers = []
    group_slices = []
    for k in sentence_ids:
        sentence = outputs["sentences"][k]
        scores = labeler_scores(outputs, k, sentence["heads"])
        for m in range(1, len(sentence["words"])):
            h = sentence["heads"][m]
            if h == 0:
                continue
            candidates, candidate_rows = label_candidates_and_rows(outputs, k, m, h, scores)
            if not candidates:
                continue
            start_idx = len(rows)
            for c in range(len(candidates)):
                rows.append(candidate_rows[c])
                if candidates[c] == sentence["labels"][m]:
                    answers.append(1.0)
                else:
                    answers.append(0.0)
            group_slices.append((start_idx, len(rows)))
    return np.array(rows, dtype=np.float64), np.array(answers, dtype=np.float64), group_slices


def predict_labels(outputs, label_weights, k, heads):
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
        logits = np.array(candidate_rows, dtype=np.float64).dot(label_weights)
        labels.append(candidates[int(logits.argmax())])
    return labels


def train_combiner(outputs, sentence_ids, l2=5.0):
    X_arc, y_arc, _, arc_groups = arc_table(outputs, sentence_ids)
    arc_weights = train_softmax_regression(X_arc, y_arc, arc_groups, l2=l2)
    X_lbl, y_lbl, lbl_groups = label_table(outputs, sentence_ids)
    label_weights = train_softmax_regression(X_lbl, y_lbl, lbl_groups, l2=l2)
    return {
        "arc_weights": arc_weights,
        "label_weights": label_weights,
        "parsers": list(outputs["parsers"]),
        "uses_mst": outputs["mst"] is not None and outputs["mst"][0] is not None,
        "uses_labeler": outputs["tree_labeler"] is not None,
    }


def combine(outputs, model, sentence_ids=None):
    if sentence_ids is None:
        sentence_ids = list(range(len(outputs["sentences"])))
    pred_heads = predict_heads(outputs, model["arc_weights"], sentence_ids)
    pred_labels = {k: predict_labels(outputs, model["label_weights"], k, pred_heads[k]) for k in sentence_ids}
    return pred_heads, pred_labels


def cross_validate(outputs, l2=5.0):
    # 2-fold cv on val set (first half vs second half)
    n = len(outputs["sentences"])
    first_half = list(range(0, n // 2))
    second_half = list(range(n // 2, n))
    all_heads = [None] * n
    all_labels = [None] * n
    for (learn_ids, predict_ids) in [(first_half, second_half), (second_half, first_half)]:
        model = train_combiner(outputs, learn_ids, l2=l2)
        pred_heads, pred_labels = combine(outputs, model, predict_ids)
        for k in predict_ids:
            all_heads[k] = pred_heads[k]
            all_labels[k] = pred_labels[k]
    return all_heads, all_labels
