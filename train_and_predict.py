"""
Main script: trains the bidirectional two-pass tagger, global-context
head-offset supertagger, global first-order Graph-MST structured perceptron,
full-tree 2nd/3rd-order label classifier, and heterogeneous (Arc-Hybrid +
Arc-Eager, L-to-R + R-to-L) graph-guided transition parser ensemble, then
decodes the globally optimal parse tree via Chu-Liu/Edmonds Maximum Spanning
Arborescence and writes the submission file.

    python train_and_predict.py           -> train on all training data,
                                             write outputs/submission.csv
    python train_and_predict.py --check   -> train on 90% of the training data
                                             and print the score on the other 10%

Extra options (can be combined with --check):
    --one-parser       train only the first parser (fast, for experiments)
    --no-second-pass   do not use the second-pass tagger, supertagger & MST guide
    --no-copy          do not copy answers of sentences seen in training
"""

import gc
import math
import os
import random
import sys
import time

from data_utils import (
    read_sentences,
    read_sample_ids,
    write_submission,
    accuracy,
    normalize_word,
    is_punct_word,
    build_lexicon,
)
from tagger import (
    train_tagger,
    jackknife_tags,
    make_tag_targets,
    make_supertag_targets,
    tag_accuracy,
)
from parser import (
    Parser,
    ArcEagerParser,
    GraphMSTParser,
    FullTreeLabeler,
    jackknife_mst,
    chu_liu_edmonds,
    flip_sentence,
    flip_tags,
    flip_supertags,
    flip_mst_info,
    flip_result,
)

import numpy as np


# ----------------------------- settings -----------------------------

DATA_FOLDER = "data"
OUTPUT_FOLDER = "outputs"

TAGGER_EPOCHS = 6
SUPERTAGGER_EPOCHS = 5
MST_EPOCHS = 7
TREE_LABELER_EPOCHS = 5
JACKKNIFE_FOLDS = 5
PARSER_EPOCHS = 12

# Heterogeneous graph-guided transition parser ensemble:
# (system_type, reading_direction, random_seed, vote_weight)
PARSERS = [
    ("arc-hybrid", "left-to-right", 1, 1.35),
    ("arc-eager",  "left-to-right", 1, 1.25),
    ("arc-hybrid", "right-to-left", 1, 1.05),
    ("arc-hybrid", "left-to-right", 2, 1.30),
    ("arc-eager",  "left-to-right", 2, 1.20),
    ("arc-hybrid", "right-to-left", 2, 1.00),
    ("arc-eager",  "right-to-left", 1, 0.95),
    ("arc-hybrid", "left-to-right", 3, 1.30),
    ("arc-hybrid", "left-to-right", 4, 1.30),
    ("arc-eager",  "left-to-right", 3, 1.20),
]

USE_SECOND_PASS_TAGGER = True
COPY_KNOWN_SENTENCES = True

# --------------------------------------------------------------------


def get_label_names(sentences):
    names = set()
    for sentence in sentences:
        for i in range(1, len(sentence["labels"])):
            names.add(sentence["labels"][i])
    return sorted(names)


def make_tags(train_sentences, other_sentence_lists):
    """
    Tag the training sentences with 5-fold jackknifing, and the other sentences
    (dev or test) with taggers trained on all training sentences:
      1. Pass-1 Forward (Left-to-Right) tagger
      2. Pass-1 Backward (Right-to-Left) tagger
      3. Pass-2 Stacked tagger conditioned on Forward + Backward tags + Global Clause/Root Lookahead
      4. Relative head-offset supertagger conditioned on Pass-2 + Backward tags + Global Lookahead
    """
    tag_sentences = make_tag_targets(train_sentences)
    tag_names = get_label_names(tag_sentences)

    print("Training the first-pass L-to-R tagger (jackknifing for training data)...")
    train_tags_fwd = jackknife_tags(
        tag_sentences, tag_names, TAGGER_EPOCHS, JACKKNIFE_FOLDS, direction="left-to-right"
    )
    print("    L-to-R tag accuracy on training data (jackknifed): %.2f" % tag_accuracy(tag_sentences, train_tags_fwd))

    print("Training the first-pass L-to-R tagger on all training sentences...")
    tagger_fwd = train_tagger(tag_sentences, tag_names, TAGGER_EPOCHS, direction="left-to-right")
    other_tags_fwd = []
    for sentences in other_sentence_lists:
        tags_list = [tagger_fwd.tag(sentence["words"]) for sentence in sentences]
        other_tags_fwd.append(tags_list)
    tagger_fwd = None
    gc.collect()

    if not USE_SECOND_PASS_TAGGER:
        empty_super = [[None] * len(s) for s in other_sentence_lists]
        return train_tags_fwd, other_tags_fwd, [None] * len(train_sentences), empty_super

    # ----- first-pass backward (Right-to-Left) tagger -----
    print("Training the first-pass R-to-L tagger (jackknifing for training data)...")
    train_tags_bwd = jackknife_tags(
        tag_sentences, tag_names, TAGGER_EPOCHS, JACKKNIFE_FOLDS, direction="right-to-left"
    )
    print("    R-to-L tag accuracy on training data (jackknifed): %.2f" % tag_accuracy(tag_sentences, train_tags_bwd))

    print("Training the first-pass R-to-L tagger on all training sentences...")
    tagger_bwd = train_tagger(tag_sentences, tag_names, TAGGER_EPOCHS, direction="right-to-left")
    other_tags_bwd = []
    for sentences in other_sentence_lists:
        tags_list = [tagger_bwd.tag(sentence["words"]) for sentence in sentences]
        other_tags_bwd.append(tags_list)
    tagger_bwd = None
    gc.collect()

    # ----- second pass: stacked bidirectional tagger with global lookahead -----
    print("Training the second-pass bidirectional tagger (jackknifing)...")
    train_tags_2 = jackknife_tags(
        tag_sentences,
        tag_names,
        TAGGER_EPOCHS,
        JACKKNIFE_FOLDS,
        first_tags_list=train_tags_fwd,
        direction="left-to-right",
        backward_tags_list=train_tags_bwd,
    )
    print("    second-pass tag accuracy on training data (jackknifed): %.2f" % tag_accuracy(tag_sentences, train_tags_2))

    print("Training the second-pass bidirectional tagger on all training sentences...")
    tagger_2 = train_tagger(
        tag_sentences,
        tag_names,
        TAGGER_EPOCHS,
        first_tags_list=train_tags_fwd,
        direction="left-to-right",
        backward_tags_list=train_tags_bwd,
    )
    other_tags_lists_2 = []
    for j in range(len(other_sentence_lists)):
        sentences = other_sentence_lists[j]
        tags_list = [
            tagger_2.tag(
                sentences[k]["words"],
                first_tags=other_tags_fwd[j][k],
                backward_tags=other_tags_bwd[j][k],
            )
            for k in range(len(sentences))
        ]
        other_tags_lists_2.append(tags_list)
    tagger_2 = None
    gc.collect()

    # ----- supertagger: relative head-offset + head coarse group -----
    super_sentences = make_supertag_targets(train_sentences)
    super_names = get_label_names(super_sentences)
    print("Training the head-offset supertagger (jackknifing)...")
    train_supertags = jackknife_tags(
        super_sentences,
        super_names,
        SUPERTAGGER_EPOCHS,
        JACKKNIFE_FOLDS,
        first_tags_list=train_tags_2,
        direction="left-to-right",
        backward_tags_list=train_tags_bwd,
    )
    print("    supertag accuracy on training data (jackknifed): %.2f" % tag_accuracy(super_sentences, train_supertags))

    print("Training the head-offset supertagger on all training sentences...")
    super_tagger = train_tagger(
        super_sentences,
        super_names,
        SUPERTAGGER_EPOCHS,
        first_tags_list=train_tags_2,
        direction="left-to-right",
        backward_tags_list=train_tags_bwd,
    )
    other_super_lists = []
    for j in range(len(other_sentence_lists)):
        sentences = other_sentence_lists[j]
        s_list = [
            super_tagger.tag(
                sentences[k]["words"],
                first_tags=other_tags_lists_2[j][k],
                backward_tags=other_tags_bwd[j][k],
            )
            for k in range(len(sentences))
        ]
        other_super_lists.append(s_list)
    super_tagger = None
    gc.collect()

    return train_tags_2, other_tags_lists_2, train_supertags, other_super_lists


def make_mst_guides(train_sentences, train_tags, train_supertags, other_sentence_lists, other_tags_lists, other_super_lists, lexicon_ltr):
    """
    Train the Global First-Order Graph-Based MST Structured Perceptron:
      - 5-fold jackknifing on `train_sentences` for zero-leakage stacking
      - full training on all `train_sentences` for `other_sentence_lists`
    """
    if not USE_SECOND_PASS_TAGGER:
        empty_other = [[None] * len(s) for s in other_sentence_lists]
        return [None] * len(train_sentences), empty_other

    print("Training the global Graph-MST parser (jackknifing for training data)...")
    train_mst = jackknife_mst(
        train_sentences,
        train_tags,
        train_supertags,
        epochs=MST_EPOCHS,
        n_folds=JACKKNIFE_FOLDS,
    )
    mst_train_heads = [m["heads"] for m in train_mst]
    dummy_labels = [["root"] * len(s["words"]) for s in train_sentences]
    mst_uas, _ = accuracy(train_sentences, mst_train_heads, dummy_labels)
    print("    global Graph-MST UAS on training data (jackknifed): %.2f" % mst_uas)

    print("Training the global Graph-MST parser on all training sentences...")
    mst_model = GraphMSTParser(lexicon=lexicon_ltr)
    mst_model.train(train_sentences, train_tags, train_supertags, epochs=MST_EPOCHS, seed=1)
    other_mst_lists = []
    for j in range(len(other_sentence_lists)):
        sentences = other_sentence_lists[j]
        t_list = other_tags_lists[j]
        s_list = other_super_lists[j]
        m_list = [
            mst_model.predict_mst_info(
                sentences[k]["words"],
                t_list[k],
                s_list[k] if s_list is not None else None,
            )
            for k in range(len(sentences))
        ]
        other_mst_lists.append(m_list)
    mst_model = None
    gc.collect()
    return train_mst, other_mst_lists


def copy_known_sentences(train_sentences, sentences, all_heads, all_labels):
    """
    If a sentence has the same words (or unambiguous normalized words) as a
    training sentence, use the majority training parse for it.
    """
    exact_counts = {}
    norm_counts = {}

    for sentence in train_sentences:
        exact_key = "\t".join(sentence["words"])
        norm_key = "\t".join(normalize_word(w) for w in sentence["words"])
        ans = (tuple(sentence["heads"]), tuple(sentence["labels"]))

        ec = exact_counts.get(exact_key)
        if ec is None:
            ec = {}
            exact_counts[exact_key] = ec
        ec[ans] = ec.get(ans, 0) + 1

        nc = norm_counts.get(norm_key)
        if nc is None:
            nc = {}
            norm_counts[norm_key] = nc
        nc[ans] = nc.get(ans, 0) + 1

    exact_best = {}
    for key, counts in exact_counts.items():
        sorted_ans = sorted(counts.items(), key=lambda kv: -kv[1])
        total = sum(counts.values())
        if sorted_ans[0][1] >= 0.75 * total:
            exact_best[key] = (list(sorted_ans[0][0][0]), list(sorted_ans[0][0][1]))

    norm_best = {}
    for key, counts in norm_counts.items():
        if len(counts) == 1:
            ans = next(iter(counts.keys()))
            norm_best[key] = (list(ans[0]), list(ans[1]))

    copied = 0
    for k in range(len(sentences)):
        exact_key = "\t".join(sentences[k]["words"])
        answer = exact_best.get(exact_key)
        if answer is None and len(sentences[k]["words"]) >= 3:
            norm_key = "\t".join(normalize_word(w) for w in sentences[k]["words"])
            answer = norm_best.get(norm_key)
        if answer is not None:
            all_heads[k] = list(answer[0])
            all_labels[k] = list(answer[1])
            copied += 1
    return copied


def train_one_parser(
    system_type,
    direction,
    seed,
    label_names,
    train_sentences,
    train_tags,
    train_supertags,
    train_mst,
    sentence_lists,
    tags_lists,
    supertags_lists,
    mst_lists,
    lexicon_ltr=None,
    lexicon_rtl=None,
):
    """Train one parser (Arc-Hybrid or Arc-Eager) and return its (heads, labels, margins)."""
    if direction == "right-to-left":
        sentences_for_training = []
        tags_for_training = []
        super_for_training = []
        mst_for_training = []
        for k in range(len(train_sentences)):
            sentences_for_training.append(flip_sentence(train_sentences[k]))
            tags_for_training.append(flip_tags(train_tags[k]))
            super_for_training.append(
                flip_supertags(train_supertags[k]) if train_supertags[k] is not None else None
            )
            mst_for_training.append(
                flip_mst_info(train_mst[k]) if train_mst[k] is not None else None
            )
        lexicon = lexicon_rtl
    else:
        sentences_for_training = train_sentences
        tags_for_training = train_tags
        super_for_training = train_supertags
        mst_for_training = train_mst
        lexicon = lexicon_ltr

    if system_type == "arc-eager":
        parser = ArcEagerParser(label_names, lexicon=lexicon)
    else:
        parser = Parser(label_names, lexicon=lexicon)

    random_generator = random.Random(seed)
    for epoch in range(PARSER_EPOCHS):
        parser.train_one_epoch(
            sentences_for_training,
            tags_for_training,
            random_generator,
            super_for_training,
            mst_for_training,
            epoch=epoch,
        )
        print("    epoch", epoch + 1, "of", PARSER_EPOCHS, "done")
    parser.model.average_weights()

    results = []
    for j in range(len(sentence_lists)):
        sentences = sentence_lists[j]
        tags_list = tags_lists[j]
        super_list = supertags_lists[j]
        mst_list = mst_lists[j]
        all_heads = []
        all_labels = []
        all_margins = []
        for k in range(len(sentences)):
            stags = super_list[k] if super_list is not None else None
            minfo = mst_list[k] if mst_list is not None else None
            if direction == "right-to-left":
                flipped = flip_sentence(sentences[k])
                f_tags = flip_tags(tags_list[k])
                f_stags = flip_supertags(stags) if stags is not None else None
                f_minfo = flip_mst_info(minfo) if minfo is not None else None
                heads, labels, margins = parser.parse(
                    flipped["words"], f_tags, False, f_stags, f_minfo
                )
                heads, labels, margins = flip_result(heads, labels, margins)
            else:
                heads, labels, margins = parser.parse(
                    sentences[k]["words"], tags_list[k], False, stags, minfo
                )
            all_heads.append(heads)
            all_labels.append(labels)
            all_margins.append(margins)
        results.append((all_heads, all_labels, all_margins))
    return results


def _refine_ud_tree(words, heads, labels):
    """
    Apply deterministic Universal Dependencies v2 structural consistency rules
    to the globally decoded parse tree.
    """
    n = len(words) - 1
    if n <= 0:
        return heads, labels

    # 1. Punctuation label consistency
    for i in range(1, n + 1):
        if heads[i] != 0 and is_punct_word(words[i]):
            labels[i] = "punct"

    # 2. Multi-word proper noun (`flat`) continuity
    for i in range(2, n):
        if heads[i + 1] == i - 1 and labels[i + 1] == "flat":
            w_i = words[i]
            if w_i and w_i[0].isupper() and w_i.replace(".", "").isalpha():
                if heads[i] in (i - 1, i + 1) and labels[i] in ("case", "compound", "dep", "nmod"):
                    heads[i] = i - 1
                    labels[i] = "flat"

    children = [[] for _ in range(n + 1)]
    for i in range(1, n + 1):
        h = heads[i]
        if 0 <= h <= n:
            children[h].append(i)

    # 3. Prepositional/possessive phrase (`case` child) `obl` vs `nmod` / `nmod:poss`
    nominal_head_labels = {
        "obj", "iobj", "nsubj", "nsubj:pass",
        "nmod", "nmod:poss", "compound", "appos",
    }
    clausal_child_labels = {
        "cop", "aux", "aux:pass", "nsubj", "nsubj:pass", "csubj", "expl",
    }

    for i in range(1, n + 1):
        case_children = [c for c in children[i] if labels[c] == "case"]
        if not case_children:
            continue

        has_poss_case = any(
            normalize_word(words[c]) in ("'s", "'") for c in case_children
        )
        if has_poss_case and heads[i] > i and labels[i] in ("nmod", "compound", "obl", "dep"):
            labels[i] = "nmod:poss"
            continue

        h = heads[i]
        if labels[i] == "obl" and 1 <= h <= n and labels[h] in nominal_head_labels:
            head_is_predicate = any(labels[ch] in clausal_child_labels for ch in children[h])
            if not head_is_predicate:
                labels[i] = "nmod"

    return heads, labels


def vote(
    predictions,
    sentences,
    weights=None,
    tags_list=None,
    supertags_list=None,
    mst_list=None,
    tree_labeler=None,
):
    """
    Combine all parsers via:
      1. Margin-confidence-weighted arc score matrix construction (combining all
         transition parsers + the Global Graph-MST parser).
      2. Exact single-root Chu-Liu/Edmonds Maximum Spanning Arborescence decoding.
      3. Full-tree 2nd/3rd-order label scoring (`FullTreeLabeler`) blended with
         the transition ensemble's confidence-weighted label votes.
      4. Universal Dependencies v2 structural consistency refinement.
    """
    if weights is None:
        weights = [1.0] * len(predictions)

    voted_heads = []
    voted_labels = []

    for k in range(len(sentences)):
        words = sentences[k]["words"]
        n = len(words) - 1
        if n <= 0:
            voted_heads.append([-1])
            voted_labels.append([""])
            continue

        punct_flags = [False] + [is_punct_word(words[i]) for i in range(1, n + 1)]
        all_punct = all(punct_flags[1:])

        # Build (n + 1) x (n + 1) global arc score matrix
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

        # Add confidence-weighted votes from every transition parser
        for p_idx, pred_tuple in enumerate(predictions):
            all_heads = pred_tuple[0]
            all_margins = pred_tuple[2] if len(pred_tuple) > 2 else None
            w_p = weights[p_idx]
            for i in range(1, n + 1):
                h = all_heads[k][i]
                if 0 <= h <= n and h != i:
                    marg = all_margins[k][i] if all_margins is not None else 4.0
                    conf_mult = 0.85 + 0.30 * (min(max(marg, 0.0), 12.0) / 12.0)
                    # Small deterministic tie-break favoring earlier parsers
                    arc_scores[h, i] += w_p * conf_mult - 1.0e-5 * p_idx

        # Add Global Graph-MST structural votes when available
        if mst_list is not None and mst_list[k] is not None:
            minfo = mst_list[k]
            mh1 = minfo["heads"]
            mh2 = minfo["heads2"]
            mconf = minfo["conf"]
            for i in range(1, n + 1):
                h1 = mh1[i]
                h2 = mh2[i]
                if 0 <= h1 <= n and h1 != i:
                    bonus = 0.95 if mconf[i] == "HIGH" else (0.65 if mconf[i] == "MID" else 0.40)
                    arc_scores[h1, i] += bonus
                if 0 <= h2 <= n and h2 != i:
                    arc_scores[h2, i] += 0.20

        # Decode exact single-root Maximum Spanning Arborescence via Chu-Liu/Edmonds
        heads = chu_liu_edmonds(arc_scores)

        # Compute full-tree label scores when `tree_labeler` is available
        tree_scores = None
        if tree_labeler is not None and tags_list is not None:
            stags = supertags_list[k] if supertags_list is not None else None
            tree_scores = tree_labeler.score_sentence_labels(words, tags_list[k], stags, heads)

        labels = [""] * (n + 1)
        for i in range(1, n + 1):
            h = heads[i]
            if h == 0:
                labels[i] = "root"
                continue

            label_votes = {}
            first_seen = {}
            for p_idx, pred_tuple in enumerate(predictions):
                all_heads = pred_tuple[0]
                all_labels = pred_tuple[1]
                all_margins = pred_tuple[2] if len(pred_tuple) > 2 else None
                lab = all_labels[k][i]
                if lab == "root":
                    continue
                w_p = weights[p_idx]
                marg = all_margins[k][i] if all_margins is not None else 4.0
                conf_mult = 0.85 + 0.30 * (min(max(marg, 0.0), 12.0) / 12.0)
                if all_heads[k][i] == h:
                    vote_w = w_p * conf_mult
                else:
                    vote_w = 0.22 * w_p
                label_votes[lab] = label_votes.get(lab, 0.0) + vote_w
                if lab not in first_seen:
                    first_seen[lab] = p_idx

            if tree_scores is not None and tree_scores[i] is not None:
                t_map = tree_scores[i]
                if t_map:
                    max_ts = max(t_map.values())
                    for lab, ts in t_map.items():
                        diff = ts - max_ts
                        if diff >= -8.0:
                            bonus = 1.15 * math.exp(0.25 * diff)
                            label_votes[lab] = label_votes.get(lab, 0.0) + bonus
                            if lab not in first_seen:
                                first_seen[lab] = len(predictions)

            if label_votes:
                best_lab = max(
                    label_votes.keys(),
                    key=lambda l: (label_votes[l], -first_seen.get(l, 999)),
                )
            else:
                best_lab = "dep"
            labels[i] = best_lab

        heads, labels = _refine_ud_tree(words, heads, labels)
        voted_heads.append(heads)
        voted_labels.append(labels)

    return voted_heads, voted_labels


def check_mode(all_train):
    """Train on 90% of the training data, measure the score on the other 10%."""
    train_sentences = []
    dev_sentences = []
    for k in range(len(all_train)):
        if k % 10 == 0:
            dev_sentences.append(all_train[k])
        else:
            train_sentences.append(all_train[k])
    print("Training sentences:", len(train_sentences), " check sentences:", len(dev_sentences))

    label_names = get_label_names(all_train)
    train_tags, other_tags, train_supertags, other_supertags = make_tags(
        train_sentences, [dev_sentences]
    )
    dev_tags = other_tags[0]
    dev_supertags = other_supertags[0]
    print("    tag accuracy on check data: %.2f" % tag_accuracy(make_tag_targets(dev_sentences), dev_tags))
    if USE_SECOND_PASS_TAGGER:
        print("    supertag accuracy on check data: %.2f" % tag_accuracy(make_supertag_targets(dev_sentences), dev_supertags))

    tag_Target_ltr = make_tag_targets(train_sentences)
    lexicon_ltr = build_lexicon(tag_Target_ltr)
    flipped_tag_Target = [flip_sentence(s) for s in tag_Target_ltr]
    for k in range(len(flipped_tag_Target)):
        flipped_tag_Target[k]["labels"] = flip_tags(tag_Target_ltr[k]["labels"])
    lexicon_rtl = build_lexicon(flipped_tag_Target)

    train_mst, other_mst = make_mst_guides(
        train_sentences,
        train_tags,
        train_supertags,
        [dev_sentences],
        [dev_tags],
        [dev_supertags],
        lexicon_ltr,
    )
    dev_mst = other_mst[0]
    if USE_SECOND_PASS_TAGGER:
        dev_mst_heads = [m["heads"] for m in dev_mst]
        dummy_labels = [["root"] * len(s["words"]) for s in dev_sentences]
        mst_dev_uas, _ = accuracy(dev_sentences, dev_mst_heads, dummy_labels)
        print("    global Graph-MST UAS on check data: %.2f" % mst_dev_uas)

    tree_labeler = None
    if USE_SECOND_PASS_TAGGER:
        print("Training the full-tree 2nd/3rd-order labeler...")
        tree_labeler = FullTreeLabeler(label_names, lexicon=lexicon_ltr)
        tree_labeler.train(
            train_sentences,
            train_tags,
            train_supertags,
            train_mst,
            epochs=TREE_LABELER_EPOCHS,
            seed=1,
        )

    predictions = []
    weights = []
    for (system_type, direction, seed, weight) in PARSERS:
        print(
            "Training parser: %s (%s, seed %d, weight %.2f)"
            % (system_type, direction, seed, weight)
        )
        results = train_one_parser(
            system_type,
            direction,
            seed,
            label_names,
            train_sentences,
            train_tags,
            train_supertags,
            train_mst,
            [dev_sentences],
            [dev_tags],
            [dev_supertags],
            [dev_mst],
            lexicon_ltr=lexicon_ltr,
            lexicon_rtl=lexicon_rtl,
        )
        predictions.append(results[0])
        weights.append(weight)
        gc.collect()
        uas, las = accuracy(dev_sentences, results[0][0], results[0][1])
        print("    this parser:      UAS %.2f  LAS %.2f" % (uas, las))
        voted_heads, voted_labels = vote(
            predictions,
            dev_sentences,
            weights,
            tags_list=dev_tags,
            supertags_list=dev_supertags,
            mst_list=dev_mst,
            tree_labeler=tree_labeler,
        )
        uas, las = accuracy(dev_sentences, voted_heads, voted_labels)
        print("    Chu-Liu/Edmonds ensemble (%d parsers + MST): UAS %.2f  LAS %.2f   <- LAS is the competition score" % (len(predictions), uas, las))

    if COPY_KNOWN_SENTENCES:
        copied = copy_known_sentences(train_sentences, dev_sentences, voted_heads, voted_labels)
        uas, las = accuracy(dev_sentences, voted_heads, voted_labels)
        print("After copying %d known sentences: UAS %.2f  LAS %.2f" % (copied, uas, las))


def final_mode(train_sentences):
    """Train on all training data and write the submission."""
    test_sentences = read_sentences(os.path.join(DATA_FOLDER, "test.csv"))
    print("Training sentences:", len(train_sentences), " test sentences:", len(test_sentences))

    label_names = get_label_names(train_sentences)
    train_tags, other_tags, train_supertags, other_supertags = make_tags(
        train_sentences, [test_sentences]
    )
    test_tags = other_tags[0]
    test_supertags = other_supertags[0]

    tag_Target_ltr = make_tag_targets(train_sentences)
    lexicon_ltr = build_lexicon(tag_Target_ltr)
    flipped_tag_Target = [flip_sentence(s) for s in tag_Target_ltr]
    for k in range(len(flipped_tag_Target)):
        flipped_tag_Target[k]["labels"] = flip_tags(tag_Target_ltr[k]["labels"])
    lexicon_rtl = build_lexicon(flipped_tag_Target)

    train_mst, other_mst = make_mst_guides(
        train_sentences,
        train_tags,
        train_supertags,
        [test_sentences],
        [test_tags],
        [test_supertags],
        lexicon_ltr,
    )
    test_mst = other_mst[0]

    tree_labeler = None
    if USE_SECOND_PASS_TAGGER:
        print("Training the full-tree 2nd/3rd-order labeler...")
        tree_labeler = FullTreeLabeler(label_names, lexicon=lexicon_ltr)
        tree_labeler.train(
            train_sentences,
            train_tags,
            train_supertags,
            train_mst,
            epochs=TREE_LABELER_EPOCHS,
            seed=1,
        )

    test_predictions = []
    weights = []
    for (system_type, direction, seed, weight) in PARSERS:
        print(
            "Training parser: %s (%s, seed %d, weight %.2f)"
            % (system_type, direction, seed, weight)
        )
        results = train_one_parser(
            system_type,
            direction,
            seed,
            label_names,
            train_sentences,
            train_tags,
            train_supertags,
            train_mst,
            [test_sentences],
            [test_tags],
            [test_supertags],
            [test_mst],
            lexicon_ltr=lexicon_ltr,
            lexicon_rtl=lexicon_rtl,
        )
        test_predictions.append(results[0])
        weights.append(weight)
        gc.collect()

    test_heads, test_labels = vote(
        test_predictions,
        test_sentences,
        weights,
        tags_list=test_tags,
        supertags_list=test_supertags,
        mst_list=test_mst,
        tree_labeler=tree_labeler,
    )
    if COPY_KNOWN_SENTENCES:
        copied = copy_known_sentences(train_sentences, test_sentences, test_heads, test_labels)
        print("Copied the training answer for", copied, "test sentences")

    # Collect "HEAD|LABEL" for every row ID.
    answer_of_id = {}
    for k in range(len(test_sentences)):
        for i in range(1, len(test_sentences[k]["words"])):
            row_id = test_sentences[k]["row_ids"][i - 1]
            answer_of_id[row_id] = str(test_heads[k][i]) + "|" + test_labels[k][i]

    for k in range(len(train_sentences)):
        for i in range(1, len(train_sentences[k]["words"])):
            row_id = train_sentences[k]["row_ids"][i - 1]
            answer_of_id[row_id] = str(train_sentences[k]["heads"][i]) + "|" + train_sentences[k]["labels"][i]

    sample_ids, header = read_sample_ids(os.path.join(DATA_FOLDER, "sample.csv"))
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    output_path = os.path.join(OUTPUT_FOLDER, "submission.csv")
    write_submission(output_path, header, sample_ids, answer_of_id)
    print("Wrote", len(sample_ids), "rows to", output_path)


def main():
    global PARSERS, USE_SECOND_PASS_TAGGER, COPY_KNOWN_SENTENCES
    if "--one-parser" in sys.argv:
        PARSERS = PARSERS[:1]
    if "--no-second-pass" in sys.argv:
        USE_SECOND_PASS_TAGGER = False
    if "--no-copy" in sys.argv:
        COPY_KNOWN_SENTENCES = False
    start_time = time.time()
    train_sentences = read_sentences(os.path.join(DATA_FOLDER, "train.csv"))
    if "--check" in sys.argv:
        check_mode(train_sentences)
    else:
        final_mode(train_sentences)
    print("Total time: %.1f minutes" % ((time.time() - start_time) / 60.0))


if __name__ == "__main__":
    main()
