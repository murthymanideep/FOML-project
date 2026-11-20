"""
Main script: trains the Induced UPOS tagger, bidirectional two-pass syntactic
tagger, global-context head-offset supertagger, 2-seed Global First-Order
Graph-MST structured perceptron, full-tree 2nd/3rd-order label classifier, and
3-system (Arc-Hybrid + Arc-Eager + Arc-Standard, L-to-R + R-to-L) graph-guided
transition parser ensemble, then decodes the globally optimal parse tree via
Chu-Liu/Edmonds Maximum Spanning Arborescence and writes the submission file.

    python train_and_predict.py           -> train on all training data,
                                             write outputs/submission.csv
    python train_and_predict.py --check   -> train on 90% of the training data
                                             and print the score on the other 10%

Extra options (can be combined with --check):
    --one-parser       train only the first parser (fast, for experiments)
    --no-second-pass   do not use the second-pass tagger, supertagger & MST guide

Run --check FIRST. At its end it learns, from the check data:
  * the learned combiner (logistic regression, see combiner.py), and
  * the best set of rule-based fixes,
prints an error report, and saves these choices in best_settings.json.
The normal run then uses them automatically (the combiner only if it beat
the normal vote on the check data).
"""

import gc
import math
import os
import json
import random
import sys
import time

from data_utils import (
    read_sentences,
    read_sample_ids,
    write_submission,
    accuracy,
    normalize_word,
    find_punct_forms,
    find_possessive_markers,
    make_hashing_repeatable,
    build_lexicon,
    build_word_clusters,
)
from tagger import (
    train_tagger,
    jackknife_tags,
    make_upos_targets,
    make_tag_targets,
    make_supertag_targets,
    tag_accuracy,
)
from parser import (
    Parser,
    ArcEagerParser,
    ArcStandardParser,
    GraphMSTParser,
    SecondOrderTreeModel,
    FullTreeLabeler,
    jackknife_mst,
    chu_liu_edmonds,
    flip_list,
    flip_sentence,
    flip_tags,
    flip_supertags,
    flip_mst_info,
    flip_result,
)

import numpy as np

import combiner


# ----------------------------- settings -----------------------------

DATA_FOLDER = "data"
OUTPUT_FOLDER = "outputs"

UPOS_EPOCHS = 6
TAGGER_EPOCHS = 6
SUPERTAGGER_EPOCHS = 5
MST_EPOCHS = 8
SECOND_ORDER_EPOCHS = 5
TREE_LABELER_EPOCHS = 6
JACKKNIFE_FOLDS = 5
PARSER_EPOCHS = 12

# 3-System Heterogeneous Graph-Guided Transition Parser Ensemble:
# (system_type, reading_direction, random_seed, vote_weight)
PARSERS = [
    ("arc-hybrid",   "left-to-right", 1, 1.35),
    ("arc-eager",    "left-to-right", 1, 1.25),
    ("arc-standard", "left-to-right", 1, 1.20),
    ("arc-hybrid",   "right-to-left", 1, 1.05),
    ("arc-eager",    "right-to-left", 1, 0.95),
    ("arc-standard", "right-to-left", 1, 0.95),
    ("arc-hybrid",   "left-to-right", 2, 1.30),
    ("arc-eager",    "left-to-right", 2, 1.20),
    ("arc-standard", "left-to-right", 2, 1.15),
    ("arc-hybrid",   "right-to-left", 2, 1.00),
    ("arc-hybrid",   "left-to-right", 3, 1.30),
    ("arc-eager",    "left-to-right", 3, 1.20),
]

USE_SECOND_PASS_TAGGER = True

# Check split (--check): the training sentences are cut into groups of this
# many neighbouring sentences, and every 10th group is the check data.
# Neighbouring sentences mostly come from the same document, so the check
# data has documents the models never saw, like the real test data.
# (Taking every 10th single sentence made the check far too easy.)
CHECK_GROUP_SIZE = 100

# Rule-based fixes applied after the vote. Each one can be switched off.
#   punct    : true punctuation always gets the label "punct"
#   flat     : continue multi-word names ("flat")
#   passive  : "nsubj" -> "nsubj:pass" when the verb has "aux:pass"
#   obl_nmod : swap "obl" / "nmod" depending on the kind of head
# --check prints the score with each rule on and off.
RULES = {
    "punct": True,
    "flat": True,
    "passive": True,
    "obl_nmod": True,
}

# The choices learned by --check (learned combiner + best rules). The normal
# run uses them if this file exists. It is a small text file in the project
# folder (not in outputs/), so it can be committed with the code: whoever
# runs "python train_and_predict.py" later gets exactly the same choices.
BEST_SETTINGS = "best_settings.json"

# The learned combiner is only used if it beat the normal vote on the check
# data by at least this much (in LAS points).
COMBINER_MIN_GAIN = 0.10

# --------------------------------------------------------------------


def get_label_names(sentences):
    names = set()
    for sentence in sentences:
        for i in range(1, len(sentence["labels"])):
            names.add(sentence["labels"][i])
    return sorted(names)


def make_tags(train_sentences, other_sentence_lists, word_clusters=None):
    """
    Tag the training sentences with 5-fold jackknifing, and the other sentences
    (dev or test) with taggers trained on all training sentences:
      0. Induced Universal POS (`UPOS`) tagger (17 syntactic classes induced from gold trees)
      1. Pass-1 Forward (Left-to-Right) tagger conditioned on `UPOS`
      2. Pass-1 Backward (Right-to-Left) tagger conditioned on `UPOS`
      3. Pass-2 Stacked tagger conditioned on Forward + Backward tags + `UPOS` + Global Lookahead
      4. Relative head-offset supertagger conditioned on Pass-2 + Backward tags + `UPOS` + Global Lookahead
    """
    # ----- 0. Induced Universal POS (UPOS) tagger -----
    upos_sentences = make_upos_targets(train_sentences)
    upos_names = get_label_names(upos_sentences)

    print("Training the Induced UPOS tagger (jackknifing for training data)...")
    train_upos = jackknife_tags(
        upos_sentences,
        upos_names,
        UPOS_EPOCHS,
        JACKKNIFE_FOLDS,
        direction="left-to-right",
        word_clusters=word_clusters,
    )
    print("    Induced UPOS accuracy on training data (jackknifed): %.2f" % tag_accuracy(upos_sentences, train_upos))

    print("Training the Induced UPOS tagger on all training sentences...")
    upos_tagger = train_tagger(
        upos_sentences,
        upos_names,
        UPOS_EPOCHS,
        direction="left-to-right",
        word_clusters=word_clusters,
    )
    other_upos_lists = []
    for sentences in other_sentence_lists:
        u_list = [upos_tagger.tag(sentence["words"]) for sentence in sentences]
        other_upos_lists.append(u_list)
    upos_tagger = None
    gc.collect()

    # ----- 1. First-pass forward (Left-to-Right) tagger -----
    tag_sentences = make_tag_targets(train_sentences)
    tag_names = get_label_names(tag_sentences)

    print("Training the first-pass L-to-R tagger (jackknifing for training data)...")
    train_tags_fwd = jackknife_tags(
        tag_sentences,
        tag_names,
        TAGGER_EPOCHS,
        JACKKNIFE_FOLDS,
        direction="left-to-right",
        upos_tags_list=train_upos,
        word_clusters=word_clusters,
    )
    print("    L-to-R tag accuracy on training data (jackknifed): %.2f" % tag_accuracy(tag_sentences, train_tags_fwd))

    print("Training the first-pass L-to-R tagger on all training sentences...")
    tagger_fwd = train_tagger(
        tag_sentences,
        tag_names,
        TAGGER_EPOCHS,
        direction="left-to-right",
        upos_tags_list=train_upos,
        word_clusters=word_clusters,
    )
    other_tags_fwd = []
    for j in range(len(other_sentence_lists)):
        sentences = other_sentence_lists[j]
        tags_list = [
            tagger_fwd.tag(sentences[k]["words"], upos_tags=other_upos_lists[j][k])
            for k in range(len(sentences))
        ]
        other_tags_fwd.append(tags_list)
    tagger_fwd = None
    gc.collect()

    if not USE_SECOND_PASS_TAGGER:
        empty_super = [[None] * len(s) for s in other_sentence_lists]
        return (
            train_tags_fwd,
            other_tags_fwd,
            [None] * len(train_sentences),
            empty_super,
            train_upos,
            other_upos_lists,
        )

    # ----- 2. First-pass backward (Right-to-Left) tagger -----
    print("Training the first-pass R-to-L tagger (jackknifing for training data)...")
    train_tags_bwd = jackknife_tags(
        tag_sentences,
        tag_names,
        TAGGER_EPOCHS,
        JACKKNIFE_FOLDS,
        direction="right-to-left",
        upos_tags_list=train_upos,
        word_clusters=word_clusters,
    )
    print("    R-to-L tag accuracy on training data (jackknifed): %.2f" % tag_accuracy(tag_sentences, train_tags_bwd))

    print("Training the first-pass R-to-L tagger on all training sentences...")
    tagger_bwd = train_tagger(
        tag_sentences,
        tag_names,
        TAGGER_EPOCHS,
        direction="right-to-left",
        upos_tags_list=train_upos,
        word_clusters=word_clusters,
    )
    other_tags_bwd = []
    for j in range(len(other_sentence_lists)):
        sentences = other_sentence_lists[j]
        tags_list = [
            tagger_bwd.tag(sentences[k]["words"], upos_tags=other_upos_lists[j][k])
            for k in range(len(sentences))
        ]
        other_tags_bwd.append(tags_list)
    tagger_bwd = None
    gc.collect()

    # ----- 3. Second pass: stacked bidirectional tagger with global lookahead -----
    print("Training the second-pass bidirectional tagger (jackknifing)...")
    train_tags_2 = jackknife_tags(
        tag_sentences,
        tag_names,
        TAGGER_EPOCHS,
        JACKKNIFE_FOLDS,
        first_tags_list=train_tags_fwd,
        direction="left-to-right",
        backward_tags_list=train_tags_bwd,
        upos_tags_list=train_upos,
        word_clusters=word_clusters,
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
        upos_tags_list=train_upos,
        word_clusters=word_clusters,
    )
    other_tags_lists_2 = []
    for j in range(len(other_sentence_lists)):
        sentences = other_sentence_lists[j]
        tags_list = [
            tagger_2.tag(
                sentences[k]["words"],
                first_tags=other_tags_fwd[j][k],
                backward_tags=other_tags_bwd[j][k],
                upos_tags=other_upos_lists[j][k],
            )
            for k in range(len(sentences))
        ]
        other_tags_lists_2.append(tags_list)
    tagger_2 = None
    gc.collect()

    # ----- 4. Supertagger: relative head-offset + head coarse group -----
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
        upos_tags_list=train_upos,
        word_clusters=word_clusters,
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
        upos_tags_list=train_upos,
        word_clusters=word_clusters,
    )
    other_super_lists = []
    for j in range(len(other_sentence_lists)):
        sentences = other_sentence_lists[j]
        s_list = [
            super_tagger.tag(
                sentences[k]["words"],
                first_tags=other_tags_lists_2[j][k],
                backward_tags=other_tags_bwd[j][k],
                upos_tags=other_upos_lists[j][k],
            )
            for k in range(len(sentences))
        ]
        other_super_lists.append(s_list)
    super_tagger = None
    gc.collect()

    return (
        train_tags_2,
        other_tags_lists_2,
        train_supertags,
        other_super_lists,
        train_upos,
        other_upos_lists,
    )


def make_mst_guides(
    train_sentences,
    train_tags,
    train_supertags,
    train_upos,
    other_sentence_lists,
    other_tags_lists,
    other_super_lists,
    other_upos_lists,
    lexicon_ltr,
    word_clusters=None,
):
    """
    Train the Global First-Order Graph-Based MST Structured Perceptron:
      - 5-fold jackknifing on `train_sentences` for zero-leakage stacking
      - 2-seed ensemble trained on all `train_sentences` for `other_sentence_lists`
    """
    if not USE_SECOND_PASS_TAGGER:
        empty_other = [[None] * len(s) for s in other_sentence_lists]
        return [None] * len(train_sentences), empty_other

    print("Training the global Graph-MST parser (jackknifing for training data)...")
    train_mst = jackknife_mst(
        train_sentences,
        train_tags,
        train_supertags,
        upos_list=train_upos,
        epochs=MST_EPOCHS,
        n_folds=JACKKNIFE_FOLDS,
        word_clusters=word_clusters,
    )
    mst_train_heads = [m["heads"] for m in train_mst]
    dummy_labels = [["root"] * len(s["words"]) for s in train_sentences]
    mst_uas, _ = accuracy(train_sentences, mst_train_heads, dummy_labels)
    print("    global Graph-MST UAS on training data (jackknifed): %.2f" % mst_uas)

    print("Training the 2-seed global Graph-MST parser on all training sentences...")
    mst_model_1 = GraphMSTParser(lexicon=lexicon_ltr)
    mst_model_1.train(
        train_sentences, train_tags, train_supertags, train_upos, epochs=MST_EPOCHS, seed=1
    )
    mst_model_2 = GraphMSTParser(lexicon=lexicon_ltr)
    mst_model_2.train(
        train_sentences, train_tags, train_supertags, train_upos, epochs=MST_EPOCHS, seed=2
    )
    other_mst_lists = []
    for j in range(len(other_sentence_lists)):
        sentences = other_sentence_lists[j]
        t_list = other_tags_lists[j]
        s_list = other_super_lists[j]
        u_list = other_upos_lists[j]
        m_list = [
            mst_model_1.predict_mst_info(
                sentences[k]["words"],
                t_list[k],
                s_list[k] if s_list is not None else None,
                u_list[k] if u_list is not None else None,
                second_model=mst_model_2,
            )
            for k in range(len(sentences))
        ]
        other_mst_lists.append(m_list)
    mst_model_1 = None
    mst_model_2 = None
    gc.collect()
    return train_mst, other_mst_lists


def train_one_parser(
    system_type,
    direction,
    seed,
    label_names,
    train_sentences,
    train_tags,
    train_supertags,
    train_mst,
    train_upos,
    sentence_lists,
    tags_lists,
    supertags_lists,
    mst_lists,
    upos_lists,
    lexicon_ltr=None,
    lexicon_rtl=None,
):
    """Train one parser (Arc-Hybrid, Arc-Eager, or Arc-Standard) and return its (heads, labels, margins)."""
    if direction == "right-to-left":
        sentences_for_training = []
        tags_for_training = []
        super_for_training = []
        mst_for_training = []
        upos_for_training = []
        for k in range(len(train_sentences)):
            sentences_for_training.append(flip_sentence(train_sentences[k]))
            tags_for_training.append(flip_tags(train_tags[k]))
            super_for_training.append(
                flip_supertags(train_supertags[k]) if train_supertags[k] is not None else None
            )
            mst_for_training.append(
                flip_mst_info(train_mst[k]) if train_mst[k] is not None else None
            )
            upos_for_training.append(
                flip_list(train_upos[k]) if train_upos is not None and train_upos[k] is not None else None
            )
        lexicon = lexicon_rtl
    else:
        sentences_for_training = train_sentences
        tags_for_training = train_tags
        super_for_training = train_supertags
        mst_for_training = train_mst
        upos_for_training = train_upos
        lexicon = lexicon_ltr

    if system_type == "arc-eager":
        parser = ArcEagerParser(label_names, lexicon=lexicon)
    elif system_type == "arc-standard":
        parser = ArcStandardParser(label_names, lexicon=lexicon)
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
            upos_for_training,
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
        u_list = upos_lists[j] if upos_lists is not None else None
        all_heads = []
        all_labels = []
        all_margins = []
        for k in range(len(sentences)):
            stags = super_list[k] if super_list is not None else None
            minfo = mst_list[k] if mst_list is not None else None
            utags = u_list[k] if u_list is not None else None
            if direction == "right-to-left":
                flipped = flip_sentence(sentences[k])
                f_tags = flip_tags(tags_list[k])
                f_stags = flip_supertags(stags) if stags is not None else None
                f_minfo = flip_mst_info(minfo) if minfo is not None else None
                f_utags = flip_list(utags) if utags is not None else None
                heads, labels, margins = parser.parse(
                    flipped["words"], f_tags, False, f_stags, f_minfo, f_utags
                )
                heads, labels, margins = flip_result(heads, labels, margins)
            else:
                heads, labels, margins = parser.parse(
                    sentences[k]["words"], tags_list[k], False, stags, minfo, utags
                )
            all_heads.append(heads)
            all_labels.append(labels)
            all_margins.append(margins)
        results.append((all_heads, all_labels, all_margins))
    return results


def _refine_ud_tree(words, heads, labels, upos=None, punct_forms=None, rules=None, possessive_markers=None):
    """
    Apply simple Universal Dependencies rules to one voted tree.

    rules       : which rules are on (see RULES at the top of this file)
    punct_forms : word forms that are almost always "punct" in the training
                  data (from find_punct_forms)
    possessive_markers : the possessive "case" words learned from the
                  training data (from find_possessive_markers)
    """
    if rules is None:
        rules = RULES
    n = len(words) - 1
    if n <= 0:
        return heads, labels

    # 1. True punctuation always gets the label "punct".
    #    Only forms that are "punct" in at least 98% of the training cases:
    #    symbols like "&", "$", "%", ":)" are NOT punctuation in this data.
    if rules["punct"] and punct_forms is not None:
        for i in range(1, n + 1):
            if heads[i] != 0 and words[i] in punct_forms:
                labels[i] = "punct"

    # 2. Multi-word proper noun (`flat`) continuity
    if rules["flat"]:
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

    # 3. Passive subject consistency (`aux:pass` child on predicate -> `nsubj:pass`)
    if rules["passive"]:
        for h in range(1, n + 1):
            has_aux_pass = any(labels[ch] == "aux:pass" for ch in children[h])
            has_cop_or_obj = any(labels[ch] in ("cop", "obj") for ch in children[h])
            if has_aux_pass and not has_cop_or_obj:
                for ch in children[h]:
                    if ch < h and labels[ch] == "nsubj":
                        labels[ch] = "nsubj:pass"

    if not rules["obl_nmod"]:
        return heads, labels

    # 4. Prepositional/possessive phrase (`case` child) `obl` <-> `nmod` / `nmod:poss`
    #    (This data uses the label names "nmod:unmarked", "obl:unmarked" and
    #    "nmod:desc"; the old names "nmod:tmod", "obl:npmod", ... never occur.)
    nominal_head_labels = {
        "obj", "iobj", "nsubj", "nsubj:pass",
        "nmod", "nmod:poss", "nmod:unmarked", "nmod:desc",
        "obl", "obl:unmarked",
        "compound", "appos", "vocative", "flat",
    }
    clausal_child_labels = {
        "cop", "aux", "aux:pass", "nsubj", "nsubj:pass", "csubj", "expl",
    }
    nominal_child_labels = {
        "det", "det:predet", "amod", "nmod:poss",
    }

    for i in range(1, n + 1):
        case_children = [c for c in children[i] if labels[c] == "case"]
        if not case_children:
            continue

        has_poss_case = False
        if possessive_markers is not None:
            for c in case_children:
                if normalize_word(words[c]) in possessive_markers:
                    has_poss_case = True
        if has_poss_case and heads[i] > i and labels[i] in ("nmod", "compound", "obl", "dep"):
            labels[i] = "nmod:poss"
            continue

        h = heads[i]
        if not (1 <= h <= n):
            continue

        head_is_predicate = any(labels[ch] in clausal_child_labels for ch in children[h])
        head_has_nom_mod = any(labels[ch] in nominal_child_labels for ch in children[h])
        hu = upos[h] if upos is not None else "NOUN"

        if labels[i] == "obl":
            if labels[h] in nominal_head_labels and not head_is_predicate and not hu.startswith("VERB"):
                labels[i] = "nmod"
            elif hu in ("NOUN", "PROPN", "PRON", "NUM") and labels[h] == "conj" and not head_is_predicate:
                labels[i] = "nmod"
        elif labels[i] == "nmod" and not has_poss_case:
            if (
                hu.startswith("VERB")
                and labels[h] in ("root", "conj", "advcl", "ccomp", "xcomp", "parataxis")
                and not head_has_nom_mod
                and not any(labels[ch] == "cop" for ch in children[h])
            ):
                labels[i] = "obl"

    return heads, labels


def apply_rules(sentences, all_heads, all_labels, upos_list, punct_forms, rules=None, possessive_markers=None):
    """Return copies of all trees with the rule-based fixes applied."""
    new_heads = []
    new_labels = []
    for k in range(len(sentences)):
        heads = list(all_heads[k])
        labels = list(all_labels[k])
        upos = None
        if upos_list is not None:
            upos = upos_list[k]
        heads, labels = _refine_ud_tree(sentences[k]["words"], heads, labels, upos, punct_forms, rules, possessive_markers)
        new_heads.append(heads)
        new_labels.append(labels)
    return new_heads, new_labels


def vote(
    predictions,
    sentences,
    weights=None,
    tags_list=None,
    supertags_list=None,
    mst_list=None,
    upos_list=None,
    tree_labeler=None,
    tree_labeler_2=None,
    second_order=None,
    punct_forms=None,
):
    """
    Combine all parsers via:
      1. Margin-confidence-weighted arc score matrix construction (combining all
         12 transition parsers across 3 transition systems + 2-seed Global Graph-MST).
      2. Single-root Chu-Liu/Edmonds Maximum Spanning Arborescence decoding,
         refined by the Second-Order Grandparent/Sibling/Valency/Crossing model.
      3. 2-seed Full-tree 2nd/3rd-order label scoring (`FullTreeLabeler`) blended
         with the transition ensemble's confidence-weighted label votes.
    The rule-based fixes are NOT applied here; call apply_rules afterwards.

    punct_forms: word forms that are almost always "punct" in the training
    data. Only these are blocked from being the head of a normal word.
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

        punct_flags = [False]
        for i in range(1, n + 1):
            punct_flags.append(punct_forms is not None and words[i] in punct_forms)
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

        cands_per_tok = [[] for _ in range(n + 1)]

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
                    arc_scores[h, i] += w_p * conf_mult - 1.0e-5 * p_idx
                    if h not in cands_per_tok[i]:
                        cands_per_tok[i].append(h)

        # Add 2-seed Global Graph-MST structural votes when available
        if mst_list is not None and mst_list[k] is not None:
            minfo = mst_list[k]
            mh1 = minfo["heads"]
            mh2 = minfo["heads2"]
            mh3 = minfo.get("heads3")
            mconf = minfo["conf"]
            for i in range(1, n + 1):
                h1 = mh1[i]
                h2 = mh2[i]
                h3 = mh3[i] if mh3 is not None else -1
                if 0 <= h1 <= n and h1 != i:
                    bonus = 1.05 if mconf[i] == "HIGH" else (0.72 if mconf[i] == "MID" else 0.45)
                    arc_scores[h1, i] += bonus
                    if h1 not in cands_per_tok[i]:
                        cands_per_tok[i].append(h1)
                if 0 <= h2 <= n and h2 != i:
                    arc_scores[h2, i] += 0.22
                    if h2 not in cands_per_tok[i]:
                        cands_per_tok[i].append(h2)
                if 0 <= h3 <= n and h3 != i:
                    arc_scores[h3, i] += 0.08
                    if h3 not in cands_per_tok[i]:
                        cands_per_tok[i].append(h3)

        # Decode exact single-root Maximum Spanning Arborescence via Chu-Liu/Edmonds
        heads = chu_liu_edmonds(arc_scores)

        # Refine with Second-Order Grandparent/Sibling/Valency/Crossing model when available
        utags = upos_list[k] if upos_list is not None else None
        stags = supertags_list[k] if supertags_list is not None else None
        if second_order is not None and tags_list is not None and n >= 2:
            heads = second_order.refine_tree(
                words,
                tags_list[k],
                stags,
                utags,
                arc_scores,
                cands_per_tok,
                init_heads=heads,
                alpha=0.22,
                n_passes=2,
            )

        # Compute full-tree label scores when `tree_labeler` is available
        tree_scores = None
        if tree_labeler is not None and tags_list is not None:
            tree_scores = tree_labeler.score_sentence_labels(
                words, tags_list[k], stags, heads, upos=utags, second_model=tree_labeler_2
            )

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
                            bonus = 1.25 * math.exp(0.25 * diff)
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

        voted_heads.append(heads)
        voted_labels.append(labels)

    return voted_heads, voted_labels


RULE_VARIANTS = [
    ("no rules", {"punct": False, "flat": False, "passive": False, "obl_nmod": False}),
    ("punct only", {"punct": True, "flat": False, "passive": False, "obl_nmod": False}),
    ("punct + flat", {"punct": True, "flat": True, "passive": False, "obl_nmod": False}),
    ("punct + passive", {"punct": True, "flat": False, "passive": True, "obl_nmod": False}),
    ("punct + obl_nmod", {"punct": True, "flat": False, "passive": False, "obl_nmod": True}),
    ("all rules", {"punct": True, "flat": True, "passive": True, "obl_nmod": True}),
]


def make_outputs(
    train_sentences,
    sentences,
    predictions,
    weights,
    tags,
    supertags,
    upos,
    mst,
    tree_labeler,
    punct_forms,
    tree_labeler_2=None,
    second_order=None,
):
    """Put everything the last step needs in one dictionary (plus word groups learned from the training data)."""
    outputs = {}
    outputs["sentences"] = sentences
    outputs["predictions"] = predictions
    outputs["weights"] = weights
    outputs["parsers"] = list(PARSERS)
    outputs["tags"] = tags
    outputs["supertags"] = supertags
    outputs["upos"] = upos
    outputs["mst"] = mst
    outputs["tree_labeler"] = tree_labeler
    outputs["tree_labeler_2"] = tree_labeler_2
    outputs["second_order"] = second_order
    outputs["punct_forms"] = punct_forms
    outputs["possessive_markers"] = find_possessive_markers(train_sentences)
    return outputs


def normal_vote(outputs):
    """The vote with hand-set weights (no rules yet)."""
    return vote(
        outputs["predictions"],
        outputs["sentences"],
        outputs["weights"],
        tags_list=outputs["tags"],
        supertags_list=outputs["supertags"],
        mst_list=outputs["mst"],
        upos_list=outputs["upos"],
        tree_labeler=outputs["tree_labeler"],
        tree_labeler_2=outputs.get("tree_labeler_2"),
        second_order=outputs.get("second_order"),
        punct_forms=outputs["punct_forms"],
    )


def finish_trees(outputs, heads, labels, rules):
    """Apply the chosen rule-based fixes."""
    sentences = outputs["sentences"]
    heads, labels = apply_rules(sentences, heads, labels, outputs["upos"], outputs["punct_forms"], rules,
                                outputs["possessive_markers"])
    return heads, labels


def choose_best_rules(outputs, voted_heads, voted_labels):
    """Print the check score with each set of rule-based fixes; return the best set."""
    print("Rule-based fixes on the voted trees:")
    best_rules = None
    best_las = -1.0
    best_name = ""
    for (name, rules) in RULE_VARIANTS:
        heads, labels = finish_trees(outputs, voted_heads, voted_labels, rules)
        uas, las = accuracy(outputs["sentences"], heads, labels)
        print("    %-18s UAS %.2f  LAS %.2f" % (name, uas, las))
        if las > best_las:
            best_las = las
            best_rules = rules
            best_name = name
    print("    -> best: %s" % best_name)
    return best_rules


def most_common(counts, how_many):
    """The keys with the biggest counts, biggest first."""
    pairs = []
    for key in counts:
        pairs.append((counts[key], key))
    pairs.sort(reverse=True)
    return pairs[:how_many]


def print_error_report(sentences, heads, labels):
    """Where are the remaining mistakes?"""
    total = 0
    wrong_head = 0
    wrong_label = 0
    total_of = {}
    wrong_of = {}
    mix_ups = {}
    for k in range(len(sentences)):
        for i in range(1, len(sentences[k]["words"])):
            gold_head = sentences[k]["heads"][i]
            gold_label = sentences[k]["labels"][i]
            total = total + 1
            total_of[gold_label] = total_of.get(gold_label, 0) + 1
            if heads[k][i] != gold_head:
                wrong_head = wrong_head + 1
                wrong_of[gold_label] = wrong_of.get(gold_label, 0) + 1
            elif labels[k][i] != gold_label:
                wrong_label = wrong_label + 1
                wrong_of[gold_label] = wrong_of.get(gold_label, 0) + 1
                mix_up = gold_label + " -> " + labels[k][i]
                mix_ups[mix_up] = mix_ups.get(mix_up, 0) + 1

    print("Error report (%d words):" % total)
    print("    wrong head:                 %5d  (%.2f%% of all words)" % (wrong_head, 100.0 * wrong_head / total))
    print("    right head but wrong label: %5d  (%.2f%% of all words)" % (wrong_label, 100.0 * wrong_label / total))
    print("  Correct labels with the most errors (errors / count, points lost):")
    for (count, label) in most_common(wrong_of, 15):
        print("    %-15s %5d / %5d   -%.2f" % (label, count, total_of[label], 100.0 * count / total))
    print("  Most common label mix-ups when the head is right (correct -> predicted):")
    for (count, mix_up) in most_common(mix_ups, 12):
        print("    %-32s %5d" % (mix_up, count))


def learn_best_settings(outputs):
    """
    End of --check: choose the best rules, test the learned combiner fairly
    (2-fold cross-validation), print an error report, and save the choices
    in BEST_SETTINGS for the normal run.
    """
    sentences = outputs["sentences"]
    voted_heads, voted_labels = normal_vote(outputs)
    best_rules = choose_best_rules(outputs, voted_heads, voted_labels)

    print("Learned combiner (logistic regression, 2-fold cross-validation on the check data):")
    hand_heads, hand_labels = finish_trees(outputs, voted_heads, voted_labels, best_rules)
    uas, hand_las = accuracy(sentences, hand_heads, hand_labels)
    print("    normal vote        UAS %.2f  LAS %.2f" % (uas, hand_las))
    cv_heads, cv_labels = combiner.cross_validate(outputs)
    cv_heads, cv_labels = finish_trees(outputs, cv_heads, cv_labels, best_rules)
    uas, combiner_las = accuracy(sentences, cv_heads, cv_labels)
    print("    learned combiner   UAS %.2f  LAS %.2f" % (uas, combiner_las))

    use_combiner = combiner_las >= hand_las + COMBINER_MIN_GAIN
    if use_combiner:
        print("    -> the normal run will use the learned combiner (+%.2f)" % (combiner_las - hand_las))
        best_las = combiner_las
        print_error_report(sentences, cv_heads, cv_labels)
    else:
        print("    -> the normal run will use the normal vote (the combiner did not gain %.2f)" % COMBINER_MIN_GAIN)
        best_las = hand_las
        print_error_report(sentences, hand_heads, hand_labels)
    print("FINAL check score with the best choices: LAS %.2f   <- LAS is the competition score" % best_las)

    # Learn the combiner on ALL check sentences, for the normal run.
    model = combiner.train_combiner(outputs, list(range(len(sentences))))
    parsers = []
    for parser_settings in model["parsers"]:
        parsers.append(list(parser_settings))
    settings = {}
    settings["rules"] = best_rules
    settings["use_combiner"] = bool(use_combiner)
    settings["normal_vote_las"] = round(hand_las, 2)
    settings["combiner_las"] = round(combiner_las, 2)
    settings["combiner"] = {
        "arc_weights": [float(x) for x in model["arc_weights"]],
        "label_weights": [float(x) for x in model["label_weights"]],
        "parsers": parsers,
        "uses_mst": bool(model["uses_mst"]),
        "uses_labeler": bool(model["uses_labeler"]),
        "uses_second_order": bool(model["uses_second_order"]),
    }
    with open(BEST_SETTINGS, "w", encoding="utf-8") as settings_file:
        json.dump(settings, settings_file, indent=1)
    print("Saved the best choices in", BEST_SETTINGS, "(the normal run uses them; commit this file with the code)")


def final_trees(outputs):
    """
    End of the normal run: the trees for the test sentences, with the best
    choices from --check (if BEST_SETTINGS exists).
    """
    rules = RULES
    model = None
    if not os.path.exists(BEST_SETTINGS):
        print("No", BEST_SETTINGS, "found (run --check first): using RULES and the normal vote")
    else:
        with open(BEST_SETTINGS, "r", encoding="utf-8") as settings_file:
            settings = json.load(settings_file)
        rules = settings["rules"]
        print("Using the best choices from the check run: rules %s" % rules)
        if settings["use_combiner"]:
            saved = settings["combiner"]
            model = {}
            model["arc_weights"] = np.array(saved["arc_weights"])
            model["label_weights"] = np.array(saved["label_weights"])
            current_parsers = []
            for parser_settings in outputs["parsers"]:
                current_parsers.append(list(parser_settings))
            same_mst = saved["uses_mst"] == (outputs["mst"] is not None and outputs["mst"][0] is not None)
            same_labeler = saved["uses_labeler"] == (outputs["tree_labeler"] is not None)
            same_so = saved.get("uses_second_order") == (outputs.get("second_order") is not None)
            if saved["parsers"] != current_parsers or not same_mst or not same_labeler or not same_so:
                print("The check run used different parsers or options: cannot use the learned combiner.")
                model = None

    if model is not None:
        print("Using the learned combiner (check data: %.2f LAS, normal vote %.2f)"
              % (settings["combiner_las"], settings["normal_vote_las"]))
        heads_of, labels_of = combiner.combine(outputs, model)
        heads = []
        labels = []
        for k in range(len(outputs["sentences"])):
            heads.append(heads_of[k])
            labels.append(labels_of[k])
    else:
        print("Using the normal vote")
        heads, labels = normal_vote(outputs)
    return finish_trees(outputs, heads, labels, rules)


def write_final_submission(test_sentences, test_heads, test_labels, train_sentences):
    """Write outputs/submission.csv in the format of data/sample.csv."""
    # Collect "HEAD|LABEL" for every row ID.
    answer_of_id = {}
    for k in range(len(test_sentences)):
        for i in range(1, len(test_sentences[k]["words"])):
            row_id = test_sentences[k]["row_ids"][i - 1]
            answer_of_id[row_id] = str(test_heads[k][i]) + "|" + test_labels[k][i]

    # The sample file also lists the training rows. Kaggle only scores the
    # test rows, so we simply fill the training rows with their answers.
    for k in range(len(train_sentences)):
        for i in range(1, len(train_sentences[k]["words"])):
            row_id = train_sentences[k]["row_ids"][i - 1]
            answer_of_id[row_id] = str(train_sentences[k]["heads"][i]) + "|" + train_sentences[k]["labels"][i]

    sample_ids, header = read_sample_ids(os.path.join(DATA_FOLDER, "sample.csv"))
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    output_path = os.path.join(OUTPUT_FOLDER, "submission.csv")
    write_submission(output_path, header, sample_ids, answer_of_id)
    print("Wrote", len(sample_ids), "rows to", output_path)


def check_mode(all_train):
    """Train on 90% of the training data, measure the score on the other 10%."""
    train_sentences = []
    dev_sentences = []
    for k in range(len(all_train)):
        group = k // CHECK_GROUP_SIZE
        if group % 10 == 0:
            dev_sentences.append(all_train[k])
        else:
            train_sentences.append(all_train[k])
    print("Training sentences:", len(train_sentences), " check sentences:", len(dev_sentences))
    print("(check data = every 10th group of %d neighbouring sentences)" % CHECK_GROUP_SIZE)

    label_names = get_label_names(all_train)
    punct_forms = find_punct_forms(train_sentences)

    print("Learning unsupervised Bigram Class-HMM word clusters from training words...")
    word_clusters = build_word_clusters(train_sentences)
    print("    clustered %d vocabulary entries (64 fine / 16 coarse)" % len(word_clusters))

    (
        train_tags,
        other_tags,
        train_supertags,
        other_supertags,
        train_upos,
        other_upos,
    ) = make_tags(train_sentences, [dev_sentences], word_clusters=word_clusters)
    dev_tags = other_tags[0]
    dev_supertags = other_supertags[0]
    dev_upos = other_upos[0]
    print("    UPOS accuracy on check data: %.2f" % tag_accuracy(make_upos_targets(dev_sentences), dev_upos))
    print("    tag accuracy on check data: %.2f" % tag_accuracy(make_tag_targets(dev_sentences), dev_tags))
    if USE_SECOND_PASS_TAGGER:
        print("    supertag accuracy on check data: %.2f" % tag_accuracy(make_supertag_targets(dev_sentences), dev_supertags))

    tag_Target_ltr = make_tag_targets(train_sentences)
    lexicon_ltr = build_lexicon(tag_Target_ltr, word_clusters=word_clusters)
    flipped_tag_Target = [flip_sentence(s) for s in tag_Target_ltr]
    for k in range(len(flipped_tag_Target)):
        flipped_tag_Target[k]["labels"] = flip_tags(tag_Target_ltr[k]["labels"])
    lexicon_rtl = build_lexicon(flipped_tag_Target, word_clusters=word_clusters)

    train_mst, other_mst = make_mst_guides(
        train_sentences,
        train_tags,
        train_supertags,
        train_upos,
        [dev_sentences],
        [dev_tags],
        [dev_supertags],
        [dev_upos],
        lexicon_ltr,
        word_clusters=word_clusters,
    )
    dev_mst = other_mst[0]
    if USE_SECOND_PASS_TAGGER:
        dev_mst_heads = [m["heads"] for m in dev_mst]
        dummy_labels = [["root"] * len(s["words"]) for s in dev_sentences]
        mst_dev_uas, _ = accuracy(dev_sentences, dev_mst_heads, dummy_labels)
        print("    2-seed global Graph-MST UAS on check data: %.2f" % mst_dev_uas)

    second_order = None
    tree_labeler = None
    tree_labeler_2 = None
    if USE_SECOND_PASS_TAGGER:
        print("Training the 2nd-order Grandparent/Sibling/Valency graph tree model...")
        second_order = SecondOrderTreeModel(lexicon=lexicon_ltr)
        second_order.train(
            train_sentences,
            train_tags,
            train_supertags,
            train_mst,
            train_upos,
            epochs=SECOND_ORDER_EPOCHS,
            seed=1,
        )

        print("Training the 2-seed full-tree 2nd/3rd-order labeler...")
        tree_labeler = FullTreeLabeler(label_names, lexicon=lexicon_ltr)
        tree_labeler.train(
            train_sentences,
            train_tags,
            train_supertags,
            train_mst,
            train_upos,
            epochs=TREE_LABELER_EPOCHS,
            seed=1,
        )
        tree_labeler_2 = FullTreeLabeler(label_names, lexicon=lexicon_ltr)
        tree_labeler_2.train(
            train_sentences,
            train_tags,
            train_supertags,
            train_mst,
            train_upos,
            epochs=TREE_LABELER_EPOCHS,
            seed=2,
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
            train_upos,
            [dev_sentences],
            [dev_tags],
            [dev_supertags],
            [dev_mst],
            [dev_upos],
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
            upos_list=dev_upos,
            tree_labeler=tree_labeler,
            tree_labeler_2=tree_labeler_2,
            second_order=second_order,
            punct_forms=punct_forms,
        )
        uas, las = accuracy(dev_sentences, voted_heads, voted_labels)
        print("    Chu-Liu/Edmonds ensemble (%d parsers + MST + 2nd-order, before rules): UAS %.2f  LAS %.2f" % (len(predictions), uas, las))

    outputs = make_outputs(
        train_sentences,
        dev_sentences,
        predictions,
        weights,
        dev_tags,
        dev_supertags,
        dev_upos,
        dev_mst,
        tree_labeler,
        punct_forms,
        tree_labeler_2=tree_labeler_2,
        second_order=second_order,
    )
    learn_best_settings(outputs)


def final_mode(train_sentences):
    """Train on all training data and write the submission."""
    test_sentences = read_sentences(os.path.join(DATA_FOLDER, "test.csv"))
    print("Training sentences:", len(train_sentences), " test sentences:", len(test_sentences))

    label_names = get_label_names(train_sentences)
    punct_forms = find_punct_forms(train_sentences)

    print("Learning unsupervised Bigram Class-HMM word clusters from training words...")
    word_clusters = build_word_clusters(train_sentences)
    print("    clustered %d vocabulary entries (64 fine / 16 coarse)" % len(word_clusters))

    (
        train_tags,
        other_tags,
        train_supertags,
        other_supertags,
        train_upos,
        other_upos,
    ) = make_tags(train_sentences, [test_sentences], word_clusters=word_clusters)
    test_tags = other_tags[0]
    test_supertags = other_supertags[0]
    test_upos = other_upos[0]

    tag_Target_ltr = make_tag_targets(train_sentences)
    lexicon_ltr = build_lexicon(tag_Target_ltr, word_clusters=word_clusters)
    flipped_tag_Target = [flip_sentence(s) for s in tag_Target_ltr]
    for k in range(len(flipped_tag_Target)):
        flipped_tag_Target[k]["labels"] = flip_tags(tag_Target_ltr[k]["labels"])
    lexicon_rtl = build_lexicon(flipped_tag_Target, word_clusters=word_clusters)

    train_mst, other_mst = make_mst_guides(
        train_sentences,
        train_tags,
        train_supertags,
        train_upos,
        [test_sentences],
        [test_tags],
        [test_supertags],
        [test_upos],
        lexicon_ltr,
        word_clusters=word_clusters,
    )
    test_mst = other_mst[0]

    second_order = None
    tree_labeler = None
    tree_labeler_2 = None
    if USE_SECOND_PASS_TAGGER:
        print("Training the 2nd-order Grandparent/Sibling/Valency graph tree model...")
        second_order = SecondOrderTreeModel(lexicon=lexicon_ltr)
        second_order.train(
            train_sentences,
            train_tags,
            train_supertags,
            train_mst,
            train_upos,
            epochs=SECOND_ORDER_EPOCHS,
            seed=1,
        )

        print("Training the 2-seed full-tree 2nd/3rd-order labeler...")
        tree_labeler = FullTreeLabeler(label_names, lexicon=lexicon_ltr)
        tree_labeler.train(
            train_sentences,
            train_tags,
            train_supertags,
            train_mst,
            train_upos,
            epochs=TREE_LABELER_EPOCHS,
            seed=1,
        )
        tree_labeler_2 = FullTreeLabeler(label_names, lexicon=lexicon_ltr)
        tree_labeler_2.train(
            train_sentences,
            train_tags,
            train_supertags,
            train_mst,
            train_upos,
            epochs=TREE_LABELER_EPOCHS,
            seed=2,
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
            train_upos,
            [test_sentences],
            [test_tags],
            [test_supertags],
            [test_mst],
            [test_upos],
            lexicon_ltr=lexicon_ltr,
            lexicon_rtl=lexicon_rtl,
        )
        test_predictions.append(results[0])
        weights.append(weight)
        gc.collect()

    outputs = make_outputs(
        train_sentences,
        test_sentences,
        test_predictions,
        weights,
        test_tags,
        test_supertags,
        test_upos,
        test_mst,
        tree_labeler,
        punct_forms,
        tree_labeler_2=tree_labeler_2,
        second_order=second_order,
    )
    test_heads, test_labels = final_trees(outputs)
    write_final_submission(test_sentences, test_heads, test_labels, train_sentences)


def main():
    global PARSERS, USE_SECOND_PASS_TAGGER
    # Must come first: restarts the script with repeatable hashing if needed.
    make_hashing_repeatable()

    if "--one-parser" in sys.argv:
        PARSERS = PARSERS[:1]
    if "--no-second-pass" in sys.argv:
        USE_SECOND_PASS_TAGGER = False
    start_time = time.time()
    train_sentences = read_sentences(os.path.join(DATA_FOLDER, "train.csv"))
    if "--check" in sys.argv:
        check_mode(train_sentences)
    else:
        final_mode(train_sentences)
    print("Total time: %.1f minutes" % ((time.time() - start_time) / 60.0))


if __name__ == "__main__":
    main()
