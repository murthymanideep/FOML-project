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
    --no-copy          do not copy answers of sentences seen in training

At the end, both modes save the outputs of all trained parsers in
outputs/check_cache.pkl or outputs/final_cache.pkl. Then revote.py can redo
only the last step (vote + rule-based fixes) in a few minutes, without
training anything again.
"""

import gc
import math
import os
import pickle
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
    make_hashing_repeatable,
    build_lexicon,
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


# ----------------------------- settings -----------------------------

DATA_FOLDER = "data"
OUTPUT_FOLDER = "outputs"

UPOS_EPOCHS = 6
TAGGER_EPOCHS = 6
SUPERTAGGER_EPOCHS = 5
MST_EPOCHS = 8
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
COPY_KNOWN_SENTENCES = True

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

CHECK_CACHE = os.path.join(OUTPUT_FOLDER, "check_cache.pkl")
FINAL_CACHE = os.path.join(OUTPUT_FOLDER, "final_cache.pkl")

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
      0. Induced Universal POS (`UPOS`) tagger (19 syntactic classes induced from gold trees)
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
        upos_sentences, upos_names, UPOS_EPOCHS, JACKKNIFE_FOLDS, direction="left-to-right"
    )
    print("    Induced UPOS accuracy on training data (jackknifed): %.2f" % tag_accuracy(upos_sentences, train_upos))

    print("Training the Induced UPOS tagger on all training sentences...")
    upos_tagger = train_tagger(upos_sentences, upos_names, UPOS_EPOCHS, direction="left-to-right")
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
    )
    print("    L-to-R tag accuracy on training data (jackknifed): %.2f" % tag_accuracy(tag_sentences, train_tags_fwd))

    print("Training the first-pass L-to-R tagger on all training sentences...")
    tagger_fwd = train_tagger(
        tag_sentences,
        tag_names,
        TAGGER_EPOCHS,
        direction="left-to-right",
        upos_tags_list=train_upos,
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
    )
    print("    R-to-L tag accuracy on training data (jackknifed): %.2f" % tag_accuracy(tag_sentences, train_tags_bwd))

    print("Training the first-pass R-to-L tagger on all training sentences...")
    tagger_bwd = train_tagger(
        tag_sentences,
        tag_names,
        TAGGER_EPOCHS,
        direction="right-to-left",
        upos_tags_list=train_upos,
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


def _refine_ud_tree(words, heads, labels, upos=None, punct_forms=None, rules=None):
    """
    Apply simple Universal Dependencies rules to one voted tree.

    rules       : which rules are on (see RULES at the top of this file)
    punct_forms : word forms that are almost always "punct" in the training
                  data (from find_punct_forms)
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

        has_poss_case = any(
            normalize_word(words[c]) in ("'s", "'") for c in case_children
        )
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


def apply_rules(sentences, all_heads, all_labels, upos_list, punct_forms, rules=None):
    """Return copies of all trees with the rule-based fixes applied."""
    new_heads = []
    new_labels = []
    for k in range(len(sentences)):
        heads = list(all_heads[k])
        labels = list(all_labels[k])
        upos = None
        if upos_list is not None:
            upos = upos_list[k]
        heads, labels = _refine_ud_tree(sentences[k]["words"], heads, labels, upos, punct_forms, rules)
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
    punct_forms=None,
):
    """
    Combine all parsers via:
      1. Margin-confidence-weighted arc score matrix construction (combining all
         12 transition parsers across 3 transition systems + 2-seed Global Graph-MST).
      2. Single-root Chu-Liu/Edmonds Maximum Spanning Arborescence decoding.
      3. Full-tree 2nd/3rd-order label scoring (`FullTreeLabeler`) blended with
         the transition ensemble's confidence-weighted label votes.
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
                if 0 <= h2 <= n and h2 != i:
                    arc_scores[h2, i] += 0.22
                if 0 <= h3 <= n and h3 != i:
                    arc_scores[h3, i] += 0.08

        # Decode exact single-root Maximum Spanning Arborescence via Chu-Liu/Edmonds
        heads = chu_liu_edmonds(arc_scores)

        # Compute full-tree label scores when `tree_labeler` is available
        tree_scores = None
        utags = upos_list[k] if upos_list is not None else None
        if tree_labeler is not None and tags_list is not None:
            stags = supertags_list[k] if supertags_list is not None else None
            tree_scores = tree_labeler.score_sentence_labels(
                words, tags_list[k], stags, heads, upos=utags
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


def print_rule_table(train_sentences, sentences, voted_heads, voted_labels, upos_list, punct_forms):
    """Print the check score with each set of rule-based fixes (cheap: no new vote)."""
    print("Rule-based fixes on the voted trees (choose the best in RULES at the top of the file):")
    for (name, rules) in RULE_VARIANTS:
        heads, labels = apply_rules(sentences, voted_heads, voted_labels, upos_list, punct_forms, rules)
        if COPY_KNOWN_SENTENCES:
            copy_known_sentences(train_sentences, sentences, heads, labels)
        uas, las = accuracy(sentences, heads, labels)
        print("    %-18s UAS %.2f  LAS %.2f" % (name, uas, las))


def save_cache(path, cache):
    """Save the outputs of all trained models, for revote.py."""
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    with open(path, "wb") as cache_file:
        pickle.dump(cache, cache_file, protocol=pickle.HIGHEST_PROTOCOL)
    print("Saved the parser outputs in", path, "(revote.py can use them)")


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
    (
        train_tags,
        other_tags,
        train_supertags,
        other_supertags,
        train_upos,
        other_upos,
    ) = make_tags(train_sentences, [dev_sentences])
    dev_tags = other_tags[0]
    dev_supertags = other_supertags[0]
    dev_upos = other_upos[0]
    print("    UPOS accuracy on check data: %.2f" % tag_accuracy(make_upos_targets(dev_sentences), dev_upos))
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
        train_upos,
        [dev_sentences],
        [dev_tags],
        [dev_supertags],
        [dev_upos],
        lexicon_ltr,
    )
    dev_mst = other_mst[0]
    if USE_SECOND_PASS_TAGGER:
        dev_mst_heads = [m["heads"] for m in dev_mst]
        dummy_labels = [["root"] * len(s["words"]) for s in dev_sentences]
        mst_dev_uas, _ = accuracy(dev_sentences, dev_mst_heads, dummy_labels)
        print("    2-seed global Graph-MST UAS on check data: %.2f" % mst_dev_uas)

    tree_labeler = None
    if USE_SECOND_PASS_TAGGER:
        print("Training the full-tree 2nd/3rd-order labeler...")
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
            punct_forms=punct_forms,
        )
        uas, las = accuracy(dev_sentences, voted_heads, voted_labels)
        print("    Chu-Liu/Edmonds ensemble (%d parsers + MST, before rules): UAS %.2f  LAS %.2f" % (len(predictions), uas, las))

    print_rule_table(train_sentences, dev_sentences, voted_heads, voted_labels, dev_upos, punct_forms)

    final_heads, final_labels = apply_rules(dev_sentences, voted_heads, voted_labels, dev_upos, punct_forms, RULES)
    if COPY_KNOWN_SENTENCES:
        copied = copy_known_sentences(train_sentences, dev_sentences, final_heads, final_labels)
        print("Copied the training answer for %d check sentences" % copied)
    uas, las = accuracy(dev_sentences, final_heads, final_labels)
    print("FINAL check score with the current settings: UAS %.2f  LAS %.2f   <- LAS is the competition score" % (uas, las))

    cache = {
        "mode": "check",
        "train_sentences": train_sentences,
        "sentences": dev_sentences,
        "predictions": predictions,
        "weights": weights,
        "parsers": list(PARSERS),
        "tags": dev_tags,
        "supertags": dev_supertags,
        "upos": dev_upos,
        "mst": dev_mst,
        "tree_labeler": tree_labeler,
        "punct_forms": punct_forms,
    }
    save_cache(CHECK_CACHE, cache)


def final_mode(train_sentences):
    """Train on all training data and write the submission."""
    test_sentences = read_sentences(os.path.join(DATA_FOLDER, "test.csv"))
    print("Training sentences:", len(train_sentences), " test sentences:", len(test_sentences))

    label_names = get_label_names(train_sentences)
    punct_forms = find_punct_forms(train_sentences)
    (
        train_tags,
        other_tags,
        train_supertags,
        other_supertags,
        train_upos,
        other_upos,
    ) = make_tags(train_sentences, [test_sentences])
    test_tags = other_tags[0]
    test_supertags = other_supertags[0]
    test_upos = other_upos[0]

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
        train_upos,
        [test_sentences],
        [test_tags],
        [test_supertags],
        [test_upos],
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
            train_upos,
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

    voted_heads, voted_labels = vote(
        test_predictions,
        test_sentences,
        weights,
        tags_list=test_tags,
        supertags_list=test_supertags,
        mst_list=test_mst,
        upos_list=test_upos,
        tree_labeler=tree_labeler,
        punct_forms=punct_forms,
    )
    test_heads, test_labels = apply_rules(test_sentences, voted_heads, voted_labels, test_upos, punct_forms, RULES)
    if COPY_KNOWN_SENTENCES:
        copied = copy_known_sentences(train_sentences, test_sentences, test_heads, test_labels)
        print("Copied the training answer for", copied, "test sentences")

    write_final_submission(test_sentences, test_heads, test_labels, train_sentences)

    cache = {
        "mode": "final",
        "train_sentences": train_sentences,
        "sentences": test_sentences,
        "predictions": test_predictions,
        "weights": weights,
        "parsers": list(PARSERS),
        "tags": test_tags,
        "supertags": test_supertags,
        "upos": test_upos,
        "mst": test_mst,
        "tree_labeler": tree_labeler,
        "punct_forms": punct_forms,
    }
    save_cache(FINAL_CACHE, cache)


def main():
    global PARSERS, USE_SECOND_PASS_TAGGER, COPY_KNOWN_SENTENCES
    # Must come first: restarts the script with repeatable hashing if needed.
    make_hashing_repeatable()

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
