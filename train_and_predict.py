"""
Train the dependency parsing pipeline and generate outputs/submission.csv.

Usage:
    python train_and_predict.py --check   # 90/10 validation split + hyperparameter tuning
    python train_and_predict.py           # train on 100% of train.csv and predict test.csv
"""

import gc
import json
import math
import os
import random
import sys
import time

import numpy as np

import combiner
from data_utils import (
    accuracy,
    build_lexicon,
    find_possessive_markers,
    find_punct_forms,
    make_hashing_repeatable,
    normalize_word,
    read_sample_ids,
    read_sentences,
    write_submission,
)
from parser import (
    ArcEagerParser,
    ArcStandardParser,
    FullTreeLabeler,
    GraphMSTParser,
    Parser,
    chu_liu_edmonds,
    flip_list,
    flip_mst_info,
    flip_result,
    flip_sentence,
    flip_supertags,
    flip_tags,
    jackknife_mst,
)
from tagger import (
    jackknife_tags,
    make_supertag_targets,
    make_tag_targets,
    make_upos_targets,
    tag_accuracy,
    train_tagger,
)


# Config
DATA_FOLDER = "data"
OUTPUT_FOLDER = "outputs"
BEST_SETTINGS = "best_settings.json"

UPOS_EPOCHS = 6
TAGGER_EPOCHS = 6
SUPERTAGGER_EPOCHS = 5
TAGGER_SEEDS = 3
MST_EPOCHS = 8
MST_SEEDS = 4
TREE_LABELER_EPOCHS = 6
TREE_LABELER_SEEDS = 4
JACKKNIFE_FOLDS = 5
PARSER_EPOCHS = 12

# Validation split: hold out every 10th block of 100 neighbouring sentences (10%)
CHECK_GROUP_SIZE = 100
COMBINER_MIN_GAIN = 0.10
L2_GRID = [1.0, 5.0, 15.0]

# Balanced 18-parser ensemble across 3 transition systems and both reading directions:
# (system_type, direction, seed, vote_weight)
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
    ("arc-eager",    "right-to-left", 2, 0.95),
    ("arc-standard", "right-to-left", 2, 0.95),
    ("arc-hybrid",   "left-to-right", 3, 1.30),
    ("arc-eager",    "left-to-right", 3, 1.20),
    ("arc-standard", "left-to-right", 3, 1.15),
    ("arc-hybrid",   "left-to-right", 4, 1.25),
    ("arc-eager",    "left-to-right", 4, 1.15),
    ("arc-standard", "left-to-right", 4, 1.10),
]

DEFAULT_RULES = {
    "punct": True,
    "flat": True,
    "passive": True,
    "obl_nmod": True,
}

RULE_VARIANTS = [
    ("no rules",         {"punct": False, "flat": False, "passive": False, "obl_nmod": False}),
    ("punct only",       {"punct": True,  "flat": False, "passive": False, "obl_nmod": False}),
    ("punct + flat",     {"punct": True,  "flat": True,  "passive": False, "obl_nmod": False}),
    ("punct + passive",  {"punct": True,  "flat": False, "passive": True,  "obl_nmod": False}),
    ("punct + obl_nmod", {"punct": True,  "flat": False, "passive": False, "obl_nmod": True}),
    ("all rules",        {"punct": True,  "flat": True,  "passive": True,  "obl_nmod": True}),
]


def get_label_names(sentences):
    names = set()
    for sentence in sentences:
        for i in range(1, len(sentence["labels"])):
            names.add(sentence["labels"][i])
    return sorted(names)


def make_tags(train_sentences, eval_sentences):
    """Train UPOS, bidirectional role taggers, and supertagger (5-fold jackknife on train)."""
    # 1. Induced UPOS tagger
    upos_sentences = make_upos_targets(train_sentences)
    upos_names = get_label_names(upos_sentences)
    print("[1/6] Training UPOS tagger (5-fold CV + %d-seed full)..." % TAGGER_SEEDS)
    train_upos = jackknife_tags(
        upos_sentences, upos_names, UPOS_EPOCHS, JACKKNIFE_FOLDS, direction="left-to-right"
    )
    print("  UPOS CV accuracy: %.2f%%" % tag_accuracy(upos_sentences, train_upos))
    upos_tagger = train_tagger(
        upos_sentences, upos_names, UPOS_EPOCHS, direction="left-to-right", n_seeds=TAGGER_SEEDS
    )
    eval_upos = [upos_tagger.tag(s["words"]) for s in eval_sentences]
    upos_tagger = None
    gc.collect()

    # 2. First-pass forward and backward syntactic taggers
    tag_sentences = make_tag_targets(train_sentences)
    tag_names = get_label_names(tag_sentences)

    print("[2/6] Training forward & backward syntactic taggers...")
    train_tags_fwd = jackknife_tags(
        tag_sentences, tag_names, TAGGER_EPOCHS, JACKKNIFE_FOLDS,
        direction="left-to-right", upos_tags_list=train_upos,
    )
    print("  L-to-R tag CV accuracy: %.2f%%" % tag_accuracy(tag_sentences, train_tags_fwd))
    tagger_fwd = train_tagger(
        tag_sentences, tag_names, TAGGER_EPOCHS,
        direction="left-to-right", upos_tags_list=train_upos, n_seeds=TAGGER_SEEDS,
    )
    eval_tags_fwd = [
        tagger_fwd.tag(eval_sentences[k]["words"], upos_tags=eval_upos[k])
        for k in range(len(eval_sentences))
    ]
    tagger_fwd = None
    gc.collect()

    train_tags_bwd = jackknife_tags(
        tag_sentences, tag_names, TAGGER_EPOCHS, JACKKNIFE_FOLDS,
        direction="right-to-left", upos_tags_list=train_upos,
    )
    print("  R-to-L tag CV accuracy: %.2f%%" % tag_accuracy(tag_sentences, train_tags_bwd))
    tagger_bwd = train_tagger(
        tag_sentences, tag_names, TAGGER_EPOCHS,
        direction="right-to-left", upos_tags_list=train_upos, n_seeds=TAGGER_SEEDS,
    )
    eval_tags_bwd = [
        tagger_bwd.tag(eval_sentences[k]["words"], upos_tags=eval_upos[k])
        for k in range(len(eval_sentences))
    ]
    tagger_bwd = None
    gc.collect()

    # 3. Second-pass bidirectional stacked tagger
    print("[3/6] Training stacked bidirectional tagger...")
    train_tags_2 = jackknife_tags(
        tag_sentences, tag_names, TAGGER_EPOCHS, JACKKNIFE_FOLDS,
        first_tags_list=train_tags_fwd, direction="left-to-right",
        backward_tags_list=train_tags_bwd, upos_tags_list=train_upos,
    )
    print("  Stacked tag CV accuracy: %.2f%%" % tag_accuracy(tag_sentences, train_tags_2))
    tagger_2 = train_tagger(
        tag_sentences, tag_names, TAGGER_EPOCHS,
        first_tags_list=train_tags_fwd, direction="left-to-right",
        backward_tags_list=train_tags_bwd, upos_tags_list=train_upos,
        n_seeds=TAGGER_SEEDS,
    )
    eval_tags_2 = [
        tagger_2.tag(
            eval_sentences[k]["words"],
            first_tags=eval_tags_fwd[k],
            backward_tags=eval_tags_bwd[k],
            upos_tags=eval_upos[k],
        )
        for k in range(len(eval_sentences))
    ]
    tagger_2 = None
    gc.collect()

    # 4. Head-offset supertagger
    super_sentences = make_supertag_targets(train_sentences)
    super_names = get_label_names(super_sentences)
    print("[4/6] Training head-offset supertagger...")
    train_supertags = jackknife_tags(
        super_sentences, super_names, SUPERTAGGER_EPOCHS, JACKKNIFE_FOLDS,
        first_tags_list=train_tags_2, direction="left-to-right",
        backward_tags_list=train_tags_bwd, upos_tags_list=train_upos,
    )
    print("  Supertag CV accuracy: %.2f%%" % tag_accuracy(super_sentences, train_supertags))
    super_tagger = train_tagger(
        super_sentences, super_names, SUPERTAGGER_EPOCHS,
        first_tags_list=train_tags_2, direction="left-to-right",
        backward_tags_list=train_tags_bwd, upos_tags_list=train_upos,
        n_seeds=TAGGER_SEEDS,
    )
    eval_supertags = [
        super_tagger.tag(
            eval_sentences[k]["words"],
            first_tags=eval_tags_2[k],
            backward_tags=eval_tags_bwd[k],
            upos_tags=eval_upos[k],
        )
        for k in range(len(eval_sentences))
    ]
    super_tagger = None
    gc.collect()

    return train_tags_2, eval_tags_2, train_supertags, eval_supertags, train_upos, eval_upos


def make_mst_guides(
    train_sentences,
    train_tags,
    train_supertags,
    train_upos,
    eval_sentences,
    eval_tags,
    eval_supertags,
    eval_upos,
    lexicon_ltr,
):
    """Train 5-fold jackknifed Graph-MST for train and a 3-seed Graph-MST for eval."""
    print("[5/6] Training Graph-MST parser (5-fold CV + %d-seed full)..." % MST_SEEDS)
    train_mst = jackknife_mst(
        train_sentences, train_tags, train_supertags,
        upos_list=train_upos, epochs=MST_EPOCHS, n_folds=JACKKNIFE_FOLDS,
    )
    dummy_labels = [["root"] * len(s["words"]) for s in train_sentences]
    mst_uas, _ = accuracy(train_sentences, [m["heads"] for m in train_mst], dummy_labels)
    print("  Graph-MST CV UAS: %.2f%%" % mst_uas)

    mst_models = []
    for s in range(1, MST_SEEDS + 1):
        mdl = GraphMSTParser(lexicon=lexicon_ltr)
        mdl.train(train_sentences, train_tags, train_supertags, train_upos, epochs=MST_EPOCHS, seed=s)
        mst_models.append(mdl)

    eval_mst = [
        mst_models[0].predict_mst_info(
            eval_sentences[k]["words"],
            eval_tags[k],
            eval_supertags[k],
            eval_upos[k],
            extra_models=mst_models[1:],
        )
        for k in range(len(eval_sentences))
    ]
    mst_models = None
    gc.collect()
    return train_mst, eval_mst


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
    eval_sentences,
    eval_tags,
    eval_supertags,
    eval_mst,
    eval_upos,
    lexicon_ltr,
    lexicon_rtl,
):
    """Train one transition parser and return its (heads, labels, margins) on eval_sentences."""
    if direction == "right-to-left":
        tr_sents = [flip_sentence(s) for s in train_sentences]
        tr_tags = [flip_tags(t) for t in train_tags]
        tr_super = [flip_supertags(st) for st in train_supertags]
        tr_mst = [flip_mst_info(m) for m in train_mst]
        tr_upos = [flip_list(u) for u in train_upos]
        lexicon = lexicon_rtl
    else:
        tr_sents = train_sentences
        tr_tags = train_tags
        tr_super = train_supertags
        tr_mst = train_mst
        tr_upos = train_upos
        lexicon = lexicon_ltr

    if system_type == "arc-eager":
        parser = ArcEagerParser(label_names, lexicon=lexicon)
    elif system_type == "arc-standard":
        parser = ArcStandardParser(label_names, lexicon=lexicon)
    else:
        parser = Parser(label_names, lexicon=lexicon)

    rng = random.Random(seed)
    for epoch in range(PARSER_EPOCHS):
        parser.train_one_epoch(tr_sents, tr_tags, rng, tr_super, tr_mst, tr_upos, epoch=epoch)
    parser.model.average_weights()

    all_heads = []
    all_labels = []
    all_margins = []
    for k in range(len(eval_sentences)):
        if direction == "right-to-left":
            flipped = flip_sentence(eval_sentences[k])
            heads, labels, margins = parser.parse(
                flipped["words"],
                flip_tags(eval_tags[k]),
                False,
                flip_supertags(eval_supertags[k]),
                flip_mst_info(eval_mst[k]),
                flip_list(eval_upos[k]),
            )
            heads, labels, margins = flip_result(heads, labels, margins)
        else:
            heads, labels, margins = parser.parse(
                eval_sentences[k]["words"],
                eval_tags[k],
                False,
                eval_supertags[k],
                eval_mst[k],
                eval_upos[k],
            )
        all_heads.append(heads)
        all_labels.append(labels)
        all_margins.append(margins)
    return (all_heads, all_labels, all_margins)


def _refine_ud_tree(words, heads, labels, upos=None, punct_forms=None, rules=None, possessive_markers=None):
    """Apply simple UD consistency fixes to one decoded tree."""
    if rules is None:
        rules = DEFAULT_RULES
    n = len(words) - 1
    if n <= 0:
        return heads, labels

    # 1. Punctuation tokens always get the "punct" label
    if rules["punct"] and punct_forms is not None:
        for i in range(1, n + 1):
            if heads[i] != 0 and words[i] in punct_forms:
                labels[i] = "punct"

    # 2. Keep multi-word proper names ("flat") connected
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

    # 3. Passive predicates (aux:pass) take nsubj:pass instead of nsubj
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

    # 4. Prepositional & possessive phrases: obl <-> nmod / nmod:poss
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

        has_poss_case = possessive_markers is not None and any(
            normalize_word(words[c]) in possessive_markers for c in case_children
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


def apply_rules(sentences, all_heads, all_labels, upos_list, punct_forms, rules, possessive_markers):
    """Apply UD consistency rules to copies of all trees."""
    new_heads = []
    new_labels = []
    for k in range(len(sentences)):
        heads, labels = _refine_ud_tree(
            sentences[k]["words"],
            list(all_heads[k]),
            list(all_labels[k]),
            upos_list[k] if upos_list is not None else None,
            punct_forms,
            rules,
            possessive_markers,
        )
        new_heads.append(heads)
        new_labels.append(labels)
    return new_heads, new_labels


def vote(outputs):
    """Combine all parsers + 3-seed MST via Chu-Liu/Edmonds and 3-seed full-tree labeler."""
    predictions = outputs["predictions"]
    sentences = outputs["sentences"]
    weights = outputs["weights"]
    tags_list = outputs["tags"]
    supertags_list = outputs["supertags"]
    mst_list = outputs["mst"]
    upos_list = outputs["upos"]
    tree_labeler = outputs["tree_labeler"]
    extra_labelers = outputs.get("extra_labelers")
    punct_forms = outputs["punct_forms"]

    voted_heads = []
    voted_labels = []

    for k in range(len(sentences)):
        words = sentences[k]["words"]
        n = len(words) - 1
        if n <= 0:
            voted_heads.append([-1])
            voted_labels.append([""])
            continue

        punct_flags = [False] + [punct_forms is not None and words[i] in punct_forms for i in range(1, n + 1)]
        all_punct = all(punct_flags[1:])

        # Build the global arc score matrix
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

        for p_idx, (all_heads, _, all_margins) in enumerate(predictions):
            w_p = weights[p_idx]
            for i in range(1, n + 1):
                h = all_heads[k][i]
                if 0 <= h <= n and h != i:
                    conf_mult = 0.85 + 0.30 * (min(max(all_margins[k][i], 0.0), 12.0) / 12.0)
                    arc_scores[h, i] += w_p * conf_mult - 1.0e-5 * p_idx

        if mst_list is not None and mst_list[k] is not None:
            minfo = mst_list[k]
            mh1, mh2, mh3, mconf = minfo["heads"], minfo["heads2"], minfo.get("heads3"), minfo["conf"]
            mmarg = minfo.get("margins")
            mst_scale = max(len(predictions) / 12.0, 1.0)
            for i in range(1, n + 1):
                h1 = mh1[i]
                h2 = mh2[i]
                h3 = mh3[i] if mh3 is not None else -1
                if 0 <= h1 <= n and h1 != i:
                    base_b = 1.05 if mconf[i] == "HIGH" else (0.72 if mconf[i] == "MID" else 0.45)
                    if mmarg is not None:
                        cont_b = 0.45 + 0.65 * (min(max(float(mmarg[i]), 0.0), 8.0) / 8.0)
                        base_b = 0.5 * (base_b + cont_b)
                    arc_scores[h1, i] += mst_scale * base_b
                if 0 <= h2 <= n and h2 != i:
                    arc_scores[h2, i] += mst_scale * 0.22
                if 0 <= h3 <= n and h3 != i:
                    arc_scores[h3, i] += mst_scale * 0.08

        heads = chu_liu_edmonds(arc_scores)

        utags = upos_list[k] if upos_list is not None else None
        stags = supertags_list[k] if supertags_list is not None else None
        tree_scores = None
        if tree_labeler is not None:
            tree_scores = tree_labeler.score_sentence_labels(
                words, tags_list[k], stags, heads, upos=utags, extra_models=extra_labelers,
            )

        labels = [""] * (n + 1)
        for i in range(1, n + 1):
            h = heads[i]
            if h == 0:
                labels[i] = "root"
                continue

            label_votes = {}
            first_seen = {}
            for p_idx, (all_heads, all_labels, all_margins) in enumerate(predictions):
                lab = all_labels[k][i]
                if lab == "root":
                    continue
                w_p = weights[p_idx]
                conf_mult = 0.85 + 0.30 * (min(max(all_margins[k][i], 0.0), 12.0) / 12.0)
                vote_w = w_p * conf_mult if all_heads[k][i] == h else 0.22 * w_p
                label_votes[lab] = label_votes.get(lab, 0.0) + vote_w
                if lab not in first_seen:
                    first_seen[lab] = p_idx

            if tree_scores is not None and tree_scores[i] is not None:
                t_map = tree_scores[i]
                if t_map:
                    lbl_scale = max(len(predictions) / 12.0, 1.0)
                    max_ts = max(t_map.values())
                    for lab, ts in t_map.items():
                        diff = ts - max_ts
                        if diff >= -8.0:
                            label_votes[lab] = label_votes.get(lab, 0.0) + (1.25 * lbl_scale) * math.exp(0.25 * diff)
                            if lab not in first_seen:
                                first_seen[lab] = len(predictions)

            if label_votes:
                labels[i] = max(label_votes.keys(), key=lambda l: (label_votes[l], -first_seen.get(l, 999)))
            else:
                labels[i] = "dep"

        voted_heads.append(heads)
        voted_labels.append(labels)

    return voted_heads, voted_labels


def finish_trees(outputs, heads, labels, rules):
    return apply_rules(
        outputs["sentences"], heads, labels, outputs["upos"],
        outputs["punct_forms"], rules, outputs["possessive_markers"],
    )


def print_error_report(sentences, heads, labels):
    """Print a short summary of remaining head and label errors on validation."""
    total = wrong_head = wrong_label = 0
    total_of = {}
    wrong_of = {}
    mix_ups = {}
    for k in range(len(sentences)):
        for i in range(1, len(sentences[k]["words"])):
            gh = sentences[k]["heads"][i]
            gl = sentences[k]["labels"][i]
            total += 1
            total_of[gl] = total_of.get(gl, 0) + 1
            if heads[k][i] != gh:
                wrong_head += 1
                wrong_of[gl] = wrong_of.get(gl, 0) + 1
            elif labels[k][i] != gl:
                wrong_label += 1
                wrong_of[gl] = wrong_of.get(gl, 0) + 1
                pair = gl + " -> " + labels[k][i]
                mix_ups[pair] = mix_ups.get(pair, 0) + 1

    print("\nError breakdown (%d tokens):" % total)
    print("  Wrong head:             %5d (%.2f%%)" % (wrong_head, 100.0 * wrong_head / total))
    print("  Right head, wrong label:%5d (%.2f%%)" % (wrong_label, 100.0 * wrong_label / total))
    print("  Top error labels:")
    for (cnt, lab) in sorted(((c, l) for l, c in wrong_of.items()), reverse=True)[:12]:
        print("    %-15s %5d / %5d  (-%.2f%%)" % (lab, cnt, total_of[lab], 100.0 * cnt / total))
    print("  Top label confusions (gold -> pred):")
    for (cnt, pair) in sorted(((c, p) for p, c in mix_ups.items()), reverse=True)[:10]:
        print("    %-30s %5d" % (pair, cnt))


def run_pipeline(train_sentences, eval_sentences, is_val=False):
    """Shared training pipeline used for both 90/10 validation and 100% final training."""
    label_names = get_label_names(train_sentences)
    punct_forms = find_punct_forms(train_sentences)

    train_tags, eval_tags, train_supertags, eval_supertags, train_upos, eval_upos = make_tags(
        train_sentences, eval_sentences
    )
    if is_val:
        print("  Val UPOS: %.2f%% | Tag: %.2f%% | Supertag: %.2f%%" % (
            tag_accuracy(make_upos_targets(eval_sentences), eval_upos),
            tag_accuracy(make_tag_targets(eval_sentences), eval_tags),
            tag_accuracy(make_supertag_targets(eval_sentences), eval_supertags),
        ))

    tag_targets_ltr = make_tag_targets(train_sentences)
    lexicon_ltr = build_lexicon(tag_targets_ltr)
    flipped_targets = [flip_sentence(s) for s in tag_targets_ltr]
    for k in range(len(flipped_targets)):
        flipped_targets[k]["labels"] = flip_tags(tag_targets_ltr[k]["labels"])
    lexicon_rtl = build_lexicon(flipped_targets)

    train_mst, eval_mst = make_mst_guides(
        train_sentences, train_tags, train_supertags, train_upos,
        eval_sentences, eval_tags, eval_supertags, eval_upos,
        lexicon_ltr,
    )
    if is_val:
        dummy = [["root"] * len(s["words"]) for s in eval_sentences]
        mst_uas, _ = accuracy(eval_sentences, [m["heads"] for m in eval_mst], dummy)
        print("  Val %d-seed Graph-MST UAS: %.2f%%" % (MST_SEEDS, mst_uas))

    print("[6/6] Training %d-seed full-tree labeler and %d transition parsers..."
          % (TREE_LABELER_SEEDS, len(PARSERS)))
    labeler_models = []
    for s in range(1, TREE_LABELER_SEEDS + 1):
        lbl = FullTreeLabeler(label_names, lexicon=lexicon_ltr)
        lbl.train(
            train_sentences, train_tags, train_supertags, train_mst, train_upos,
            epochs=TREE_LABELER_EPOCHS, seed=s,
        )
        labeler_models.append(lbl)

    predictions = []
    weights = []
    for idx, (system_type, direction, seed, weight) in enumerate(PARSERS, 1):
        res = train_one_parser(
            system_type, direction, seed, label_names,
            train_sentences, train_tags, train_supertags, train_mst, train_upos,
            eval_sentences, eval_tags, eval_supertags, eval_mst, eval_upos,
            lexicon_ltr, lexicon_rtl,
        )
        predictions.append(res)
        weights.append(weight)
        gc.collect()
        if is_val:
            uas, las = accuracy(eval_sentences, res[0], res[1])
            print("  [%2d/%d] %-12s %-13s seed=%d -> UAS %.2f  LAS %.2f"
                  % (idx, len(PARSERS), system_type, direction, seed, uas, las))
        else:
            print("  [%2d/%d] %-12s %-13s seed=%d done"
                  % (idx, len(PARSERS), system_type, direction, seed))

    return {
        "sentences": eval_sentences,
        "predictions": predictions,
        "weights": weights,
        "parsers": list(PARSERS),
        "tags": eval_tags,
        "supertags": eval_supertags,
        "upos": eval_upos,
        "mst": eval_mst,
        "tree_labeler": labeler_models[0],
        "extra_labelers": labeler_models[1:],
        "second_order": None,
        "punct_forms": punct_forms,
        "possessive_markers": find_possessive_markers(train_sentences),
    }


def tune_on_validation(outputs):
    """Tune UD rules and combiner L2 on the 10% validation set and save best_settings.json."""
    sentences = outputs["sentences"]
    voted_heads, voted_labels = vote(outputs)
    base_uas, base_las = accuracy(sentences, voted_heads, voted_labels)
    print("\nEnsemble vote (before rules): UAS %.2f  LAS %.2f" % (base_uas, base_las))

    # 1. Pick best UD rule combination
    print("\nTesting post-processing rules on validation:")
    best_rules = DEFAULT_RULES
    best_rules_las = -1.0
    best_rule_name = ""
    for (name, rules) in RULE_VARIANTS:
        fh, fl = finish_trees(outputs, voted_heads, voted_labels, rules)
        uas, las = accuracy(sentences, fh, fl)
        print("  %-18s -> UAS %.2f  LAS %.2f" % (name, uas, las))
        if las > best_rules_las:
            best_rules_las = las
            best_rules = rules
            best_rule_name = name
    print("  Best rules: %s" % best_rule_name)

    hand_heads, hand_labels = finish_trees(outputs, voted_heads, voted_labels, best_rules)
    hand_uas, hand_las = accuracy(sentences, hand_heads, hand_labels)

    # 2. Tune softmax combiner L2 via 2-fold CV on the validation split
    print("\nTuning softmax combiner L2 (2-fold CV on validation):")
    best_l2 = 5.0
    best_comb_las = -1.0
    best_comb_trees = None
    for l2 in L2_GRID:
        ch, cl = combiner.cross_validate(outputs, l2=l2)
        ch, cl = finish_trees(outputs, ch, cl, best_rules)
        uas, las = accuracy(sentences, ch, cl)
        print("  l2=%-4.1f -> UAS %.2f  LAS %.2f" % (l2, uas, las))
        if las > best_comb_las:
            best_comb_las = las
            best_l2 = l2
            best_comb_trees = (ch, cl)

    use_combiner = best_comb_las >= hand_las + COMBINER_MIN_GAIN
    print("\nValidation summary:")
    print("  Normal vote:               UAS %.2f  LAS %.2f" % (hand_uas, hand_las))
    print("  Softmax combiner (l2=%.1f): LAS %.2f" % (best_l2, best_comb_las))
    if use_combiner:
        print("  -> Selected: softmax combiner (+%.2f LAS)" % (best_comb_las - hand_las))
        print_error_report(sentences, best_comb_trees[0], best_comb_trees[1])
    else:
        print("  -> Selected: normal vote")
        print_error_report(sentences, hand_heads, hand_labels)

    model = combiner.train_combiner(outputs, list(range(len(sentences))), l2=best_l2)
    settings = {
        "rules": best_rules,
        "use_combiner": bool(use_combiner),
        "combiner_l2": best_l2,
        "normal_vote_las": round(hand_las, 2),
        "combiner_las": round(best_comb_las, 2),
        "combiner": {
            "arc_weights": [float(x) for x in model["arc_weights"]],
            "label_weights": [float(x) for x in model["label_weights"]],
            "parsers": [list(p) for p in model["parsers"]],
            "uses_mst": bool(model["uses_mst"]),
            "uses_labeler": bool(model["uses_labeler"]),
        },
    }
    with open(BEST_SETTINGS, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=1)
    print("\nSaved tuned settings to %s" % BEST_SETTINGS)


def decode_final_trees(outputs):
    """Decode test trees using best_settings.json if available, otherwise normal vote."""
    rules = DEFAULT_RULES
    model = None

    if os.path.exists(BEST_SETTINGS):
        with open(BEST_SETTINGS, "r", encoding="utf-8") as f:
            settings = json.load(f)
        rules = settings.get("rules", DEFAULT_RULES)
        if settings.get("use_combiner"):
            saved = settings["combiner"]
            same_parsers = saved["parsers"] == [list(p) for p in outputs["parsers"]]
            same_mst = saved["uses_mst"] == (outputs["mst"] is not None)
            same_labeler = saved["uses_labeler"] == (outputs["tree_labeler"] is not None)
            if same_parsers and same_mst and same_labeler:
                model = {
                    "arc_weights": np.array(saved["arc_weights"]),
                    "label_weights": np.array(saved["label_weights"]),
                }
        print("Loaded tuned settings from %s" % BEST_SETTINGS)
    else:
        print("No %s found; using default vote and rules" % BEST_SETTINGS)

    if model is not None:
        print("Decoding test trees with softmax combiner...")
        heads_of, labels_of = combiner.combine(outputs, model)
        heads = [heads_of[k] for k in range(len(outputs["sentences"]))]
        labels = [labels_of[k] for k in range(len(outputs["sentences"]))]
    else:
        print("Decoding test trees with normal vote...")
        heads, labels = vote(outputs)

    return finish_trees(outputs, heads, labels, rules)


def write_final_submission(test_sentences, test_heads, test_labels, train_sentences):
    """Write outputs/submission.csv matching data/sample.csv IDs."""
    answer_of_id = {}
    for k in range(len(test_sentences)):
        for i in range(1, len(test_sentences[k]["words"])):
            row_id = test_sentences[k]["row_ids"][i - 1]
            answer_of_id[row_id] = "%d|%s" % (test_heads[k][i], test_labels[k][i])

    # Include training IDs only if sample.csv lists them (Kaggle evaluates test rows)
    for k in range(len(train_sentences)):
        for i in range(1, len(train_sentences[k]["words"])):
            row_id = train_sentences[k]["row_ids"][i - 1]
            answer_of_id[row_id] = "%d|%s" % (train_sentences[k]["heads"][i], train_sentences[k]["labels"][i])

    sample_ids, header = read_sample_ids(os.path.join(DATA_FOLDER, "sample.csv"))
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    output_path = os.path.join(OUTPUT_FOLDER, "submission.csv")
    write_submission(output_path, header, sample_ids, answer_of_id)
    print("Saved %d predictions to %s" % (len(sample_ids), output_path))


def check_mode(all_train):
    """90% train / 10% document-grouped validation split for tuning."""
    train_sentences = []
    val_sentences = []
    for k, sentence in enumerate(all_train):
        if (k // CHECK_GROUP_SIZE) % 10 == 0:
            val_sentences.append(sentence)
        else:
            train_sentences.append(sentence)
    print("Validation mode: %d train / %d val sentences" % (len(train_sentences), len(val_sentences)))
    outputs = run_pipeline(train_sentences, val_sentences, is_val=True)
    tune_on_validation(outputs)


def final_mode(train_sentences):
    """Train on 100% of train.csv and write test predictions."""
    test_sentences = read_sentences(os.path.join(DATA_FOLDER, "test.csv"))
    print("Final mode: %d train / %d test sentences" % (len(train_sentences), len(test_sentences)))
    outputs = run_pipeline(train_sentences, test_sentences, is_val=False)
    test_heads, test_labels = decode_final_trees(outputs)
    write_final_submission(test_sentences, test_heads, test_labels, train_sentences)


def main():
    make_hashing_repeatable()
    start_time = time.time()
    train_sentences = read_sentences(os.path.join(DATA_FOLDER, "train.csv"))
    if "--check" in sys.argv:
        check_mode(train_sentences)
    else:
        final_mode(train_sentences)
    print("Finished in %.1f minutes" % ((time.time() - start_time) / 60.0))


if __name__ == "__main__":
    main()
