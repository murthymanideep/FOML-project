"""
Main script: trains the bidirectional two-pass tagger, head-offset supertagger,
and heterogeneous (Arc-Hybrid + Arc-Eager, L-to-R + R-to-L) parser ensemble,
and writes the submission file.

    python train_and_predict.py           -> train on all training data,
                                             write outputs/submission.csv
    python train_and_predict.py --check   -> train on 90% of the training data
                                             and print the score on the other 10%

Extra options (can be combined with --check):
    --one-parser       train only the first parser (fast, for experiments)
    --no-second-pass   do not use the second-pass tagger & supertagger
    --no-copy          do not copy answers of sentences seen in training
"""

import gc
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
    flip_sentence,
    flip_tags,
    flip_supertags,
    flip_result,
)


# ----------------------------- settings -----------------------------

DATA_FOLDER = "data"
OUTPUT_FOLDER = "outputs"

TAGGER_EPOCHS = 6
SUPERTAGGER_EPOCHS = 5
JACKKNIFE_FOLDS = 5
PARSER_EPOCHS = 12

# Heterogeneous parser ensemble:
# (system_type, reading_direction, random_seed, vote_weight)
# Combines bottom-up Arc-Hybrid and top-down Arc-Eager transition systems
# in both Left-to-Right and Right-to-Left reading directions.
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
      3. Pass-2 Stacked tagger conditioned on both Forward and Backward Pass-1 tags
      4. Relative head-offset supertagger conditioned on Pass-2 and Backward tags
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

    # ----- second pass: stacked bidirectional tagger -----
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
    sentence_lists,
    tags_lists,
    supertags_lists,
    lexicon_ltr=None,
    lexicon_rtl=None,
):
    """Train one parser (Arc-Hybrid or Arc-Eager) and return its (heads, labels)."""
    if direction == "right-to-left":
        sentences_for_training = []
        tags_for_training = []
        super_for_training = []
        for k in range(len(train_sentences)):
            sentences_for_training.append(flip_sentence(train_sentences[k]))
            tags_for_training.append(flip_tags(train_tags[k]))
            if train_supertags[k] is not None:
                super_for_training.append(flip_supertags(train_supertags[k]))
            else:
                super_for_training.append(None)
        lexicon = lexicon_rtl
    else:
        sentences_for_training = train_sentences
        tags_for_training = train_tags
        super_for_training = train_supertags
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
            epoch=epoch,
        )
        print("    epoch", epoch + 1, "of", PARSER_EPOCHS, "done")
    parser.model.average_weights()

    results = []
    for j in range(len(sentence_lists)):
        sentences = sentence_lists[j]
        tags_list = tags_lists[j]
        super_list = supertags_lists[j]
        all_heads = []
        all_labels = []
        for k in range(len(sentences)):
            stags = super_list[k] if super_list is not None else None
            if direction == "right-to-left":
                flipped = flip_sentence(sentences[k])
                f_tags = flip_tags(tags_list[k])
                f_stags = flip_supertags(stags) if stags is not None else None
                heads, labels = parser.parse(flipped["words"], f_tags, False, f_stags)
                heads, labels = flip_result(heads, labels)
            else:
                heads, labels = parser.parse(sentences[k]["words"], tags_list[k], False, stags)
            all_heads.append(heads)
            all_labels.append(labels)
        results.append((all_heads, all_labels))
    return results


def _best_label_for_head(head_idx, word_idx, predictions, weights, sentence_idx, fallback_label="dep"):
    """Return the highest-weighted label for attaching `word_idx` to `head_idx`."""
    if head_idx == 0:
        return "root"
    label_votes = {}
    first_seen = {}
    for m, (all_heads, all_labels) in enumerate(predictions):
        if all_heads[sentence_idx][word_idx] == head_idx:
            lab = all_labels[sentence_idx][word_idx]
            if lab == "root":
                continue
            label_votes[lab] = label_votes.get(lab, 0.0) + weights[m]
            if lab not in first_seen:
                first_seen[lab] = m
    if label_votes:
        return max(label_votes.keys(), key=lambda l: (label_votes[l], -first_seen[l]))
    any_votes = {}
    for m, (_, all_labels) in enumerate(predictions):
        lab = all_labels[sentence_idx][word_idx]
        if lab != "root":
            any_votes[lab] = any_votes.get(lab, 0.0) + weights[m]
    if any_votes:
        return max(any_votes.keys(), key=lambda l: any_votes[l])
    return fallback_label


def _refine_ud_tree(words, heads, labels):
    """
    Apply deterministic Universal Dependencies v2 structural consistency rules
    to the voted parse tree:
      1. Pure punctuation tokens (never at root) always take label `punct`.
      2. Prepositional phrases (`case` child present) modifying a pure nominal
         argument (`obj`, `iobj`, `nsubj`, `nsubj:pass`, `nmod`, `nmod:poss`,
         `compound`, `appos` without copula/auxiliary/subject children) take
         `nmod` rather than `obl`; possessive `'s` phrases preceding their head
         take `nmod:poss`.
      3. Multi-word proper names (`flat`): if token `i+1` attaches to `i-1` as
         `flat` and intervening capitalized proper name `i` was attached to
         `i+1` or `i-1` as `case`/`compound`/`dep`, attach `i` to `i-1` as `flat`.
    """
    n = len(words) - 1
    if n <= 0:
        return heads, labels

    # 1. Punctuation label consistency
    for i in range(1, n + 1):
        if heads[i] != 0 and is_punct_word(words[i]):
            labels[i] = "punct"

    # 2. Multi-word proper noun (`flat`) continuity (e.g., "Mary Jo Foley")
    for i in range(2, n):
        if heads[i + 1] == i - 1 and labels[i + 1] == "flat":
            w_i = words[i]
            if w_i and w_i[0].isupper() and w_i.replace(".", "").isalpha():
                if heads[i] in (i - 1, i + 1) and labels[i] in ("case", "compound", "dep", "nmod"):
                    heads[i] = i - 1
                    labels[i] = "flat"

    # Build child lists for structural label checks
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

        # Check possessive clitic `'s`
        has_poss_case = any(
            normalize_word(words[c]) in ("'s", "'") for c in case_children
        )
        if has_poss_case and heads[i] > i and labels[i] in ("nmod", "compound", "obl", "dep"):
            labels[i] = "nmod:poss"
            continue

        # Check PP attached as `obl` to a pure nominal head
        h = heads[i]
        if labels[i] == "obl" and 1 <= h <= n and labels[h] in nominal_head_labels:
            head_is_predicate = any(labels[ch] in clausal_child_labels for ch in children[h])
            if not head_is_predicate:
                labels[i] = "nmod"

    return heads, labels


def _enforce_valid_tree(
    words, heads, labels, head_votes_per_word, predictions, weights, sentence_idx
):
    """
    Ensure the voted dependency graph is a valid single-root, cycle-free UD tree:
      1. Pure punctuation tokens do not serve as heads of non-punctuation tokens.
      2. Exactly one word has head == 0 (with label 'root').
      3. No non-root word has label 'root'.
      4. Any directed cycles are broken by switching the weakest cycle arc to
         its best outside head.
      5. Post-hoc UD v2 structural label consistency refinement.
    """
    n = len(heads) - 1
    if n <= 0:
        return heads, labels

    punct_flags = [False] * (n + 1)
    for i in range(1, n + 1):
        punct_flags[i] = is_punct_word(words[i])

    # 1. Prevent non-punctuation words from attaching to punctuation heads
    if not all(punct_flags[1:]):
        for i in range(1, n + 1):
            h = heads[i]
            if not punct_flags[i] and h > 0 and punct_flags[h]:
                valid_cands = {
                    cand_h: v
                    for cand_h, v in head_votes_per_word[i].items()
                    if cand_h != i and (cand_h == 0 or not punct_flags[cand_h])
                }
                if valid_cands:
                    new_h = max(valid_cands.keys(), key=lambda ch: valid_cands[ch])
                    heads[i] = new_h
                    labels[i] = _best_label_for_head(new_h, i, predictions, weights, sentence_idx)

    # 2. Enforce exactly one root (head == 0), preferring non-punctuation tokens
    roots = [i for i in range(1, n + 1) if heads[i] == 0]
    non_punct_roots = [i for i in roots if not punct_flags[i]]
    if non_punct_roots:
        best_r = max(non_punct_roots, key=lambda i: head_votes_per_word[i].get(0, 0.0))
    elif len(roots) > 0:
        best_r = max(roots, key=lambda i: head_votes_per_word[i].get(0, 0.0))
    else:
        cands = [i for i in range(1, n + 1) if not punct_flags[i]] or list(range(1, n + 1))
        best_r = max(cands, key=lambda i: head_votes_per_word[i].get(0, -1.0))

    for i in roots:
        if i != best_r:
            non_zero = {
                h: v
                for h, v in head_votes_per_word[i].items()
                if h != 0 and h != i and (punct_flags[i] or not punct_flags[h])
            }
            if non_zero:
                new_h = max(non_zero.keys(), key=lambda h: non_zero[h])
            else:
                new_h = best_r
            heads[i] = new_h
            labels[i] = _best_label_for_head(new_h, i, predictions, weights, sentence_idx)

    heads[best_r] = 0
    labels[best_r] = "root"
    root_idx = best_r

    # Ensure no non-root token has label == "root"
    for i in range(1, n + 1):
        if i != root_idx and labels[i] == "root":
            labels[i] = _best_label_for_head(heads[i], i, predictions, weights, sentence_idx)

    # 3. Break any directed cycles
    for _ in range(n + 1):
        visited = [0] * (n + 1)
        visited[0] = 2
        cycle_nodes = None

        for start in range(1, n + 1):
            if visited[start] != 0:
                continue
            path = []
            curr = start
            while curr != 0 and visited[curr] == 0:
                visited[curr] = 1
                path.append(curr)
                curr = heads[curr]
            if curr != 0 and visited[curr] == 1:
                idx = path.index(curr)
                cycle_nodes = set(path[idx:])
                break
            for node in path:
                visited[node] = 2

        if not cycle_nodes:
            break

        best_change = None
        best_loss = 1e9
        for u in cycle_nodes:
            current_score = head_votes_per_word[u].get(heads[u], 0.0)
            for cand_h, cand_score in head_votes_per_word[u].items():
                if cand_h == 0 or cand_h == u or cand_h in cycle_nodes:
                    continue
                if not punct_flags[u] and punct_flags[cand_h]:
                    continue
                anc = cand_h
                ok = True
                steps = 0
                while anc != 0 and steps <= n:
                    if anc in cycle_nodes:
                        ok = False
                        break
                    anc = heads[anc]
                    steps += 1
                if ok and anc == 0:
                    loss = current_score - cand_score
                    if loss < best_loss:
                        best_loss = loss
                        best_change = (u, cand_h)

        if best_change is not None:
            u, new_h = best_change
        else:
            u = min(cycle_nodes, key=lambda node: head_votes_per_word[node].get(heads[node], 0.0))
            new_h = root_idx
        heads[u] = new_h
        labels[u] = _best_label_for_head(new_h, u, predictions, weights, sentence_idx)

    return _refine_ud_tree(words, heads, labels)


def vote(predictions, sentences, weights=None):
    """
    Combine the answers of several parsers using weighted voting,
    global single-root cycle-free tree enforcement, and UD label refinement.
    """
    if weights is None:
        weights = [1.0] * len(predictions)

    voted_heads = []
    voted_labels = []
    for k in range(len(sentences)):
        words = sentences[k]["words"]
        heads = [-1]
        labels = [""]
        head_votes_per_word = [{}]
        n = len(words) - 1

        for i in range(1, n + 1):
            head_votes = {}
            first_seen_h = {}
            for m, (all_heads, _) in enumerate(predictions):
                head = all_heads[k][i]
                head_votes[head] = head_votes.get(head, 0.0) + weights[m]
                if head not in first_seen_h:
                    first_seen_h[head] = m
            head_votes_per_word.append(head_votes)

            best_head = max(head_votes.keys(), key=lambda h: (head_votes[h], -first_seen_h[h]))
            best_label = _best_label_for_head(best_head, i, predictions, weights, k)

            heads.append(best_head)
            labels.append(best_label)

        heads, labels = _enforce_valid_tree(
            words, heads, labels, head_votes_per_word, predictions, weights, k
        )
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
            [dev_sentences],
            [dev_tags],
            [dev_supertags],
            lexicon_ltr=lexicon_ltr,
            lexicon_rtl=lexicon_rtl,
        )
        predictions.append(results[0])
        weights.append(weight)
        gc.collect()
        uas, las = accuracy(dev_sentences, results[0][0], results[0][1])
        print("    this parser:      UAS %.2f  LAS %.2f" % (uas, las))
        voted_heads, voted_labels = vote(predictions, dev_sentences, weights)
        uas, las = accuracy(dev_sentences, voted_heads, voted_labels)
        print("    vote of %d parsers: UAS %.2f  LAS %.2f   <- LAS is the competition score" % (len(predictions), uas, las))

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
            [test_sentences],
            [test_tags],
            [test_supertags],
            lexicon_ltr=lexicon_ltr,
            lexicon_rtl=lexicon_rtl,
        )
        test_predictions.append(results[0])
        weights.append(weight)
        gc.collect()

    test_heads, test_labels = vote(test_predictions, test_sentences, weights)
    if COPY_KNOWN_SENTENCES:
        copied = copy_known_sentences(train_sentences, test_sentences, test_heads, test_labels)
        print("Copied the training answer for", copied, "test sentences")

    # Collect "HEAD|LABEL" for every row ID.
    # For training IDs present in sample.csv, use their exact ground-truth
    # labels from train.csv; for test IDs, use the ensemble's predictions.
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
