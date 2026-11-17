"""
Redo only the LAST step of train_and_predict.py: the vote of all parsers and
the rule-based fixes. It uses the parser outputs that train_and_predict.py
saved at the end of its run, so nothing is trained again. This takes a few
minutes instead of hours.

    python revote.py --check   -> try several vote settings on the check data
                                  and print a table of scores.
                                  Needs outputs/check_cache.pkl, made by
                                  "python train_and_predict.py --check".

    python revote.py           -> write outputs/submission.csv again from the
                                  saved test outputs, with the settings below
                                  and RULES in train_and_predict.py.
                                  Needs outputs/final_cache.pkl, made by
                                  "python train_and_predict.py".
"""

import os
import pickle
import sys
import time

from data_utils import accuracy, make_hashing_repeatable
import train_and_predict as main_script


# ----------------------------- settings -----------------------------
# Used by "python revote.py" (making the submission). Change them after
# looking at the table that "python revote.py --check" prints.

SKIP_SYSTEM = None        # None, or "arc-standard" / "arc-eager" / "arc-hybrid"
SKIP_DIRECTION = None     # None, or "right-to-left" / "left-to-right"
USE_MST_VOTES = True      # add the votes of the graph-based (MST) parser
USE_TREE_LABELER = True   # add the label scores of the full-tree labeler
EQUAL_WEIGHTS = False     # True: every parser's vote counts the same

# --------------------------------------------------------------------


def load_cache(path):
    if not os.path.exists(path):
        print("Cannot find", path)
        print("Run train_and_predict.py first; it saves this file at the end.")
        sys.exit(1)
    print("Loading", path, "...")
    with open(path, "rb") as cache_file:
        return pickle.load(cache_file)


def choose_parsers(cache, skip_system=None, skip_direction=None, only_system=None):
    """Pick which saved parser outputs take part in the vote."""
    predictions = []
    weights = []
    for k in range(len(cache["parsers"])):
        system_type, direction, seed, weight = cache["parsers"][k]
        if system_type == skip_system or direction == skip_direction:
            continue
        if only_system is not None and system_type != only_system:
            continue
        predictions.append(cache["predictions"][k])
        weights.append(cache["weights"][k])
    return predictions, weights


def run_vote(cache, predictions, weights, use_mst=True, use_labeler=True, rules=None):
    """The vote, then the rule-based fixes, then copying known sentences."""
    if rules is None:
        rules = main_script.RULES
    sentences = cache["sentences"]

    mst_list = None
    if use_mst:
        mst_list = cache["mst"]
    labeler = None
    if use_labeler:
        labeler = cache["tree_labeler"]

    heads, labels = main_script.vote(
        predictions,
        sentences,
        weights,
        tags_list=cache["tags"],
        supertags_list=cache["supertags"],
        mst_list=mst_list,
        upos_list=cache["upos"],
        tree_labeler=labeler,
        punct_forms=cache["punct_forms"],
    )
    heads, labels = main_script.apply_rules(sentences, heads, labels, cache["upos"], cache["punct_forms"], rules)
    if main_script.COPY_KNOWN_SENTENCES:
        main_script.copy_known_sentences(cache["train_sentences"], sentences, heads, labels)
    return heads, labels


def print_score(name, sentences, heads, labels):
    uas, las = accuracy(sentences, heads, labels)
    print("    %-30s UAS %.2f  LAS %.2f" % (name, uas, las))


def check():
    cache = load_cache(main_script.CHECK_CACHE)
    sentences = cache["sentences"]
    n_words = 0
    for sentence in sentences:
        n_words = n_words + len(sentence["words"]) - 1
    print("Check data: %d sentences, %d words" % (len(sentences), n_words))
    print("(differences smaller than about 0.2 are just noise)")

    # 1. Every parser on its own
    print("\nEach parser on its own:")
    for k in range(len(cache["parsers"])):
        system_type, direction, seed, weight = cache["parsers"][k]
        heads, labels, margins = cache["predictions"][k]
        name = system_type + " " + direction + " seed " + str(seed)
        print_score(name, sentences, heads, labels)

    # 2. The rule-based fixes on and off (one vote, then each set of rules)
    print()
    all_predictions, all_weights = choose_parsers(cache)
    voted_heads, voted_labels = main_script.vote(
        all_predictions,
        sentences,
        all_weights,
        tags_list=cache["tags"],
        supertags_list=cache["supertags"],
        mst_list=cache["mst"],
        upos_list=cache["upos"],
        tree_labeler=cache["tree_labeler"],
        punct_forms=cache["punct_forms"],
    )
    main_script.print_rule_table(cache["train_sentences"], sentences, voted_heads, voted_labels,
                                 cache["upos"], cache["punct_forms"])

    # 3. Different vote settings (all with the rules in RULES)
    print("\nVote settings (all with RULES from train_and_predict.py):")
    heads, labels = run_vote(cache, all_predictions, all_weights)
    print_score("all parsers (current)", sentences, heads, labels)

    equal_weights = []
    for weight in all_weights:
        equal_weights.append(1.0)
    heads, labels = run_vote(cache, all_predictions, equal_weights)
    print_score("equal weights", sentences, heads, labels)

    heads, labels = run_vote(cache, all_predictions, all_weights, use_mst=False)
    print_score("without MST votes", sentences, heads, labels)

    heads, labels = run_vote(cache, all_predictions, all_weights, use_labeler=False)
    print_score("without tree labeler", sentences, heads, labels)

    for system_type in ["arc-standard", "arc-eager", "arc-hybrid"]:
        predictions, weights = choose_parsers(cache, skip_system=system_type)
        if len(predictions) > 0 and len(predictions) < len(all_predictions):
            heads, labels = run_vote(cache, predictions, weights)
            print_score("without " + system_type, sentences, heads, labels)

    predictions, weights = choose_parsers(cache, skip_direction="right-to-left")
    if len(predictions) > 0 and len(predictions) < len(all_predictions):
        heads, labels = run_vote(cache, predictions, weights)
        print_score("without right-to-left", sentences, heads, labels)


def make_submission():
    cache = load_cache(main_script.FINAL_CACHE)
    predictions, weights = choose_parsers(cache, skip_system=SKIP_SYSTEM, skip_direction=SKIP_DIRECTION)
    if EQUAL_WEIGHTS:
        weights = []
        for k in range(len(predictions)):
            weights.append(1.0)
    print("Voting with %d parsers, MST votes: %s, tree labeler: %s, rules: %s"
          % (len(predictions), USE_MST_VOTES, USE_TREE_LABELER, main_script.RULES))
    heads, labels = run_vote(cache, predictions, weights, USE_MST_VOTES, USE_TREE_LABELER)
    main_script.write_final_submission(cache["sentences"], heads, labels, cache["train_sentences"])


def main():
    # Must come first: the saved models only work with the same hashing.
    make_hashing_repeatable()
    start_time = time.time()
    if "--check" in sys.argv:
        check()
    else:
        make_submission()
    print("Total time: %.1f minutes" % ((time.time() - start_time) / 60.0))


if __name__ == "__main__":
    main()
