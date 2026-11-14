"""
Main script: trains everything and writes the submission file.

    python train_and_predict.py           -> train on all training data,
                                             write outputs/submission.csv
    python train_and_predict.py --check   -> train on 90% of the training data
                                             and print the score on the other 10%

Extra options (can be combined with --check):
    --one-parser       train only the first parser (fast, for experiments)
    --no-second-pass   do not use the second-pass tagger
    --no-copy          do not copy answers of sentences seen in training
    --stack            stacking: final parsers see the answers of guide parsers
    --big-table        use a weight table twice as big (needs about 1.8 GB)
    --tiny             use only 300 sentences, to test that the code runs

Steps:
  1. Read the data.
  2. Train a tagger and give every word a tag (a guess of its relation).
  3. Train several parsers (some read left-to-right, some right-to-left).
  4. Let the parsers vote on the head and label of every word.
  5. Write the answers in the format of data/sample.csv.
"""

import gc
import os
import random
import sys
import time

from data_utils import read_sentences, read_sample_ids, write_submission, accuracy
from tagger import train_tagger, jackknife_tags, make_tag_targets, tag_accuracy
from parser import Parser, parse_all, flip_sentence, flip_list, flip_result


# ----------------------------- settings -----------------------------

DATA_FOLDER = "data"
OUTPUT_FOLDER = "outputs"

TAGGER_EPOCHS = 5
JACKKNIFE_FOLDS = 5
PARSER_EPOCHS = 10

# The parsers that vote: (reading direction, random seed).
PARSERS = [
    ("left-to-right", 1),
    ("right-to-left", 1),
    ("left-to-right", 2),
    ("right-to-left", 2),
    ("left-to-right", 3),
    ("right-to-left", 3),
    ("left-to-right", 4),
]

# Second-pass tagger: a second tagger that can also see the first tagger's
# guesses for the words on the right. Switch off with --no-second-pass.
USE_SECOND_PASS_TAGGER = True

# If a test sentence is word-for-word the same as a training sentence, copy
# the training answer. Switch off with --no-copy.
COPY_KNOWN_SENTENCES = True

# Stacking: first train "guide" parsers, then let the final parsers look at
# the guides' answers. Switch on with --stack.
USE_STACKING = False
GUIDE_PARSERS = [("left-to-right", 101), ("right-to-left", 101)]
GUIDE_EPOCHS = 6

# Size of the parser's weight table: 2 ** PARSER_TABLE_BITS rows.
# 20 needs about 0.9 GB of memory, 21 (option --big-table) about 1.8 GB.
PARSER_TABLE_BITS = 20

# --------------------------------------------------------------------


def get_label_names(sentences):
    names = set()
    for sentence in sentences:
        for i in range(1, len(sentence["labels"])):
            names.add(sentence["labels"][i])
    return sorted(names)


def make_tags(train_sentences, other_sentence_lists):
    """
    Tag the training sentences with jackknifing, and the other sentences
    (dev or test) with a tagger trained on all training sentences.
    """
    tag_sentences = make_tag_targets(train_sentences)
    tag_names = get_label_names(tag_sentences)

    print("Training the tagger (jackknifing for the training data)...")
    train_tags = jackknife_tags(tag_sentences, tag_names, TAGGER_EPOCHS, JACKKNIFE_FOLDS)
    print("    tag accuracy on training data (jackknifed): %.2f" % tag_accuracy(tag_sentences, train_tags))

    print("Training the tagger on all training sentences...")
    tagger = train_tagger(tag_sentences, tag_names, TAGGER_EPOCHS)
    other_tags_lists = []
    for sentences in other_sentence_lists:
        tags_list = []
        for sentence in sentences:
            tags_list.append(tagger.tag(sentence["words"]))
        other_tags_lists.append(tags_list)
    tagger = None
    gc.collect()

    if not USE_SECOND_PASS_TAGGER:
        return train_tags, other_tags_lists

    # ----- second pass -----
    # The second tagger gets the first tagger's guesses as extra features.
    # For the training sentences these guesses are the jackknifed ones, so
    # they contain realistic mistakes. The output for the training sentences
    # is jackknifed again, for the same reason as before.
    print("Training the second-pass tagger (jackknifing)...")
    train_tags_2 = jackknife_tags(tag_sentences, tag_names, TAGGER_EPOCHS, JACKKNIFE_FOLDS, train_tags)
    print("    second-pass tag accuracy on training data (jackknifed): %.2f" % tag_accuracy(tag_sentences, train_tags_2))

    print("Training the second-pass tagger on all training sentences...")
    tagger_2 = train_tagger(tag_sentences, tag_names, TAGGER_EPOCHS, train_tags)
    other_tags_lists_2 = []
    for j in range(len(other_sentence_lists)):
        sentences = other_sentence_lists[j]
        tags_list = []
        for k in range(len(sentences)):
            tags_list.append(tagger_2.tag(sentences[k]["words"], None, other_tags_lists[j][k]))
        other_tags_lists_2.append(tags_list)
    tagger_2 = None
    gc.collect()
    return train_tags_2, other_tags_lists_2


def copy_known_sentences(train_sentences, sentences, all_heads, all_labels):
    """
    If a sentence has exactly the same words as a training sentence, use the
    training answer for it. (Skipped when the training data has two
    different answers for the same words.) Returns how many were copied.
    """
    answer_of_words = {}
    for sentence in train_sentences:
        key = "\t".join(sentence["words"])
        answer = (sentence["heads"], sentence["labels"])
        if key not in answer_of_words:
            answer_of_words[key] = answer
        elif answer_of_words[key] is not None and answer_of_words[key] != answer:
            answer_of_words[key] = None     # training data disagrees with itself

    copied = 0
    for k in range(len(sentences)):
        key = "\t".join(sentences[k]["words"])
        answer = answer_of_words.get(key)
        if answer is not None:
            all_heads[k] = answer[0]
            all_labels[k] = answer[1]
            copied = copied + 1
    return copied


def flip_guides(guides):
    """Turn the guides of one sentence around (for a right-to-left parser)."""
    if guides is None:
        return None
    flipped = []
    for (heads, labels) in guides:
        flipped.append(flip_result(heads, labels))
    return flipped


def train_one_parser(direction, seed, label_names, epochs, train_sentences, train_tags,
                     sentence_lists, tags_lists, train_guides=None, guides_lists=None):
    """
    Train one parser and return its (heads, labels) for each list of sentences.

    train_guides / guides_lists are only used with stacking: for every
    sentence, the answers of the guide parsers.
    """
    guides_for_training = None
    if direction == "right-to-left":
        sentences_for_training = []
        tags_for_training = []
        for k in range(len(train_sentences)):
            sentences_for_training.append(flip_sentence(train_sentences[k]))
            tags_for_training.append(flip_list(train_tags[k]))
        if train_guides is not None:
            guides_for_training = []
            for k in range(len(train_sentences)):
                guides_for_training.append(flip_guides(train_guides[k]))
    else:
        sentences_for_training = train_sentences
        tags_for_training = train_tags
        guides_for_training = train_guides

    parser = Parser(label_names, 2 ** PARSER_TABLE_BITS)
    random_generator = random.Random(seed)
    for epoch in range(epochs):
        parser.train_one_epoch(sentences_for_training, tags_for_training, random_generator, guides_for_training)
        print("    epoch", epoch + 1, "of", epochs, "done")
    parser.model.average_weights()

    results = []
    for j in range(len(sentence_lists)):
        sentences = sentence_lists[j]
        tags_list = tags_lists[j]
        all_heads = []
        all_labels = []
        for k in range(len(sentences)):
            guides = None
            if guides_lists is not None:
                guides = guides_lists[j][k]
            if direction == "right-to-left":
                flipped = flip_sentence(sentences[k])
                heads, labels = parser.parse(flipped["words"], flip_list(tags_list[k]), False, flip_guides(guides))
                heads, labels = flip_result(heads, labels)
            else:
                heads, labels = parser.parse(sentences[k]["words"], tags_list[k], False, guides)
            all_heads.append(heads)
            all_labels.append(labels)
        results.append((all_heads, all_labels))
    parser = None
    gc.collect()
    return results


def make_guides(label_names, train_sentences, train_tags, other_sentence_lists, other_tags_lists):
    """
    Stacking, round 1: make the "guides" (answers of the guide parsers).

    For the TRAINING sentences we use jackknifing again: the sentences are
    split into parts, and each part is parsed by guide parsers trained on
    the other parts. This way the guides make realistic mistakes on the
    training sentences, just like they will on the test sentences.

    For the other sentences (dev or test) the guide parsers are trained on
    all training sentences.

    Returns (train_guides, other_guides_lists). The guides of one sentence
    are a list with one (heads, labels) pair per guide parser.
    """
    n = len(train_sentences)
    train_guides = []
    for k in range(n):
        train_guides.append([])

    for fold in range(JACKKNIFE_FOLDS):
        part_sentences = []
        part_tags = []
        held_out = []
        held_out_sentences = []
        held_out_tags = []
        for k in range(n):
            if k % JACKKNIFE_FOLDS == fold:
                held_out.append(k)
                held_out_sentences.append(train_sentences[k])
                held_out_tags.append(train_tags[k])
            else:
                part_sentences.append(train_sentences[k])
                part_tags.append(train_tags[k])

        for (direction, seed) in GUIDE_PARSERS:
            print("Guide parser, fold", fold + 1, "of", JACKKNIFE_FOLDS, ":", direction)
            results = train_one_parser(direction, seed, label_names, GUIDE_EPOCHS, part_sentences, part_tags,
                                       [held_out_sentences], [held_out_tags])
            all_heads, all_labels = results[0]
            for position in range(len(held_out)):
                k = held_out[position]
                train_guides[k].append((all_heads[position], all_labels[position]))

    # How good are the guides on the training sentences?
    for g in range(len(GUIDE_PARSERS)):
        guide_heads = []
        guide_labels = []
        for k in range(n):
            guide_heads.append(train_guides[k][g][0])
            guide_labels.append(train_guides[k][g][1])
        uas, las = accuracy(train_sentences, guide_heads, guide_labels)
        print("    guide %d on training data (jackknifed): UAS %.2f  LAS %.2f" % (g, uas, las))

    other_guides_lists = []
    for sentences in other_sentence_lists:
        guides_list = []
        for k in range(len(sentences)):
            guides_list.append([])
        other_guides_lists.append(guides_list)

    for (direction, seed) in GUIDE_PARSERS:
        print("Guide parser on all training sentences:", direction)
        results = train_one_parser(direction, seed, label_names, GUIDE_EPOCHS, train_sentences, train_tags,
                                   other_sentence_lists, other_tags_lists)
        for j in range(len(other_sentence_lists)):
            all_heads, all_labels = results[j]
            for k in range(len(other_sentence_lists[j])):
                other_guides_lists[j][k].append((all_heads[k], all_labels[k]))
    return train_guides, other_guides_lists


def vote(predictions, sentences):
    """
    Combine the answers of several parsers.
    For each word: first choose the head most parsers agree on; then, among
    the parsers that chose this head, the most common label. On a tie, the
    answer of the earlier parser in the list wins.
    """
    voted_heads = []
    voted_labels = []
    for k in range(len(sentences)):
        heads = [-1]
        labels = [""]
        for i in range(1, len(sentences[k]["words"])):
            head_votes = {}
            for (all_heads, all_labels) in predictions:
                head = all_heads[k][i]
                head_votes[head] = head_votes.get(head, 0) + 1
            best_head = None
            for (all_heads, all_labels) in predictions:
                head = all_heads[k][i]
                if best_head is None or head_votes[head] > head_votes[best_head]:
                    best_head = head

            label_votes = {}
            for (all_heads, all_labels) in predictions:
                if all_heads[k][i] == best_head:
                    label = all_labels[k][i]
                    label_votes[label] = label_votes.get(label, 0) + 1
            best_label = None
            for (all_heads, all_labels) in predictions:
                if all_heads[k][i] == best_head:
                    label = all_labels[k][i]
                    if best_label is None or label_votes[label] > label_votes[best_label]:
                        best_label = label

            heads.append(best_head)
            labels.append(best_label)
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
    train_tags, other_tags = make_tags(train_sentences, [dev_sentences])
    dev_tags = other_tags[0]
    print("    tag accuracy on check data: %.2f" % tag_accuracy(make_tag_targets(dev_sentences), dev_tags))

    train_guides = None
    dev_guides_lists = None
    if USE_STACKING:
        train_guides, dev_guides_lists = make_guides(label_names, train_sentences, train_tags, [dev_sentences], [dev_tags])
        for g in range(len(GUIDE_PARSERS)):
            guide_heads = []
            guide_labels = []
            for k in range(len(dev_sentences)):
                guide_heads.append(dev_guides_lists[0][k][g][0])
                guide_labels.append(dev_guides_lists[0][k][g][1])
            uas, las = accuracy(dev_sentences, guide_heads, guide_labels)
            print("    guide %d on check data: UAS %.2f  LAS %.2f" % (g, uas, las))

    predictions = []
    for (direction, seed) in PARSERS:
        print("Training parser:", direction, "seed", seed)
        results = train_one_parser(direction, seed, label_names, PARSER_EPOCHS, train_sentences, train_tags,
                                   [dev_sentences], [dev_tags], train_guides, dev_guides_lists)
        predictions.append(results[0])
        gc.collect()
        uas, las = accuracy(dev_sentences, results[0][0], results[0][1])
        print("    this parser:      UAS %.2f  LAS %.2f" % (uas, las))
        voted_heads, voted_labels = vote(predictions, dev_sentences)
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
    train_tags, other_tags = make_tags(train_sentences, [test_sentences])
    test_tags = other_tags[0]

    # The sample file also lists the IDs of the training rows, so we predict
    # those too (with the same models) to fill every row of the submission.
    train_guides = None
    guides_lists = None
    if USE_STACKING:
        train_guides, test_guides_lists = make_guides(label_names, train_sentences, train_tags, [test_sentences], [test_tags])
        guides_lists = [test_guides_lists[0], train_guides]

    test_predictions = []
    train_predictions = []
    for (direction, seed) in PARSERS:
        print("Training parser:", direction, "seed", seed)
        results = train_one_parser(direction, seed, label_names, PARSER_EPOCHS, train_sentences, train_tags,
                                   [test_sentences, train_sentences], [test_tags, train_tags],
                                   train_guides, guides_lists)
        test_predictions.append(results[0])
        train_predictions.append(results[1])
        gc.collect()

    test_heads, test_labels = vote(test_predictions, test_sentences)
    if COPY_KNOWN_SENTENCES:
        copied = copy_known_sentences(train_sentences, test_sentences, test_heads, test_labels)
        print("Copied the training answer for", copied, "test sentences")
    train_heads, train_labels = vote(train_predictions, train_sentences)
    uas, las = accuracy(train_sentences, train_heads, train_labels)
    print("Score on the training data itself (too optimistic): UAS %.2f  LAS %.2f" % (uas, las))

    # Collect "HEAD|LABEL" for every row ID.
    answer_of_id = {}
    sentence_groups = [(test_sentences, test_heads, test_labels), (train_sentences, train_heads, train_labels)]
    for (sentences, all_heads, all_labels) in sentence_groups:
        for k in range(len(sentences)):
            for i in range(1, len(sentences[k]["words"])):
                row_id = sentences[k]["row_ids"][i - 1]
                answer_of_id[row_id] = str(all_heads[k][i]) + "|" + all_labels[k][i]

    sample_ids, header = read_sample_ids(os.path.join(DATA_FOLDER, "sample.csv"))
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    output_path = os.path.join(OUTPUT_FOLDER, "submission.csv")
    write_submission(output_path, header, sample_ids, answer_of_id)
    print("Wrote", len(sample_ids), "rows to", output_path)


def main():
    global PARSERS, USE_SECOND_PASS_TAGGER, COPY_KNOWN_SENTENCES, USE_STACKING
    global PARSER_TABLE_BITS, PARSER_EPOCHS, GUIDE_EPOCHS, TAGGER_EPOCHS
    if "--stack" in sys.argv:
        USE_STACKING = True
    if "--big-table" in sys.argv:
        PARSER_TABLE_BITS = 21
    if "--one-parser" in sys.argv:
        PARSERS = PARSERS[:1]
    if "--no-second-pass" in sys.argv:
        USE_SECOND_PASS_TAGGER = False
    if "--no-copy" in sys.argv:
        COPY_KNOWN_SENTENCES = False
    start_time = time.time()
    train_sentences = read_sentences(os.path.join(DATA_FOLDER, "train.csv"))
    if "--tiny" in sys.argv:
        # Only to test quickly that the code runs. The scores mean nothing.
        train_sentences = train_sentences[:300]
        PARSER_EPOCHS = 1
        GUIDE_EPOCHS = 1
        TAGGER_EPOCHS = 1
        PARSER_TABLE_BITS = 14
    if "--check" in sys.argv or "--tiny" in sys.argv:
        check_mode(train_sentences)
    else:
        final_mode(train_sentences)
    print("Total time: %.1f minutes" % ((time.time() - start_time) / 60.0))


if __name__ == "__main__":
    main()
