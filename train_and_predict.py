"""
Main script: trains everything and writes the submission file.

    python train_and_predict.py           -> train on all training data,
                                             write outputs/submission.csv
    python train_and_predict.py --check   -> train on 90% of the training data
                                             and print the score on the other 10%

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
]

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
    return train_tags, other_tags_lists


def train_one_parser(direction, seed, label_names, train_sentences, train_tags, sentence_lists, tags_lists):
    """Train one parser and return its (heads, labels) for each list of sentences."""
    if direction == "right-to-left":
        sentences_for_training = []
        tags_for_training = []
        for k in range(len(train_sentences)):
            sentences_for_training.append(flip_sentence(train_sentences[k]))
            tags_for_training.append(flip_list(train_tags[k]))
    else:
        sentences_for_training = train_sentences
        tags_for_training = train_tags

    parser = Parser(label_names)
    random_generator = random.Random(seed)
    for epoch in range(PARSER_EPOCHS):
        parser.train_one_epoch(sentences_for_training, tags_for_training, random_generator)
        print("    epoch", epoch + 1, "of", PARSER_EPOCHS, "done")
    parser.model.average_weights()

    results = []
    for j in range(len(sentence_lists)):
        sentences = sentence_lists[j]
        tags_list = tags_lists[j]
        all_heads = []
        all_labels = []
        for k in range(len(sentences)):
            if direction == "right-to-left":
                flipped = flip_sentence(sentences[k])
                heads, labels = parser.parse(flipped["words"], flip_list(tags_list[k]))
                heads, labels = flip_result(heads, labels)
            else:
                heads, labels = parser.parse(sentences[k]["words"], tags_list[k])
            all_heads.append(heads)
            all_labels.append(labels)
        results.append((all_heads, all_labels))
    return results


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

    predictions = []
    for (direction, seed) in PARSERS:
        print("Training parser:", direction, "seed", seed)
        results = train_one_parser(direction, seed, label_names, train_sentences, train_tags, [dev_sentences], [dev_tags])
        predictions.append(results[0])
        gc.collect()
        uas, las = accuracy(dev_sentences, results[0][0], results[0][1])
        print("    this parser:      UAS %.2f  LAS %.2f" % (uas, las))
        voted_heads, voted_labels = vote(predictions, dev_sentences)
        uas, las = accuracy(dev_sentences, voted_heads, voted_labels)
        print("    vote of %d parsers: UAS %.2f  LAS %.2f   <- LAS is the competition score" % (len(predictions), uas, las))


def final_mode(train_sentences):
    """Train on all training data and write the submission."""
    test_sentences = read_sentences(os.path.join(DATA_FOLDER, "test.csv"))
    print("Training sentences:", len(train_sentences), " test sentences:", len(test_sentences))

    label_names = get_label_names(train_sentences)
    train_tags, other_tags = make_tags(train_sentences, [test_sentences])
    test_tags = other_tags[0]

    # The sample file also lists the IDs of the training rows, so we predict
    # those too (with the same models) to fill every row of the submission.
    test_predictions = []
    train_predictions = []
    for (direction, seed) in PARSERS:
        print("Training parser:", direction, "seed", seed)
        results = train_one_parser(direction, seed, label_names, train_sentences, train_tags,
                                   [test_sentences, train_sentences], [test_tags, train_tags])
        test_predictions.append(results[0])
        train_predictions.append(results[1])
        gc.collect()

    test_heads, test_labels = vote(test_predictions, test_sentences)
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
    start_time = time.time()
    train_sentences = read_sentences(os.path.join(DATA_FOLDER, "train.csv"))
    if "--check" in sys.argv:
        check_mode(train_sentences)
    else:
        final_mode(train_sentences)
    print("Total time: %.1f minutes" % ((time.time() - start_time) / 60.0))


if __name__ == "__main__":
    main()
