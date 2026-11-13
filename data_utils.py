"""
Helper functions for reading the data, writing the submission file and
measuring accuracy.

A sentence is stored as a dictionary with these keys:

    "name"    : the sentence id, for example "train_000001"
    "row_ids" : the ID column of each word (list of strings)
    "words"   : the words of the sentence
    "heads"   : the correct head of each word   (only for training data)
    "labels"  : the correct relation of each word (only for training data)

IMPORTANT: position 0 of "words", "heads" and "labels" is a fake word called
"<ROOT>". The real words start at position 1. This way the word at position i
has token_id i, and a head of 0 means "the root", exactly like in the data.
"""

import csv


ROOT_WORD = "<ROOT>"


def make_empty_sentence(name):
    sentence = {}
    sentence["name"] = name
    sentence["row_ids"] = []
    sentence["words"] = [ROOT_WORD]
    sentence["heads"] = [-1]
    sentence["labels"] = [ROOT_WORD]
    return sentence


def read_sentences(file_path):
    """Read train.csv or test.csv and group the rows into sentences."""
    sentences = []
    current_sentence = None

    with open(file_path, newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            # A new sentence starts when the sentence_id changes.
            if current_sentence is None or row["sentence_id"] != current_sentence["name"]:
                current_sentence = make_empty_sentence(row["sentence_id"])
                sentences.append(current_sentence)

            # Check that the words come in order 1, 2, 3, ...
            expected_token_id = len(current_sentence["words"])
            if int(row["token_id"]) != expected_token_id:
                raise ValueError("Unexpected token_id in row " + row["ID"])

            current_sentence["row_ids"].append(row["ID"])
            current_sentence["words"].append(row["token"])

            # The training file also has the answer, written like "10|punct".
            if "target" in row:
                parts = row["target"].split("|")
                current_sentence["heads"].append(int(parts[0]))
                current_sentence["labels"].append(parts[1])

    return sentences


def read_sample_ids(file_path):
    """Return the list of IDs (in order) and the column names of the sample file."""
    ids = []
    with open(file_path, newline="", encoding="utf-8") as csv_file:
        reader = csv.reader(csv_file)
        header = next(reader)
        for row in reader:
            ids.append(row[0])
    return ids, header


def write_submission(file_path, header, ids, answer_of_id):
    """Write one line "ID,HEAD|LABEL" for every ID, in the given order."""
    with open(file_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)
        for row_id in ids:
            writer.writerow([row_id, answer_of_id[row_id]])


def accuracy(sentences, predicted_heads, predicted_labels):
    """
    Return (UAS, LAS) as percentages.

    UAS = share of words with the correct head.
    LAS = share of words with the correct head AND the correct label.
          LAS is the score used in the competition.
    """
    total = 0
    head_correct = 0
    both_correct = 0
    for k in range(len(sentences)):
        sentence = sentences[k]
        for i in range(1, len(sentence["words"])):
            total = total + 1
            if predicted_heads[k][i] == sentence["heads"][i]:
                head_correct = head_correct + 1
                if predicted_labels[k][i] == sentence["labels"][i]:
                    both_correct = both_correct + 1
    uas = 100.0 * head_correct / total
    las = 100.0 * both_correct / total
    return uas, las


def normalize_word(word):
    """Lower-case a word and replace numbers by a common symbol."""
    if word == ROOT_WORD:
        return word
    if word == "":
        return "<EMPTY>"
    if word.isdigit() and len(word) == 4:
        return "<YEAR>"
    if word[0].isdigit():
        return "<NUMBER>"
    return word.lower()


def word_shape(word):
    """
    Describe how a word looks: upper case -> X, lower case -> x, digit -> d,
    other characters stay the same. Repeated symbols are written once.
    Example: "McDonald's" -> "XxXx'x",  "2026" -> "d",  "U.S." -> "X.X."
    """
    shape = ""
    for character in word:
        if character.isupper():
            symbol = "X"
        elif character.islower():
            symbol = "x"
        elif character.isdigit():
            symbol = "d"
        else:
            symbol = character
        if len(shape) == 0 or shape[-1] != symbol:
            shape = shape + symbol
    return shape
