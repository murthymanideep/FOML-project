"""
The tagger: gives every word a "tag" before parsing.

The data has no part-of-speech tags, but the parser works much better when it
knows roughly what kind of word it is looking at. So we train a simple tagger
that guesses, for each word, its dependency relation (det, nsubj, case, ...).
The parser then uses these guessed tags as extra information.

The tagger reads the sentence from left to right and uses an averaged
perceptron. Its features are the word itself, its ending (suffix), its shape,
the neighbour words and the tags it already chose for the previous words.
"""

import random

from data_utils import normalize_word, word_shape
from perceptron import AveragedPerceptron


START = "<START>"
END = "<END>"


class Tagger:

    def __init__(self, tag_names, n_rows=2 ** 19):
        self.tag_names = tag_names
        self.tag_number = {}
        for k in range(len(tag_names)):
            self.tag_number[tag_names[k]] = k
        self.model = AveragedPerceptron(len(tag_names), n_rows)

    def get_features(self, words, normal_words, i, previous_tag, previous_tag_2):
        """Features for the word at position i (positions start at 1)."""
        n = len(words) - 1
        word = words[i]
        normal = normal_words[i]

        # Neighbour words (or START / END outside the sentence).
        if i - 1 >= 1:
            previous_word = normal_words[i - 1]
        else:
            previous_word = START
        if i - 2 >= 1:
            previous_word_2 = normal_words[i - 2]
        else:
            previous_word_2 = START
        if i + 1 <= n:
            next_word = normal_words[i + 1]
        else:
            next_word = END
        if i + 2 <= n:
            next_word_2 = normal_words[i + 2]
        else:
            next_word_2 = END

        features = []
        features.append("bias")
        features.append("w=" + normal)
        features.append("suffix1=" + normal[-1:])
        features.append("suffix2=" + normal[-2:])
        features.append("suffix3=" + normal[-3:])
        features.append("suffix4=" + normal[-4:])
        features.append("prefix1=" + normal[:1])
        features.append("prefix3=" + normal[:3])
        features.append("shape=" + word_shape(word))
        features.append("first_word=" + str(i == 1) + "+" + word_shape(word)[:1])
        features.append("t-1=" + previous_tag)
        features.append("t-2+t-1=" + previous_tag_2 + "+" + previous_tag)
        features.append("t-1+w=" + previous_tag + "+" + normal)
        features.append("w-1=" + previous_word)
        features.append("w-1+w=" + previous_word + "+" + normal)
        features.append("w-1suffix3=" + previous_word[-3:])
        features.append("w-2=" + previous_word_2)
        features.append("w+1=" + next_word)
        features.append("w+w+1=" + normal + "+" + next_word)
        features.append("w+1suffix3=" + next_word[-3:])
        features.append("w+2=" + next_word_2)
        features.append("w+1+w+2=" + next_word + "+" + next_word_2)
        features.append("t-1+w+1=" + previous_tag + "+" + next_word)
        features.append("position=" + str(min(i, 4)) + "+" + str(min(n - i, 4)))
        return features

    def tag(self, words, correct_tags=None):
        """
        Tag one sentence. Returns a list of tags, with "<ROOT>" at position 0.

        If correct_tags is given we are training: after each word the model
        is updated when its guess was wrong.
        """
        normal_words = []
        for word in words:
            normal_words.append(normalize_word(word))

        tags = [words[0]]
        previous_tag = START
        previous_tag_2 = START
        for i in range(1, len(words)):
            features = self.get_features(words, normal_words, i, previous_tag, previous_tag_2)
            rows = self.model.feature_rows(features)
            scores = self.model.scores(rows)
            guess = int(scores.argmax())

            if correct_tags is not None:
                correct = self.tag_number[correct_tags[i]]
                self.model.update(rows, correct, guess)
                # Use the correct tag as history while training.
                guessed_tag = correct_tags[i]
            else:
                guessed_tag = self.tag_names[guess]

            tags.append(guessed_tag)
            previous_tag_2 = previous_tag
            previous_tag = guessed_tag
        return tags

    def train(self, sentences, epochs, seed=1):
        """Train on sentences; the "labels" of the sentence are the tags to learn."""
        random_generator = random.Random(seed)
        order = list(range(len(sentences)))
        for epoch in range(epochs):
            random_generator.shuffle(order)
            for k in order:
                sentence = sentences[k]
                self.tag(sentence["words"], sentence["labels"])
                self.model.next_step()
        self.model.average_weights()


def train_tagger(sentences, tag_names, epochs):
    tagger = Tagger(tag_names)
    tagger.train(sentences, epochs)
    return tagger


def jackknife_tags(sentences, tag_names, epochs, n_folds=5):
    """
    Give tags to the TRAINING sentences in a fair way.

    If we tagged the training sentences with a tagger that was trained on
    them, the tags would be almost perfect, and the parser would learn to
    trust them too much. On new sentences the tags will contain mistakes.
    So we split the training data into n_folds parts; each part is tagged by
    a tagger trained on the other parts. ("jackknifing")
    """
    all_tags = []
    for k in range(len(sentences)):
        all_tags.append(None)

    for fold in range(n_folds):
        train_part = []
        held_out = []
        for k in range(len(sentences)):
            if k % n_folds == fold:
                held_out.append(k)
            else:
                train_part.append(sentences[k])

        tagger = train_tagger(train_part, tag_names, epochs)
        for k in held_out:
            all_tags[k] = tagger.tag(sentences[k]["words"])
        print("    jackknife fold", fold + 1, "of", n_folds, "done")
    return all_tags


def tag_accuracy(sentences, tags):
    total = 0
    correct = 0
    for k in range(len(sentences)):
        for i in range(1, len(sentences[k]["words"])):
            total = total + 1
            if tags[k][i] == sentences[k]["labels"][i]:
                correct = correct + 1
    return 100.0 * correct / total


def make_tag_targets(sentences):
    """
    The tags we teach the tagger: the relation of the word plus the side of
    its head: "<" when the head is on the left, ">" when it is on the right.
    Example: "det>" means "a determiner whose head comes later".
    The root word just gets "root".
    Returns new sentence dictionaries with these tags as "labels".
    """
    result = []
    for sentence in sentences:
        tags = [sentence["labels"][0]]
        for i in range(1, len(sentence["words"])):
            head = sentence["heads"][i]
            tag = sentence["labels"][i]
            if head != 0 and head < i:
                tag = tag + "<"
            if head > i:
                tag = tag + ">"
            tags.append(tag)
        new_sentence = {}
        new_sentence["words"] = sentence["words"]
        new_sentence["labels"] = tags
        result.append(new_sentence)
    return result
