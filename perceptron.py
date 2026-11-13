"""
A multi-class averaged perceptron.

How it works, in short:

  * Every decision is described by a list of "features". A feature is a
    short piece of text, for example "word=the" or "s0t+n0t=det+nsubj".
  * Every feature has one weight per class (per possible answer).
  * The score of a class is the sum of the weights of all present features.
    The class with the highest score is the prediction.
  * Training: when the prediction is wrong, add 1 to the weights of the
    correct class and subtract 1 from the weights of the predicted class.
  * Averaging: at the end we use the average of the weights over all training
    steps instead of the last weights. This is much more stable.

Storage: all weights live in one big numpy table with one row per feature and
one column per class. To find the row of a feature we use a hash function
(crc32), which turns the text into a number. Two different features can
sometimes land on the same row, but that happens rarely enough to not matter
much, and it keeps the memory use fixed.
"""

import zlib

import numpy as np


class AveragedPerceptron:

    def __init__(self, n_classes, n_rows):
        self.n_classes = n_classes
        self.n_rows = n_rows
        self.weights = np.zeros((n_rows, n_classes), dtype=np.float32)
        # "totals" is used to compute the average weights cheaply (see below).
        self.totals = np.zeros((n_rows, n_classes), dtype=np.float32)
        self.step = 1
        self.is_averaged = False

    def feature_rows(self, features):
        """Turn a list of feature texts into a list of row numbers."""
        rows = []
        for feature in features:
            number = zlib.crc32(feature.encode("utf-8"))
            rows.append(number % self.n_rows)
        return rows

    def scores(self, rows, use_average=False):
        """Return a numpy array with the score of every class."""
        if use_average and not self.is_averaged:
            # Average weight = weights - totals / step  (see average_weights).
            table = self.weights[rows] - self.totals[rows] / self.step
        else:
            table = self.weights[rows]
        return table.sum(axis=0)

    def update(self, rows, correct_class, predicted_class):
        """The perceptron rule: reward the correct class, punish the wrong one."""
        if correct_class == predicted_class:
            return
        self.weights[rows, correct_class] += 1.0
        self.weights[rows, predicted_class] -= 1.0
        # Remember *when* the change happened, to compute the average later.
        self.totals[rows, correct_class] += self.step
        self.totals[rows, predicted_class] -= self.step

    def next_step(self):
        """Call this once after each training example (we use one sentence)."""
        self.step = self.step + 1

    def average_weights(self):
        """
        Replace the weights by their average over all steps.

        A change of +1 made at step t is present in the weights for the
        remaining (step - t) steps, so its share of the average is
        1 - t / step. Summing over all changes gives:
            average = weights - totals / step
        """
        np.divide(self.totals, self.step, out=self.totals)
        self.weights -= self.totals
        self.totals = None   # not needed any more, frees memory
        self.is_averaged = True
