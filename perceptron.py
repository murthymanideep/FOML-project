"""
A fast multi-class averaged perceptron with bitmasked feature hashing
and optional margin-scaled updates.

How it works:
  * Every decision is described by a list of discrete features.
  * Every feature has one weight per class (per possible answer).
  * The score of a class is the sum of the weights of all present features.
  * Training: when the prediction is wrong (or within the safety margin),
    add `scale` to the weights of the correct class and subtract `scale`
    from the weights of the predicted class.
  * Averaging: at the end we use the average of the weights over all training
    steps instead of the final step weights, which is much more stable and
    generalizes significantly better.
"""

import numpy as np


class AveragedPerceptron:

    def __init__(self, n_classes, n_rows):
        self.n_classes = n_classes
        self.n_rows = n_rows
        self.mask = n_rows - 1 if (n_rows & (n_rows - 1)) == 0 else None
        self.weights = np.zeros((n_rows, n_classes), dtype=np.float32)
        self.totals = np.zeros((n_rows, n_classes), dtype=np.float32)
        self.step = 1
        self.is_averaged = False

    def feature_rows(self, features):
        """Turn a list of feature strings into a numpy array of row indices."""
        mask = self.mask
        if mask is not None:
            return np.fromiter((hash(f) & mask for f in features), dtype=np.int32, count=len(features))
        n_rows = self.n_rows
        return np.fromiter((hash(f) % n_rows for f in features), dtype=np.int32, count=len(features))

    def scores(self, rows, use_average=False):
        """Return a numpy array with the score of every class."""
        if use_average and not self.is_averaged:
            table = self.weights[rows] - self.totals[rows] * (1.0 / self.step)
            return table.sum(axis=0)
        return self.weights[rows].sum(axis=0)

    def update(self, rows, correct_class, predicted_class, scale=1.0):
        """The perceptron rule: reward the correct class, punish the wrong one."""
        if correct_class == predicted_class:
            return
        step_scale = np.float32(scale * self.step)
        s = np.float32(scale)
        self.weights[rows, correct_class] += s
        self.weights[rows, predicted_class] -= s
        self.totals[rows, correct_class] += step_scale
        self.totals[rows, predicted_class] -= step_scale

    def next_step(self):
        """Call this once after each training example (one sentence)."""
        self.step += 1

    def average_weights(self):
        """
        Replace the weights by their average over all training steps:
            average = weights - totals / step
        """
        if self.is_averaged or self.totals is None:
            return
        np.divide(self.totals, np.float32(self.step), out=self.totals)
        self.weights -= self.totals
        self.totals = None
        self.is_averaged = True
