"""
A fast multi-class averaged perceptron with bitmasked feature hashing,
Passive-Aggressive (PA-I / 1-best MIRA) norm-scaled margin updates,
and batch structured updates for sequence beam search.

How it works:
  * Every decision is described by a list of discrete features.
  * Every feature has one weight per class (per possible answer).
  * The score of a class is the sum of the weights of all present features.
  * Training (PA-I / 1-best MIRA): when the score margin between the correct
    class and the highest-scoring competing class is less than `margin`,
    compute the exact minimum-norm step size `tau = min(cap, loss / (2 * K))`
    (where K is the number of active features) so weights move just enough to
    satisfy the margin without overshooting.
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
        if correct_class == predicted_class or scale == 0.0:
            return
        step_scale = np.float32(scale * self.step)
        s = np.float32(scale)
        self.weights[rows, correct_class] += s
        self.weights[rows, predicted_class] -= s
        self.totals[rows, correct_class] += step_scale
        self.totals[rows, predicted_class] -= step_scale

    def update_pa(self, rows, correct_class, predicted_class, scores, margin=10.0, cap=1.0):
        """
        Passive-Aggressive (PA-I / 1-best MIRA) update:
          tau = min(cap, (margin - (s_correct - s_pred)) / (2 * len(rows)))
        """
        if correct_class == predicted_class:
            return
        loss = margin - (float(scores[correct_class]) - float(scores[predicted_class]))
        if loss <= 0.0:
            return
        denom = 2.0 * max(len(rows), 1)
        tau = loss / denom
        if tau > cap:
            tau = cap
        self.update(rows, correct_class, predicted_class, scale=tau)

    def update_one_class(self, rows, class_idx, delta):
        """Add `delta` to `class_idx` for every row in `rows`."""
        if delta == 0.0 or len(rows) == 0:
            return
        d = np.float32(delta)
        sd = np.float32(delta * self.step)
        self.weights[rows, class_idx] += d
        self.totals[rows, class_idx] += sd

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
