"""
Multi-class averaged perceptron with hashed features and
passive-aggressive (PA-I) step sizes.
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
        """Map feature strings to row indices in the weight table."""
        mask = self.mask
        if mask is not None:
            return np.fromiter((hash(f) & mask for f in features), dtype=np.int32, count=len(features))
        n_rows = self.n_rows
        return np.fromiter((hash(f) % n_rows for f in features), dtype=np.int32, count=len(features))

    def scores(self, rows, use_average=False):
        """Return the score vector for all classes."""
        if use_average and not self.is_averaged:
            table = self.weights[rows] - self.totals[rows] * (1.0 / self.step)
            return table.sum(axis=0)
        return self.weights[rows].sum(axis=0)

    def update(self, rows, correct_class, predicted_class, scale=1.0):
        """Reward the right class and penalize the predicted one."""
        if correct_class == predicted_class or scale == 0.0:
            return
        step_scale = np.float32(scale * self.step)
        s = np.float32(scale)
        self.weights[rows, correct_class] += s
        self.weights[rows, predicted_class] -= s
        self.totals[rows, correct_class] += step_scale
        self.totals[rows, predicted_class] -= step_scale

    def update_pa(self, rows, correct_class, predicted_class, scores, margin=10.0, cap=1.0):
        """Passive-aggressive (PA-I) update scaled by the number of active features."""
        if correct_class == predicted_class:
            return
        loss = margin - (float(scores[correct_class]) - float(scores[predicted_class]))
        if loss <= 0.0:
            return
        tau = loss / (2.0 * max(len(rows), 1))
        if tau > cap:
            tau = cap
        self.update(rows, correct_class, predicted_class, scale=tau)

    def update_one_class(self, rows, class_idx, delta):
        """Add delta to a single class for the given feature rows."""
        if delta == 0.0 or len(rows) == 0:
            return
        d = np.float32(delta)
        sd = np.float32(delta * self.step)
        self.weights[rows, class_idx] += d
        self.totals[rows, class_idx] += sd

    def next_step(self):
        """Advance the step counter after each sentence."""
        self.step += 1

    def average_weights(self):
        """Replace weights with their average over all training steps."""
        if self.is_averaged or self.totals is None:
            return
        np.divide(self.totals, np.float32(self.step), out=self.totals)
        self.weights -= self.totals
        self.totals = None
        self.is_averaged = True
