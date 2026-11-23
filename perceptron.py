# simple averaged perceptron using feature hashing

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
        # hash each feature string to a row in the weight table
        mask = self.mask
        if mask is not None:
            return np.fromiter((hash(f) & mask for f in features), dtype=np.int32, count=len(features))
        n_rows = self.n_rows
        return np.fromiter((hash(f) % n_rows for f in features), dtype=np.int32, count=len(features))

    def scores(self, rows, use_average=False):
        if use_average and not self.is_averaged:
            table = self.weights[rows] - self.totals[rows] * (1.0 / self.step)
            return table.sum(axis=0)
        return self.weights[rows].sum(axis=0)

    def update(self, rows, correct_class, predicted_class, scale=1.0):
        if correct_class == predicted_class:
            return
        step_scale = np.float32(scale * self.step)
        s = np.float32(scale)
        self.weights[rows, correct_class] += s
        self.weights[rows, predicted_class] -= s
        self.totals[rows, correct_class] += step_scale
        self.totals[rows, predicted_class] -= step_scale

    def next_step(self):
        self.step += 1

    def average_weights(self):
        # finish weight averaging after training is done
        if self.is_averaged or self.totals is None:
            return
        np.divide(self.totals, np.float32(self.step), out=self.totals)
        self.weights -= self.totals
        self.totals = None
        self.is_averaged = True
