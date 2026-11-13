"""
The dependency parser (transition-based, "arc-hybrid" system).

The parser builds the tree step by step, like this:

  * a STACK of words that are waiting for a head (starts empty),
  * a BUFFER of words not looked at yet (starts as: 1, 2, ..., n, ROOT).
    The fake ROOT word sits at the very end of the buffer.

At every step the parser chooses one of three moves:

  SHIFT        : move the first buffer word onto the stack.
  LEFT(label)  : the top stack word gets the first buffer word as its head,
                 with this label. The top word is removed from the stack.
  RIGHT(label) : the top stack word gets the second stack word as its head,
                 with this label. The top word is removed from the stack.

When the stack is empty and only ROOT is left in the buffer, every word has
a head and the tree is finished. The word that got ROOT as head is the root.

Choosing the move is a classification problem. We describe the current
situation with features (the words and tags on top of the stack and at the
front of the buffer, their children, distances, ...) and let an averaged
perceptron pick the best move.

Training uses a "dynamic oracle": at each step we compute which moves still
allow the best possible tree (moves with zero cost). If the model picks a
move with a cost, we update the model. Then we continue with the move the
model picked, even if it was wrong, so the model also learns what to do
after its own mistakes.
"""

import random

import numpy as np

from data_utils import normalize_word
from perceptron import AveragedPerceptron


SHIFT = 0
LEFT = 1
RIGHT = 2
NONE = -1          # "no word here"
VERY_LOW = -1.0e9  # score given to moves that are not allowed


class ParserState:
    """Everything the parser knows while building one tree."""

    def __init__(self, n_words):
        self.n = n_words
        self.stack = []
        self.buffer_position = 1      # next word in the buffer; n+1 means only ROOT is left
        self.heads = []
        self.labels = []
        self.left_children = []
        self.right_children = []
        for i in range(n_words + 1):
            self.heads.append(NONE)
            self.labels.append("<NONE>")
            self.left_children.append([])
            self.right_children.append([])

    def buffer_word(self, k):
        """The k-th word of the buffer (k = 0 is the front). ROOT is word 0."""
        position = self.buffer_position + k
        if position <= self.n:
            return position
        if position == self.n + 1:
            return 0
        return NONE

    def stack_word(self, k):
        """The k-th word of the stack (k = 0 is the top)."""
        if len(self.stack) > k:
            return self.stack[len(self.stack) - 1 - k]
        return NONE

    def is_finished(self):
        return len(self.stack) == 0 and self.buffer_position > self.n

    def in_buffer(self, word):
        """Is this word still in the buffer? (ROOT always is.)"""
        return word == 0 or word >= self.buffer_position

    def can_shift(self):
        return self.buffer_position <= self.n

    def can_left(self):
        if len(self.stack) == 0:
            return False
        # Only one word may be attached to ROOT, and only at the very end.
        if self.buffer_word(0) == 0:
            return len(self.stack) == 1
        return True

    def can_right(self):
        return len(self.stack) >= 2

    def add_arc(self, head, dependent, label):
        self.heads[dependent] = head
        self.labels[dependent] = label
        if dependent < head:
            self.left_children[head].append(dependent)
        else:
            self.right_children[head].append(dependent)

    def apply(self, move, label):
        if move == SHIFT:
            self.stack.append(self.buffer_position)
            self.buffer_position = self.buffer_position + 1
        elif move == LEFT:
            dependent = self.stack.pop()
            self.add_arc(self.buffer_word(0), dependent, label)
        else:
            dependent = self.stack.pop()
            self.add_arc(self.stack[-1], dependent, label)


def leftmost_child(state, word, k):
    """The k-th leftmost child of a word (k = 1 or 2), or NONE."""
    if word == NONE:
        return NONE
    children = sorted(state.left_children[word])
    if len(children) >= k:
        return children[k - 1]
    return NONE


def rightmost_child(state, word, k):
    """The k-th rightmost child of a word (k = 1 or 2), or NONE."""
    if word == NONE:
        return NONE
    children = sorted(state.right_children[word])
    if len(children) >= k:
        return children[len(children) - k]
    return NONE


def distance_text(a, b):
    """Distance between two words, put into a few groups."""
    if a == NONE or b == NONE:
        return "none"
    if a == 0 or b == 0:
        return "root"
    d = abs(a - b)
    if d <= 4:
        return str(d)
    if d <= 9:
        return "5-9"
    return "10+"


def move_costs(state, gold_heads, gold_children):
    """
    Dynamic oracle: how many correct arcs become impossible after each move?
    Returns [cost of SHIFT, cost of LEFT, cost of RIGHT]. Moves that are not
    allowed get a very high cost.
    """
    s0 = state.stack_word(0)
    s1 = state.stack_word(1)
    b0 = state.buffer_word(0)

    shift_cost = 1000
    if state.can_shift():
        shift_cost = 0
        # After SHIFT, b0 can only get its head from s0 or from the buffer,
        # so a head deeper in the stack is lost.
        head = gold_heads[b0]
        if head != s0 and head in state.stack:
            shift_cost = shift_cost + 1
        # After SHIFT, words in the stack can no longer become children of b0.
        for child in gold_children[b0]:
            if child in state.stack:
                shift_cost = shift_cost + 1

    left_cost = 1000
    if state.can_left():
        left_cost = 0
        # s0 gets b0 as head; if its real head is s1 or later in the buffer, it is lost.
        head = gold_heads[s0]
        if head != b0 and (head == s1 or state.in_buffer(head)):
            left_cost = left_cost + 1
        # s0 leaves the stack, so its children still in the buffer are lost.
        for child in gold_children[s0]:
            if state.in_buffer(child):
                left_cost = left_cost + 1

    right_cost = 1000
    if state.can_right():
        right_cost = 0
        # s0 gets s1 as head; if its real head is in the buffer, it is lost.
        if state.in_buffer(gold_heads[s0]):
            right_cost = right_cost + 1
        for child in gold_children[s0]:
            if state.in_buffer(child):
                right_cost = right_cost + 1

    return [shift_cost, left_cost, right_cost]


class Parser:

    def __init__(self, label_names, n_rows=2 ** 20):
        self.label_names = label_names
        self.n_labels = len(label_names)
        self.label_number = {}
        for k in range(self.n_labels):
            self.label_number[label_names[k]] = k
        # Classes: 0 = SHIFT, 1..L = LEFT with label, L+1..2L = RIGHT with label.
        self.model = AveragedPerceptron(1 + 2 * self.n_labels, n_rows)

    # ----- converting between class numbers and (move, label) -----

    def class_of(self, move, label_number):
        if move == SHIFT:
            return 0
        if move == LEFT:
            return 1 + label_number
        return 1 + self.n_labels + label_number

    def move_of(self, class_number):
        if class_number == 0:
            return SHIFT, None
        if class_number <= self.n_labels:
            return LEFT, self.label_names[class_number - 1]
        return RIGHT, self.label_names[class_number - 1 - self.n_labels]

    # ----- features -----

    def get_features(self, words, tags, state):
        s0 = state.stack_word(0)
        s1 = state.stack_word(1)
        s2 = state.stack_word(2)
        n0 = state.buffer_word(0)
        n1 = state.buffer_word(1)
        n2 = state.buffer_word(2)
        if n1 == 0:      # ROOT is only used as the front of the buffer
            n1 = NONE
        if n2 == 0:
            n2 = NONE

        s0_lc1 = leftmost_child(state, s0, 1)
        s0_lc2 = leftmost_child(state, s0, 2)
        s0_rc1 = rightmost_child(state, s0, 1)
        s0_rc2 = rightmost_child(state, s0, 2)
        n0_lc1 = leftmost_child(state, n0, 1)
        n0_lc2 = leftmost_child(state, n0, 2)
        s1_rc1 = rightmost_child(state, s1, 1)
        s1_lc1 = leftmost_child(state, s1, 1)

        def w(word):
            if word == NONE:
                return "<NONE>"
            return words[word]

        def t(word):
            if word == NONE:
                return "<NONE>"
            return tags[word]

        def lab(word):
            if word == NONE:
                return "<NONE>"
            return state.labels[word]

        # Words, tags and word+tag of the main positions.
        s0w = w(s0)
        s0t = t(s0)
        s0wt = s0w + "/" + s0t
        s1w = w(s1)
        s1t = t(s1)
        s1wt = s1w + "/" + s1t
        n0w = w(n0)
        n0t = t(n0)
        n0wt = n0w + "/" + n0t
        n1w = w(n1)
        n1t = t(n1)
        n2t = t(n2)
        s2t = t(s2)

        distance = distance_text(s0, n0)
        distance_s1 = distance_text(s1, s0)

        s0_n_left = "<NONE>"
        s0_n_right = "<NONE>"
        s0_left_labels = "<NONE>"
        s0_right_labels = "<NONE>"
        if s0 != NONE:
            s0_n_left = str(len(state.left_children[s0]))
            s0_n_right = str(len(state.right_children[s0]))
            s0_left_labels = label_set(state, state.left_children[s0])
            s0_right_labels = label_set(state, state.right_children[s0])
        n0_n_left = str(len(state.left_children[n0]))
        n0_left_labels = label_set(state, state.left_children[n0])
        s1_n_right = "<NONE>"
        if s1 != NONE:
            s1_n_right = str(len(state.right_children[s1]))

        f = []
        f.append("bias")
        # one position
        f.append("s0w=" + s0w)
        f.append("s0t=" + s0t)
        f.append("s0wt=" + s0wt)
        f.append("s1w=" + s1w)
        f.append("s1t=" + s1t)
        f.append("s1wt=" + s1wt)
        f.append("s2t=" + s2t)
        f.append("n0w=" + n0w)
        f.append("n0t=" + n0t)
        f.append("n0wt=" + n0wt)
        f.append("n1w=" + n1w)
        f.append("n1t=" + n1t)
        f.append("n1wt=" + n1w + "/" + n1t)
        f.append("n2w=" + w(n2))
        f.append("n2t=" + n2t)
        # pairs
        f.append("s0wt+n0wt=" + s0wt + "+" + n0wt)
        f.append("s0wt+n0w=" + s0wt + "+" + n0w)
        f.append("s0w+n0wt=" + s0w + "+" + n0wt)
        f.append("s0wt+n0t=" + s0wt + "+" + n0t)
        f.append("s0t+n0wt=" + s0t + "+" + n0wt)
        f.append("s0w+n0w=" + s0w + "+" + n0w)
        f.append("s0t+n0t=" + s0t + "+" + n0t)
        f.append("n0t+n1t=" + n0t + "+" + n1t)
        f.append("n0w+n1w=" + n0w + "+" + n1w)
        f.append("s1wt+s0wt=" + s1wt + "+" + s0wt)
        f.append("s1w+s0w=" + s1w + "+" + s0w)
        f.append("s1t+s0t=" + s1t + "+" + s0t)
        f.append("s1t+s0w=" + s1t + "+" + s0w)
        f.append("s1w+s0t=" + s1w + "+" + s0t)
        # three tags
        f.append("n0t+n1t+n2t=" + n0t + "+" + n1t + "+" + n2t)
        f.append("s0t+n0t+n1t=" + s0t + "+" + n0t + "+" + n1t)
        f.append("s1t+s0t+n0t=" + s1t + "+" + s0t + "+" + n0t)
        f.append("s2t+s1t+s0t=" + s2t + "+" + s1t + "+" + s0t)
        f.append("s1t+s0w+n0t=" + s1t + "+" + s0w + "+" + n0t)
        f.append("s0t+s0lc1t+n0t=" + s0t + "+" + t(s0_lc1) + "+" + n0t)
        f.append("s0t+s0rc1t+n0t=" + s0t + "+" + t(s0_rc1) + "+" + n0t)
        f.append("s0t+n0t+n0lc1t=" + s0t + "+" + n0t + "+" + t(n0_lc1))
        f.append("s1t+s1rc1t+s0t=" + s1t + "+" + t(s1_rc1) + "+" + s0t)
        f.append("s1t+s0t+s0rc1t=" + s1t + "+" + s0t + "+" + t(s0_rc1))
        # distances
        f.append("s0w+d=" + s0w + "+" + distance)
        f.append("s0t+d=" + s0t + "+" + distance)
        f.append("n0w+d=" + n0w + "+" + distance)
        f.append("n0t+d=" + n0t + "+" + distance)
        f.append("s0w+n0w+d=" + s0w + "+" + n0w + "+" + distance)
        f.append("s0t+n0t+d=" + s0t + "+" + n0t + "+" + distance)
        f.append("s1t+s0t+d1=" + s1t + "+" + s0t + "+" + distance_s1)
        f.append("s1w+s0w+d1=" + s1w + "+" + s0w + "+" + distance_s1)
        # number of children (valency)
        f.append("s0w+vl=" + s0w + "+" + s0_n_left)
        f.append("s0t+vl=" + s0t + "+" + s0_n_left)
        f.append("s0w+vr=" + s0w + "+" + s0_n_right)
        f.append("s0t+vr=" + s0t + "+" + s0_n_right)
        f.append("n0w+vl=" + n0w + "+" + n0_n_left)
        f.append("n0t+vl=" + n0t + "+" + n0_n_left)
        f.append("s1t+vr=" + s1t + "+" + s1_n_right)
        # children of s0, n0 and s1
        f.append("s0lc1w=" + w(s0_lc1))
        f.append("s0lc1t=" + t(s0_lc1))
        f.append("s0lc1l=" + lab(s0_lc1))
        f.append("s0lc2t=" + t(s0_lc2))
        f.append("s0lc2l=" + lab(s0_lc2))
        f.append("s0rc1w=" + w(s0_rc1))
        f.append("s0rc1t=" + t(s0_rc1))
        f.append("s0rc1l=" + lab(s0_rc1))
        f.append("s0rc2t=" + t(s0_rc2))
        f.append("s0rc2l=" + lab(s0_rc2))
        f.append("n0lc1w=" + w(n0_lc1))
        f.append("n0lc1t=" + t(n0_lc1))
        f.append("n0lc1l=" + lab(n0_lc1))
        f.append("n0lc2t=" + t(n0_lc2))
        f.append("n0lc2l=" + lab(n0_lc2))
        f.append("s1rc1t=" + t(s1_rc1))
        f.append("s1rc1l=" + lab(s1_rc1))
        f.append("s1lc1l=" + lab(s1_lc1))
        f.append("s0t+s0lc1t+s0lc2t=" + s0t + "+" + t(s0_lc1) + "+" + t(s0_lc2))
        f.append("s0t+s0rc1t+s0rc2t=" + s0t + "+" + t(s0_rc1) + "+" + t(s0_rc2))
        f.append("n0t+n0lc1t+n0lc2t=" + n0t + "+" + t(n0_lc1) + "+" + t(n0_lc2))
        f.append("s0t+s0lc1l+s0lc2l=" + s0t + "+" + lab(s0_lc1) + "+" + lab(s0_lc2))
        f.append("s0t+s0rc1l+s0rc2l=" + s0t + "+" + lab(s0_rc1) + "+" + lab(s0_rc2))
        # which labels the children already have
        f.append("s0w+left_labels=" + s0w + "+" + s0_left_labels)
        f.append("s0t+left_labels=" + s0t + "+" + s0_left_labels)
        f.append("s0w+right_labels=" + s0w + "+" + s0_right_labels)
        f.append("s0t+right_labels=" + s0t + "+" + s0_right_labels)
        f.append("n0w+left_labels=" + n0w + "+" + n0_left_labels)
        f.append("n0t+left_labels=" + n0t + "+" + n0_left_labels)
        # word endings (help with rare words)
        f.append("s0suffix=" + s0w[-3:])
        f.append("n0suffix=" + n0w[-3:])
        f.append("s0suffix+n0suffix=" + s0w[-3:] + "+" + n0w[-3:])
        # tags of the words right next to s0 and n0 in the sentence
        s0_before = NONE
        s0_after = NONE
        n0_before = NONE
        if s0 != NONE and s0 - 1 >= 1:
            s0_before = s0 - 1
        if s0 != NONE and s0 + 1 <= state.n:
            s0_after = s0 + 1
        if n0 != 0 and n0 - 1 >= 1:
            n0_before = n0 - 1
        f.append("s0t+s0+1t+n0t=" + s0t + "+" + t(s0_after) + "+" + n0t)
        f.append("s0-1t+s0t+n0t=" + t(s0_before) + "+" + s0t + "+" + n0t)
        f.append("s0t+n0-1t+n0t=" + s0t + "+" + t(n0_before) + "+" + n0t)
        f.append("s0t+s0+1t+n0-1t+n0t=" + s0t + "+" + t(s0_after) + "+" + t(n0_before) + "+" + n0t)
        return f

    # ----- choosing moves -----

    def allowed_scores(self, scores, state):
        """Give a very low score to moves that are not allowed right now."""
        scores = scores.copy()
        L = self.n_labels
        if not state.can_shift():
            scores[0] = VERY_LOW
        if not state.can_left():
            scores[1:1 + L] = VERY_LOW
        if not state.can_right():
            scores[1 + L:] = VERY_LOW
        return scores

    def gold_scores(self, scores, state, gold_heads, gold_labels, gold_children):
        """
        Keep only the scores of the "correct" classes (zero-cost moves).
        A LEFT/RIGHT move that creates a correct arc must also use the correct
        label. All other classes get a very low score.
        """
        L = self.n_labels
        costs = move_costs(state, gold_heads, gold_children)
        best_cost = min(costs)
        result = np.full(len(scores), VERY_LOW)

        if costs[SHIFT] == best_cost:
            result[0] = scores[0]

        s0 = state.stack_word(0)
        if costs[LEFT] == best_cost:
            if gold_heads[s0] == state.buffer_word(0):
                k = self.class_of(LEFT, self.label_number[gold_labels[s0]])
                result[k] = scores[k]
            else:
                result[1:1 + L] = scores[1:1 + L]

        if costs[RIGHT] == best_cost:
            if gold_heads[s0] == state.stack_word(1):
                k = self.class_of(RIGHT, self.label_number[gold_labels[s0]])
                result[k] = scores[k]
            else:
                result[1 + L:] = scores[1 + L:]
        return result

    def parse(self, words, tags, use_average=False):
        """Parse one sentence. Returns (heads, labels), position 0 unused."""
        normal_words = normalize_all(words)
        state = ParserState(len(words) - 1)
        while not state.is_finished():
            features = self.get_features(normal_words, tags, state)
            rows = self.model.feature_rows(features)
            scores = self.allowed_scores(self.model.scores(rows, use_average), state)
            move, label = self.move_of(int(scores.argmax()))
            state.apply(move, label)
        return state.heads, state.labels

    def train_one_sentence(self, words, tags, gold_heads, gold_labels):
        normal_words = normalize_all(words)
        n = len(words) - 1

        # children of every word in the correct tree (ROOT = 0)
        gold_children = []
        for i in range(n + 1):
            gold_children.append([])
        for i in range(1, n + 1):
            gold_children[gold_heads[i]].append(i)

        state = ParserState(n)
        while not state.is_finished():
            features = self.get_features(normal_words, tags, state)
            rows = self.model.feature_rows(features)
            scores = self.model.scores(rows)

            predicted = int(self.allowed_scores(scores, state).argmax())
            best_correct = int(self.gold_scores(scores, state, gold_heads, gold_labels, gold_children).argmax())
            self.model.update(rows, best_correct, predicted)

            # Follow the model's own choice (dynamic oracle / exploration).
            move, label = self.move_of(predicted)
            state.apply(move, label)
        self.model.next_step()

    def train_one_epoch(self, sentences, tags_list, random_generator):
        order = list(range(len(sentences)))
        random_generator.shuffle(order)
        for k in order:
            sentence = sentences[k]
            self.train_one_sentence(sentence["words"], tags_list[k], sentence["heads"], sentence["labels"])


def label_set(state, children):
    """The labels of a list of children, sorted and joined, e.g. "det+nsubj"."""
    labels = []
    for child in children:
        labels.append(state.labels[child])
    labels.sort()
    return "+".join(labels)


def normalize_all(words):
    result = []
    for word in words:
        result.append(normalize_word(word))
    return result


def parse_all(parser, sentences, tags_list, use_average=False):
    all_heads = []
    all_labels = []
    for k in range(len(sentences)):
        heads, labels = parser.parse(sentences[k]["words"], tags_list[k], use_average)
        all_heads.append(heads)
        all_labels.append(labels)
    return all_heads, all_labels


# ----- reading sentences backwards (right-to-left parser) -----
#
# A second kind of parser reads every sentence from the last word to the
# first one. It makes different mistakes than the normal parser, which helps
# when the parsers vote. Word i of a sentence with n words becomes word
# n + 1 - i. ROOT (0) stays 0.

def flip_position(i, n):
    if i <= 0:
        return i
    return n + 1 - i


def flip_list(values):
    """Reverse a list but keep position 0 (ROOT) in its place."""
    result = [values[0]]
    for i in range(len(values) - 1, 0, -1):
        result.append(values[i])
    return result


def flip_sentence(sentence):
    n = len(sentence["words"]) - 1
    flipped = {}
    flipped["words"] = flip_list(sentence["words"])
    if "heads" in sentence and len(sentence["heads"]) == n + 1:
        old_heads = flip_list(sentence["heads"])
        new_heads = [old_heads[0]]
        for i in range(1, n + 1):
            new_heads.append(flip_position(old_heads[i], n))
        flipped["heads"] = new_heads
        flipped["labels"] = flip_list(sentence["labels"])
    return flipped


def flip_result(heads, labels):
    """Turn the heads and labels of a backwards parse back to normal order."""
    n = len(heads) - 1
    old_heads = flip_list(heads)
    new_heads = [old_heads[0]]
    for i in range(1, n + 1):
        new_heads.append(flip_position(old_heads[i], n))
    return new_heads, flip_list(labels)
