"""
Transition-based dependency parsers (`Arc-Hybrid` and `Arc-Eager`) with:
  1. O(1) incremental maintenance of sorted children, child label sets, and
     functional child heads (`case`, `mark`, `aux`/`aux:pass`/`cop`).
  2. Full-sentence buffer lookahead (`next_clause_pos`, `np_end_pos`,
     `has_root_ahead`) so stack tokens (`cop`, `aux`, `nsubj`, `det`, `amod`)
     know where the upcoming NP head or clause predicate lies in the buffer.
  3. Hard UD structural constraints (punctuation tokens cannot govern
     non-punctuation tokens; root uniqueness).
  4. Both bottom-up `Arc-Hybrid` (`Parser`) and top-down `Arc-Eager`
     (`ArcEagerParser`, which exposes already-attached parent `s0_head` and
     `s0_label` features) trained with max-margin dynamic oracles.
"""

import random
import numpy as np

from data_utils import (
    normalize_word,
    word_shape,
    word_signature,
    base_label,
    tag_direction,
    coarse_group,
    head_offset_bucket,
    is_punct_word,
    lookup_word_priors,
)
from perceptron import AveragedPerceptron


SHIFT = 0
LEFT = 1
RIGHT = 2
REDUCE = 3
NONE = -1          # "no word here"
VERY_LOW = -1.0e9  # score given to moves that are not allowed


class SentenceContext:
    """Precomputes all token-level strings, lookahead tables, and prefix sums."""

    def __init__(self, raw_words, tags, supertags=None, lexicon=None):
        n = len(raw_words) - 1
        self.n = n
        self.raw_words = raw_words
        self.words = [normalize_word(w) for w in raw_words]
        self.tags = tags
        self.base_tags = [base_label(t) for t in tags]
        self.coarse_tags = [coarse_group(t) for t in tags]
        self.dirs = [tag_direction(t) for t in tags]
        self.shapes = [word_shape(w) for w in raw_words]
        self.sigs = [word_signature(raw_words[i], i) for i in range(n + 1)]
        self.suf3 = [w[-3:] for w in self.words]
        self.wt = [self.words[i] + "/" + tags[i] for i in range(n + 1)]
        self.wc = [self.words[i] + "/" + self.coarse_tags[i] for i in range(n + 1)]
        self.is_punct = [
            False if i == 0 else (is_punct_word(raw_words[i]) and self.coarse_tags[i] == "PUNCT")
            for i in range(n + 1)
        ]

        if lexicon is not None:
            priors = [
                lookup_word_priors(lexicon, raw_words[i], self.words[i], i)
                for i in range(n + 1)
            ]
            self.dom_tags = [p[0] for p in priors]
            self.amb_tags = [p[1] for p in priors]
            self.clusters = [p[3] for p in priors]
        else:
            self.dom_tags = ["<NONE>"] * (n + 1)
            self.amb_tags = ["<NONE>"] * (n + 1)
            self.clusters = ["<NONE>"] * (n + 1)

        if supertags is not None:
            self.supertags = supertags
            self.super_offset = [st.split(":")[0] if ":" in st else st for st in supertags]
            self.super_hcg = [st.split(":")[1] if ":" in st else "UNK" for st in supertags]
        else:
            self.supertags = ["<NONE>"] * (n + 1)
            self.super_offset = ["<NONE>"] * (n + 1)
            self.super_hcg = ["<NONE>"] * (n + 1)

        # Prefix sums for O(1) in-between structural queries
        self.verb_prefix = [0] * (n + 2)
        self.comma_prefix = [0] * (n + 2)
        self.cc_prefix = [0] * (n + 2)
        self.punct_prefix = [0] * (n + 2)

        v_cnt = 0
        cm_cnt = 0
        cc_cnt = 0
        p_cnt = 0
        for i in range(1, n + 1):
            cg = self.coarse_tags[i]
            w = self.words[i]
            if cg == "CLAUSE":
                v_cnt += 1
            if w in (",", ";", ":", "--", "-"):
                cm_cnt += 1
            if cg == "CC":
                cc_cnt += 1
            if cg == "PUNCT":
                p_cnt += 1
            self.verb_prefix[i] = v_cnt
            self.comma_prefix[i] = cm_cnt
            self.cc_prefix[i] = cc_cnt
            self.punct_prefix[i] = p_cnt
        self.verb_prefix[n + 1] = v_cnt
        self.comma_prefix[n + 1] = cm_cnt
        self.cc_prefix[n + 1] = cc_cnt
        self.punct_prefix[n + 1] = p_cnt

        # Buffer lookahead arrays:
        # 1. `next_clause_pos[i]`: first token >= i with coarse tag CLAUSE or tag 'root'
        # 2. `has_root_ahead[i]`: whether a token tagged 'root' lies at >= i
        # 3. `np_end_pos[i]`: end of the contiguous nominal/modifier chunk starting at i
        self.next_clause_pos = [0] * (n + 2)
        self.has_root_ahead = ["False"] * (n + 2)
        next_cl = 0
        seen_root = False
        for i in range(n, 0, -1):
            if self.tags[i] == "root":
                seen_root = True
            if self.coarse_tags[i] == "CLAUSE" or self.tags[i] == "root":
                next_cl = i
            self.next_clause_pos[i] = next_cl
            self.has_root_ahead[i] = "True" if seen_root else "False"

        self.np_end_pos = [0] * (n + 2)
        for i in range(1, n + 1):
            if self.coarse_tags[i] in ("DET", "MOD", "COMP", "SUBJ", "OBJ", "NMOD"):
                j = i
                while j < n and self.coarse_tags[j] in ("DET", "MOD", "COMP") and self.dirs[j] == ">":
                    if self.coarse_tags[j + 1] in ("MOD", "COMP", "SUBJ", "OBJ", "NMOD", "CLAUSE"):
                        j += 1
                    else:
                        break
                self.np_end_pos[i] = j
            else:
                self.np_end_pos[i] = i

    def between_counts(self, a, b):
        """Return binned (verbs, commas, conjunctions) strictly between a and b."""
        if a <= 0 or b <= 0:
            return "0_0_0"
        lo = a if a < b else b
        hi = b if a < b else a
        if hi - lo <= 1:
            return "adj"
        nv = min(self.verb_prefix[hi - 1] - self.verb_prefix[lo], 3)
        nc = min(self.comma_prefix[hi - 1] - self.comma_prefix[lo], 3)
        ncc = min(self.cc_prefix[hi - 1] - self.cc_prefix[lo], 2)
        return f"{nv}_{nc}_{ncc}"


class ParserState:
    """Arc-Hybrid state maintained in O(1)."""

    def __init__(self, n_words, words=None, is_punct=None):
        self.n = n_words
        self.words = words
        self.is_punct = is_punct
        self.stack = []
        self.buffer_position = 1
        size = n_words + 1
        self.heads = [NONE] * size
        self.labels = ["<NONE>"] * size
        self.left_children = [[] for _ in range(size)]
        self.right_children = [[] for _ in range(size)]
        self.left_label_str = ["<EMPTY>"] * size
        self.right_label_str = ["<EMPTY>"] * size

        self.case_word = ["<NONE>"] * size
        self.mark_word = ["<NONE>"] * size
        self.aux_info = ["<NONE>"] * size
        self.has_subj = [False] * size

    def buffer_word(self, k):
        position = self.buffer_position + k
        if position <= self.n:
            return position
        if position == self.n + 1:
            return 0
        return NONE

    def stack_word(self, k):
        idx = len(self.stack) - 1 - k
        if idx >= 0:
            return self.stack[idx]
        return NONE

    def is_finished(self):
        return len(self.stack) == 0 and self.buffer_position > self.n

    def in_buffer(self, word):
        return word == 0 or word >= self.buffer_position

    def can_shift(self):
        return self.buffer_position <= self.n

    def can_left(self):
        if not self.stack:
            return False
        b0 = self.buffer_word(0)
        if b0 == 0:
            return len(self.stack) == 1
        # Prevent pure punctuation in the buffer from becoming the head of a word
        if self.is_punct is not None and self.is_punct[b0]:
            s0 = self.stack[-1]
            if not self.is_punct[s0] and self.can_shift():
                return False
        return True

    def can_right(self):
        if len(self.stack) < 2:
            return False
        # Prevent pure punctuation on the stack from becoming the head of a word
        # only when SHIFT is still available; once the buffer is empty, RIGHT
        # is required to reduce the stack down to 1 item.
        if self.is_punct is not None and self.can_shift():
            s1 = self.stack[-2]
            s0 = self.stack[-1]
            if self.is_punct[s1] and not self.is_punct[s0]:
                return False
        return True

    def add_arc(self, head, dependent, label):
        self.heads[dependent] = head
        self.labels[dependent] = label
        b_lab = label.split(":")[0]

        if self.words is not None:
            dep_w = self.words[dependent]
            if b_lab == "case":
                self.case_word[head] = dep_w
            elif b_lab == "mark":
                self.mark_word[head] = dep_w
            elif b_lab in ("aux", "cop"):
                self.aux_info[head] = label + ":" + dep_w
            elif b_lab in ("nsubj", "csubj"):
                self.has_subj[head] = True

        if dependent < head:
            lc = self.left_children[head]
            lc.append(dependent)
            if len(lc) > 1 and lc[-1] < lc[-2]:
                lc.sort()
            labs = sorted(set(self.labels[c] for c in lc))
            self.left_label_str[head] = "+".join(labs)
        else:
            rc = self.right_children[head]
            rc.append(dependent)
            if len(rc) > 1 and rc[-1] < rc[-2]:
                rc.sort()
            labs = sorted(set(self.labels[c] for c in rc))
            self.right_label_str[head] = "+".join(labs)

    def apply(self, move, label):
        if move == SHIFT:
            self.stack.append(self.buffer_position)
            self.buffer_position += 1
        elif move == LEFT:
            dependent = self.stack.pop()
            self.add_arc(self.buffer_word(0), dependent, label)
        else:
            dependent = self.stack.pop()
            self.add_arc(self.stack[-1], dependent, label)


def leftmost_child(state, word, k):
    if word == NONE:
        return NONE
    children = state.left_children[word]
    if len(children) >= k:
        return children[k - 1]
    return NONE


def rightmost_child(state, word, k):
    if word == NONE:
        return NONE
    children = state.right_children[word]
    if len(children) >= k:
        return children[len(children) - k]
    return NONE


def distance_text(a, b):
    if a == NONE or b == NONE:
        return "none"
    if a == 0 or b == 0:
        return "root"
    d = abs(a - b)
    if d <= 5:
        return str(d)
    if d <= 10:
        return "6-10"
    return "11+"


def move_costs(state, gold_heads, gold_children):
    """Dynamic oracle for Arc-Hybrid."""
    s0 = state.stack_word(0)
    s1 = state.stack_word(1)
    b0 = state.buffer_word(0)
    stack_set = set(state.stack)

    shift_cost = 1000
    if state.can_shift():
        shift_cost = 0
        head = gold_heads[b0]
        if head != s0 and head in stack_set:
            shift_cost += 1
        for child in gold_children[b0]:
            if child in stack_set:
                shift_cost += 1

    left_cost = 1000
    if state.can_left():
        left_cost = 0
        head = gold_heads[s0]
        if head != b0 and (head == s1 or state.in_buffer(head)):
            left_cost += 1
        for child in gold_children[s0]:
            if state.in_buffer(child):
                left_cost += 1

    right_cost = 1000
    if state.can_right():
        right_cost = 0
        if state.in_buffer(gold_heads[s0]):
            right_cost += 1
        for child in gold_children[s0]:
            if state.in_buffer(child):
                right_cost += 1

    return [shift_cost, left_cost, right_cost]


def _extract_shared_features(ctx, state):
    """Shared feature extractor for both Arc-Hybrid and Arc-Eager states."""
    words = ctx.words
    tags = ctx.tags
    base_tags = ctx.base_tags
    coarse_tags = ctx.coarse_tags
    dirs = ctx.dirs
    wt_arr = ctx.wt
    wc_arr = ctx.wc
    suf3 = ctx.suf3
    sigs = ctx.sigs
    dom_tags = ctx.dom_tags
    amb_tags = ctx.amb_tags
    clusters = ctx.clusters
    supertags = ctx.supertags
    super_offset = ctx.super_offset
    super_hcg = ctx.super_hcg

    s0 = state.stack_word(0)
    s1 = state.stack_word(1)
    s2 = state.stack_word(2)
    s3 = state.stack_word(3)
    n0 = state.buffer_word(0)
    n1 = state.buffer_word(1)
    n2 = state.buffer_word(2)
    if n1 == 0:
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
    s1_rc2 = rightmost_child(state, s1, 2)
    s1_lc1 = leftmost_child(state, s1, 1)
    s1_lc2 = leftmost_child(state, s1, 2)

    s0_lc1_lc1 = leftmost_child(state, s0_lc1, 1)
    n0_lc1_lc1 = leftmost_child(state, n0_lc1, 1)
    s1_rc1_rc1 = rightmost_child(state, s1_rc1, 1)

    def w(idx):
        return "<NONE>" if idx == NONE else words[idx]

    def t(idx):
        return "<NONE>" if idx == NONE else tags[idx]

    def bt(idx):
        return "<NONE>" if idx == NONE else base_tags[idx]

    def cg(idx):
        return "<NONE>" if idx == NONE else coarse_tags[idx]

    def lab(idx):
        return "<NONE>" if idx == NONE else state.labels[idx]

    s0w = w(s0)
    s0t = t(s0)
    s0bt = bt(s0)
    s0cg = cg(s0)
    s0wt = "<NONE>" if s0 == NONE else wt_arr[s0]
    s0wc = "<NONE>" if s0 == NONE else wc_arr[s0]
    s0d = "<NONE>" if s0 == NONE else dirs[s0]
    s0sig = "<NONE>" if s0 == NONE else sigs[s0]
    s0amb = "<NONE>" if s0 == NONE else amb_tags[s0]
    s0clu = "<NONE>" if s0 == NONE else clusters[s0]

    s1w = w(s1)
    s1t = t(s1)
    s1bt = bt(s1)
    s1cg = cg(s1)
    s1wt = "<NONE>" if s1 == NONE else wt_arr[s1]
    s1wc = "<NONE>" if s1 == NONE else wc_arr[s1]
    s1d = "<NONE>" if s1 == NONE else dirs[s1]
    s1amb = "<NONE>" if s1 == NONE else amb_tags[s1]

    s2w = w(s2)
    s2t = t(s2)
    s2cg = cg(s2)
    s3t = t(s3)

    n0w = w(n0)
    n0t = t(n0)
    n0bt = bt(n0)
    n0cg = cg(n0)
    n0wt = "<NONE>" if n0 == NONE else wt_arr[n0]
    n0wc = "<NONE>" if n0 == NONE else wc_arr[n0]
    n0d = "<NONE>" if n0 == NONE else dirs[n0]
    n0sig = "<NONE>" if n0 == NONE else sigs[n0]
    n0amb = "<NONE>" if n0 == NONE else amb_tags[n0]
    n0clu = "<NONE>" if n0 == NONE else clusters[n0]

    n1w = w(n1)
    n1t = t(n1)
    n1cg = cg(n1)
    n1wt = "<NONE>" if n1 == NONE else wt_arr[n1]
    n2w = w(n2)
    n2t = t(n2)
    n2cg = cg(n2)

    distance = distance_text(s0, n0)
    distance_s1 = distance_text(s1, s0)
    between_s0_n0 = ctx.between_counts(s0, n0) if (s0 != NONE and n0 != NONE) else "none"
    between_s1_s0 = ctx.between_counts(s1, s0) if (s1 != NONE and s0 != NONE) else "none"

    if s0 != NONE:
        s0_n_left = str(min(len(state.left_children[s0]), 4))
        s0_n_right = str(min(len(state.right_children[s0]), 4))
        s0_left_labels = state.left_label_str[s0]
        s0_right_labels = state.right_label_str[s0]
        s0_case = state.case_word[s0]
        s0_mark = state.mark_word[s0]
        s0_aux = state.aux_info[s0]
        s0_subj = str(state.has_subj[s0])
    else:
        s0_n_left = s0_n_right = s0_left_labels = s0_right_labels = "<NONE>"
        s0_case = s0_mark = s0_aux = s0_subj = "<NONE>"

    if s1 != NONE:
        s1_n_left = str(min(len(state.left_children[s1]), 4))
        s1_n_right = str(min(len(state.right_children[s1]), 4))
        s1_left_labels = state.left_label_str[s1]
        s1_right_labels = state.right_label_str[s1]
        s1_case = state.case_word[s1]
        s1_mark = state.mark_word[s1]
        s1_aux = state.aux_info[s1]
        s1_subj = str(state.has_subj[s1])
    else:
        s1_n_left = s1_n_right = s1_left_labels = s1_right_labels = "<NONE>"
        s1_case = s1_mark = s1_aux = s1_subj = "<NONE>"

    if n0 != NONE:
        n0_n_left = str(min(len(state.left_children[n0]), 4))
        n0_left_labels = state.left_label_str[n0]
        n0_case = state.case_word[n0]
        n0_mark = state.mark_word[n0]
        n0_aux = state.aux_info[n0]
        n0_subj = str(state.has_subj[n0])
    else:
        n0_n_left = n0_left_labels = n0_case = n0_mark = n0_aux = n0_subj = "<NONE>"

    s0_before = s0 - 1 if (s0 > 1) else NONE
    s0_after = s0 + 1 if (0 < s0 < state.n) else NONE
    n0_before = n0 - 1 if (n0 > 1) else NONE
    s1_before = s1 - 1 if (s1 > 1) else NONE
    s1_after = s1 + 1 if (0 < s1 < state.n) else NONE

    # Supertag offset agreement checks
    s0_st = "<NONE>" if s0 == NONE else supertags[s0]
    n0_st = "<NONE>" if n0 == NONE else supertags[n0]
    s1_st = "<NONE>" if s1 == NONE else supertags[s1]
    if s0 != NONE and s0 > 0:
        s0_pred_off = super_offset[s0]
        s0_pred_hcg = super_hcg[s0]
        cand_left_off = head_offset_bucket(s0, n0) if n0 != NONE else "NONE"
        cand_right_off = head_offset_bucket(s0, s1) if s1 != NONE else "NONE"
        left_off_match = str(s0_pred_off == cand_left_off) + ":" + str(s0_pred_hcg == n0cg)
        right_off_match = str(s0_pred_off == cand_right_off) + ":" + str(s0_pred_hcg == s1cg)
    else:
        s0_pred_off = "<NONE>"
        left_off_match = "NONE"
        right_off_match = "NONE"

    # Buffer lookahead features (where is the upcoming NP head and clause predicate?)
    b_pos = state.buffer_position
    if b_pos <= state.n:
        next_cl = ctx.next_clause_pos[b_pos]
        next_cl_t = tags[next_cl] if next_cl > 0 else "<NONE>"
        next_cl_dist = distance_text(b_pos, next_cl) if next_cl > 0 else "none"
        root_ahead = ctx.has_root_ahead[b_pos]
        np_end = ctx.np_end_pos[b_pos]
        np_end_t = tags[np_end] if np_end > 0 else "<NONE>"
        np_end_dist = str(min(np_end - b_pos, 4))
    else:
        next_cl_t = "<NONE>"
        next_cl_dist = "none"
        root_ahead = "False"
        np_end_t = "<NONE>"
        np_end_dist = "0"

    f = [
        "bias",
        "s0w=" + s0w,
        "s0t=" + s0t,
        "s0bt=" + s0bt,
        "s0cg=" + s0cg,
        "s0wt=" + s0wt,
        "s0wc=" + s0wc,
        "s0sig=" + s0sig,
        "s0amb=" + s0amb,
        "s0clu=" + s0clu,
        "s1w=" + s1w,
        "s1t=" + s1t,
        "s1bt=" + s1bt,
        "s1cg=" + s1cg,
        "s1wt=" + s1wt,
        "s1wc=" + s1wc,
        "s1amb=" + s1amb,
        "s2w=" + s2w,
        "s2t=" + s2t,
        "s2cg=" + s2cg,
        "s3t=" + s3t,
        "n0w=" + n0w,
        "n0t=" + n0t,
        "n0bt=" + n0bt,
        "n0cg=" + n0cg,
        "n0wt=" + n0wt,
        "n0wc=" + n0wc,
        "n0sig=" + n0sig,
        "n0amb=" + n0amb,
        "n0clu=" + n0clu,
        "n1w=" + n1w,
        "n1t=" + n1t,
        "n1cg=" + n1cg,
        "n1wt=" + n1wt,
        "n2w=" + n2w,
        "n2t=" + n2t,
        "n2cg=" + n2cg,

        # Pair features
        "s0wt+n0wt=" + s0wt + "+" + n0wt,
        "s0wt+n0w=" + s0wt + "+" + n0w,
        "s0w+n0wt=" + s0w + "+" + n0wt,
        "s0wt+n0t=" + s0wt + "+" + n0t,
        "s0t+n0wt=" + s0t + "+" + n0wt,
        "s0w+n0w=" + s0w + "+" + n0w,
        "s0t+n0t=" + s0t + "+" + n0t,
        "s0bt+n0bt=" + s0bt + "+" + n0bt,
        "s0cg+n0cg=" + s0cg + "+" + n0cg,
        "s0wc+n0wc=" + s0wc + "+" + n0wc,
        "s0clu+n0clu=" + s0clu + "+" + n0clu,
        "n0t+n1t=" + n0t + "+" + n1t,
        "n0w+n1w=" + n0w + "+" + n1w,
        "n0wt+n1t=" + n0wt + "+" + n1t,
        "n0t+n1wt=" + n0t + "+" + n1wt,
        "s1wt+s0wt=" + s1wt + "+" + s0wt,
        "s1wt+s0w=" + s1wt + "+" + s0w,
        "s1w+s0wt=" + s1w + "+" + s0wt,
        "s1wt+s0t=" + s1wt + "+" + s0t,
        "s1t+s0wt=" + s1t + "+" + s0wt,
        "s1w+s0w=" + s1w + "+" + s0w,
        "s1t+s0t=" + s1t + "+" + s0t,
        "s1bt+s0bt=" + s1bt + "+" + s0bt,
        "s1cg+s0cg=" + s1cg + "+" + s0cg,
        "s1wc+s0wc=" + s1wc + "+" + s0wc,
        "s1t+s0w=" + s1t + "+" + s0w,
        "s1w+s0t=" + s1w + "+" + s0t,
        "s1t+n0t=" + s1t + "+" + n0t,
        "s1w+n0w=" + s1w + "+" + n0w,
        "s1wt+n0t=" + s1wt + "+" + n0t,
        "s1t+n0wt=" + s1t + "+" + n0wt,
        "s2t+s1t=" + s2t + "+" + s1t,

        # Trigram & 4-gram structural context
        "n0t+n1t+n2t=" + n0t + "+" + n1t + "+" + n2t,
        "s0t+n0t+n1t=" + s0t + "+" + n0t + "+" + n1t,
        "s1t+s0t+n0t=" + s1t + "+" + s0t + "+" + n0t,
        "s2t+s1t+s0t=" + s2t + "+" + s1t + "+" + s0t,
        "s3t+s2t+s1t+s0t=" + s3t + "+" + s2t + "+" + s1t + "+" + s0t,
        "s2t+s1t+s0t+n0t=" + s2t + "+" + s1t + "+" + s0t + "+" + n0t,
        "s1t+s0t+n0t+n1t=" + s1t + "+" + s0t + "+" + n0t + "+" + n1t,
        "s1cg+s0cg+n0cg=" + s1cg + "+" + s0cg + "+" + n0cg,
        "s2cg+s1cg+s0cg=" + s2cg + "+" + s1cg + "+" + s0cg,
        "s0cg+n0cg+n1cg=" + s0cg + "+" + n0cg + "+" + n1cg,
        "s1t+s0w+n0t=" + s1t + "+" + s0w + "+" + n0t,
        "s1w+s0t+n0t=" + s1w + "+" + s0t + "+" + n0t,
        "s1t+s0t+n0w=" + s1t + "+" + s0t + "+" + n0w,

        # Child-Combined Trigrams
        "s0t+s0lc1t+n0t=" + s0t + "+" + t(s0_lc1) + "+" + n0t,
        "s0t+s0rc1t+n0t=" + s0t + "+" + t(s0_rc1) + "+" + n0t,
        "s0t+n0t+n0lc1t=" + s0t + "+" + n0t + "+" + t(n0_lc1),
        "s1t+s1rc1t+s0t=" + s1t + "+" + t(s1_rc1) + "+" + s0t,
        "s1t+s1lc1t+s0t=" + s1t + "+" + t(s1_lc1) + "+" + s0t,
        "s1t+s0t+s0rc1t=" + s1t + "+" + s0t + "+" + t(s0_rc1),
        "s1t+s0t+s0lc1t=" + s1t + "+" + s0t + "+" + t(s0_lc1),
        "s1t+s1rc1l+s0t=" + s1t + "+" + lab(s1_rc1) + "+" + s0t,
        "s1t+s1lc1l+s0t=" + s1t + "+" + lab(s1_lc1) + "+" + s0t,
        "s0t+n0t+n0lc1l=" + s0t + "+" + n0t + "+" + lab(n0_lc1),

        # Distances & In-Between Structural Counts
        "s0w+d=" + s0w + "+" + distance,
        "s0t+d=" + s0t + "+" + distance,
        "n0w+d=" + n0w + "+" + distance,
        "n0t+d=" + n0t + "+" + distance,
        "s0w+n0w+d=" + s0w + "+" + n0w + "+" + distance,
        "s0t+n0t+d=" + s0t + "+" + n0t + "+" + distance,
        "s1t+s0t+d1=" + s1t + "+" + s0t + "+" + distance_s1,
        "s1w+s0w+d1=" + s1w + "+" + s0w + "+" + distance_s1,
        "s1t+s0w+d1=" + s1t + "+" + s0w + "+" + distance_s1,
        "s1w+s0t+d1=" + s1w + "+" + s0t + "+" + distance_s1,
        "btw_s0n0=" + s0cg + "+" + n0cg + "+" + between_s0_n0,
        "btw_s1s0=" + s1cg + "+" + s0cg + "+" + between_s1_s0,
        "btw_s1s0_t=" + s1t + "+" + s0t + "+" + between_s1_s0,

        # Valency
        "s0w+vl=" + s0w + "+" + s0_n_left,
        "s0t+vl=" + s0t + "+" + s0_n_left,
        "s0w+vr=" + s0w + "+" + s0_n_right,
        "s0t+vr=" + s0t + "+" + s0_n_right,
        "n0w+vl=" + n0w + "+" + n0_n_left,
        "n0t+vl=" + n0t + "+" + n0_n_left,
        "s1w+vl=" + s1w + "+" + s1_n_left,
        "s1t+vl=" + s1t + "+" + s1_n_left,
        "s1w+vr=" + s1w + "+" + s1_n_right,
        "s1t+vr=" + s1t + "+" + s1_n_right,

        # Children & Grandchildren
        "s0lc1w=" + w(s0_lc1),
        "s0lc1t=" + t(s0_lc1),
        "s0lc1l=" + lab(s0_lc1),
        "s0lc2t=" + t(s0_lc2),
        "s0lc2l=" + lab(s0_lc2),
        "s0rc1w=" + w(s0_rc1),
        "s0rc1t=" + t(s0_rc1),
        "s0rc1l=" + lab(s0_rc1),
        "s0rc2t=" + t(s0_rc2),
        "s0rc2l=" + lab(s0_rc2),
        "n0lc1w=" + w(n0_lc1),
        "n0lc1t=" + t(n0_lc1),
        "n0lc1l=" + lab(n0_lc1),
        "n0lc2t=" + t(n0_lc2),
        "n0lc2l=" + lab(n0_lc2),
        "s1rc1w=" + w(s1_rc1),
        "s1rc1t=" + t(s1_rc1),
        "s1rc1l=" + lab(s1_rc1),
        "s1rc2t=" + t(s1_rc2),
        "s1rc2l=" + lab(s1_rc2),
        "s1lc1w=" + w(s1_lc1),
        "s1lc1t=" + t(s1_lc1),
        "s1lc1l=" + lab(s1_lc1),
        "s1lc2l=" + lab(s1_lc2),
        "s0gc_l=" + lab(s0_lc1_lc1) + "/" + t(s0_lc1_lc1),
        "n0gc_l=" + lab(n0_lc1_lc1) + "/" + t(n0_lc1_lc1),
        "s1gc_r=" + lab(s1_rc1_rc1) + "/" + t(s1_rc1_rc1),

        # Sibling & 2nd-order combinations
        "s0t+s0lc1t+s0lc2t=" + s0t + "+" + t(s0_lc1) + "+" + t(s0_lc2),
        "s0t+s0rc1t+s0rc2t=" + s0t + "+" + t(s0_rc1) + "+" + t(s0_rc2),
        "n0t+n0lc1t+n0lc2t=" + n0t + "+" + t(n0_lc1) + "+" + t(n0_lc2),
        "s1t+s1rc1t+s1rc2t=" + s1t + "+" + t(s1_rc1) + "+" + t(s1_rc2),
        "s1t+s1lc1t+s1lc2t=" + s1t + "+" + t(s1_lc1) + "+" + t(s1_lc2),
        "s0t+s0lc1l+s0lc2l=" + s0t + "+" + lab(s0_lc1) + "+" + lab(s0_lc2),
        "s0t+s0rc1l+s0rc2l=" + s0t + "+" + lab(s0_rc1) + "+" + lab(s0_rc2),
        "s1t+s1rc1l+s1rc2l=" + s1t + "+" + lab(s1_rc1) + "+" + lab(s1_rc2),

        # Child label sets
        "s0w+left_labels=" + s0w + "+" + s0_left_labels,
        "s0t+left_labels=" + s0t + "+" + s0_left_labels,
        "s0w+right_labels=" + s0w + "+" + s0_right_labels,
        "s0t+right_labels=" + s0t + "+" + s0_right_labels,
        "n0w+left_labels=" + n0w + "+" + n0_left_labels,
        "n0t+left_labels=" + n0t + "+" + n0_left_labels,
        "s1w+left_labels=" + s1w + "+" + s1_left_labels,
        "s1t+left_labels=" + s1t + "+" + s1_left_labels,
        "s1w+right_labels=" + s1w + "+" + s1_right_labels,
        "s1t+right_labels=" + s1t + "+" + s1_right_labels,
        "s1_subj+s0t=" + s1_subj + "+" + s1t + "+" + s0t,
        "n0_subj+s0t=" + n0_subj + "+" + n0t + "+" + s0t,

        # Lexicalized Preposition (case), Subordinator (mark), and Passive Aux
        "s0_case=" + s0_case + "+" + s0t,
        "s1w+s0_case=" + s1w + "+" + s0_case + "+" + s0cg,
        "s1t+s0_case=" + s1t + "+" + s0_case + "+" + s0t,
        "s1t+s0_case+s0w=" + s1t + "+" + s0_case + "+" + s0w,
        "n0w+s0_case=" + n0w + "+" + s0_case + "+" + s0cg,
        "n0t+s0_case=" + n0t + "+" + s0_case + "+" + s0t,
        "s0w+n0_case=" + s0w + "+" + n0_case + "+" + n0cg,
        "s1_aux+s0_case=" + s1_aux + "+" + s0_case + "+" + s0cg,
        "n0_aux+s0t=" + n0_aux + "+" + s0t,
        "s1_aux+s0t=" + s1_aux + "+" + s0t,
        "s1t+s0_mark=" + s1t + "+" + s0_mark + "+" + s0t,
        "s1w+s0_mark=" + s1w + "+" + s0_mark + "+" + s0cg,
        "n0t+s0_mark=" + n0t + "+" + s0_mark + "+" + s0t,

        # Word endings
        "s0suffix=" + ("<NONE>" if s0 == NONE else suf3[s0]),
        "n0suffix=" + ("<NONE>" if n0 == NONE else suf3[n0]),
        "s1suffix=" + ("<NONE>" if s1 == NONE else suf3[s1]),
        "s0suf+n0suf=" + ("<NONE>" if s0 == NONE else suf3[s0]) + "+" + ("<NONE>" if n0 == NONE else suf3[n0]),
        "s1suf+s0suf=" + ("<NONE>" if s1 == NONE else suf3[s1]) + "+" + ("<NONE>" if s0 == NONE else suf3[s0]),

        # Surface sentence neighbors
        "s0t+s0+1t+n0t=" + s0t + "+" + t(s0_after) + "+" + n0t,
        "s0-1t+s0t+n0t=" + t(s0_before) + "+" + s0t + "+" + n0t,
        "s0t+n0-1t+n0t=" + s0t + "+" + t(n0_before) + "+" + n0t,
        "s0t+s0+1t+n0-1t+n0t=" + s0t + "+" + t(s0_after) + "+" + t(n0_before) + "+" + n0t,
        "s1-1t+s1t+s0t=" + t(s1_before) + "+" + s1t + "+" + s0t,
        "s1t+s1+1t+s0t=" + s1t + "+" + t(s1_after) + "+" + s0t,
        "s1t+s0t+s0+1t=" + s1t + "+" + s0t + "+" + t(s0_after),
        "s0-1w+s0w=" + w(s0_before) + "+" + s0w,
        "s0w+s0+1w=" + s0w + "+" + w(s0_after),
        "n0-1w+n0w=" + w(n0_before) + "+" + n0w,

        # Direction, Supertag Agreement & Buffer Lookahead
        "dirs=" + s1d + "+" + s0d + "+" + n0d,
        "s0st=" + s0_st,
        "n0st=" + n0_st,
        "s1st=" + s1_st,
        "s0st+n0cg=" + s0_st + "+" + n0cg + "+" + distance,
        "s0st+s1cg=" + s0_st + "+" + s1cg + "+" + distance_s1,
        "left_off_match=" + left_off_match + "+" + s0bt,
        "right_off_match=" + right_off_match + "+" + s0bt,
        "s0off+d=" + s0_pred_off + "+" + distance + "+" + distance_s1,
        "look_root=" + root_ahead + "+" + s0t + "+" + n0t,
        "look_cl=" + s1t + "+" + s0t + "+" + next_cl_t + "+" + next_cl_dist,
        "look_np=" + s0t + "+" + n0t + "+" + np_end_t + "+" + np_end_dist,
    ]
    return f


class Parser:
    """Arc-Hybrid Transition Parser (bottom-up right attachments)."""

    def __init__(self, label_names, lexicon=None, n_rows=2 ** 21):
        self.label_names = label_names
        self.n_labels = len(label_names)
        self.label_number = {name: k for k, name in enumerate(label_names)}
        self.lexicon = lexicon
        self.model = AveragedPerceptron(1 + 2 * self.n_labels, n_rows)

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

    def get_features(self, ctx, state):
        return _extract_shared_features(ctx, state)

    def allowed_scores(self, scores, state):
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
        L = self.n_labels
        costs = move_costs(state, gold_heads, gold_children)
        best_cost = min(costs)
        result = np.full(len(scores), VERY_LOW, dtype=np.float32)

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

    def parse(self, words, tags, use_average=False, supertags=None):
        ctx = SentenceContext(words, tags, supertags, self.lexicon)
        state = ParserState(len(words) - 1, ctx.words, ctx.is_punct)
        while not state.is_finished():
            features = self.get_features(ctx, state)
            rows = self.model.feature_rows(features)
            scores = self.allowed_scores(self.model.scores(rows, use_average), state)
            move, label = self.move_of(int(scores.argmax()))
            state.apply(move, label)
        return state.heads, state.labels

    def train_one_sentence(
        self,
        words,
        tags,
        gold_heads,
        gold_labels,
        supertags=None,
        explore_prob=0.90,
        rng=None,
    ):
        ctx = SentenceContext(words, tags, supertags, self.lexicon)
        n = len(words) - 1

        gold_children = [[] for _ in range(n + 1)]
        for i in range(1, n + 1):
            gold_children[gold_heads[i]].append(i)

        state = ParserState(n, ctx.words, ctx.is_punct)
        while not state.is_finished():
            features = self.get_features(ctx, state)
            rows = self.model.feature_rows(features)
            scores = self.model.scores(rows)

            allowed = self.allowed_scores(scores, state)
            gold = self.gold_scores(scores, state, gold_heads, gold_labels, gold_children)
            best_correct = int(gold.argmax())

            augmented = allowed.copy()
            non_gold_mask = gold <= (VERY_LOW * 0.5)
            augmented[non_gold_mask] += 1.0
            predicted = int(augmented.argmax())

            if predicted != best_correct:
                self.model.update(rows, best_correct, predicted)

            model_choice = int(allowed.argmax())
            if rng is not None and rng.random() > explore_prob:
                chosen = best_correct
            else:
                chosen = model_choice
            move, label = self.move_of(chosen)
            state.apply(move, label)
        self.model.next_step()

    def train_one_epoch(self, sentences, tags_list, random_generator, supertags_list=None, epoch=0):
        order = list(range(len(sentences)))
        random_generator.shuffle(order)
        explore_prob = 0.50 if epoch == 0 else 0.90
        for k in order:
            sentence = sentences[k]
            stags = supertags_list[k] if supertags_list is not None else None
            self.train_one_sentence(
                sentence["words"],
                tags_list[k],
                sentence["heads"],
                sentence["labels"],
                stags,
                explore_prob=explore_prob,
                rng=random_generator,
            )


class ArcEagerState(ParserState):
    """
    Arc-Eager state (Nivre 2003; Goldberg & Nivre 2012).
    ROOT (0) starts at the bottom of the stack; buffer runs 1..n.
    Moves:
      SHIFT        : push b0 onto stack
      REDUCE       : pop s0 (when s0 already has a head)
      LEFT(label)  : attach b0 -> s0 (when s0 has no head), pop s0
      RIGHT(label) : attach s0 -> b0, push b0 onto stack
    """

    def __init__(self, n_words, words=None, is_punct=None):
        super().__init__(n_words, words, is_punct)
        self.stack = [0]

    def buffer_word(self, k):
        position = self.buffer_position + k
        if position <= self.n:
            return position
        return NONE

    def is_finished(self):
        return self.buffer_position > self.n and len(self.stack) <= 1

    def in_buffer(self, word):
        return word >= self.buffer_position

    def can_shift(self):
        return self.buffer_position <= self.n

    def can_reduce(self):
        if len(self.stack) <= 1:
            return False
        s0 = self.stack[-1]
        if self.heads[s0] != NONE:
            return True
        # When the buffer is exhausted, allow popping any remaining stack items
        return self.buffer_position > self.n

    def can_left(self):
        if len(self.stack) <= 1 or self.buffer_position > self.n:
            return False
        s0 = self.stack[-1]
        if s0 == 0 or self.heads[s0] != NONE:
            return False
        b0 = self.buffer_position
        if self.is_punct is not None and self.is_punct[b0] and not self.is_punct[s0]:
            if self.can_shift():
                return False
        return True

    def can_right(self):
        if not self.stack:
            return False
        s0 = self.stack[-1]
        if self.buffer_position <= self.n:
            b0 = self.buffer_position
            if s0 == 0 and len(self.right_children[0]) >= 1 and self.buffer_position < self.n:
                return False
            if self.is_punct is not None and self.is_punct[s0] and not self.is_punct[b0]:
                if self.can_shift() or self.can_reduce():
                    return False
            return True
        return False

    def apply(self, move, label):
        if move == SHIFT:
            self.stack.append(self.buffer_position)
            self.buffer_position += 1
        elif move == REDUCE:
            s0 = self.stack.pop()
            if self.heads[s0] == NONE:
                # Attach unattached reduced node to stack top
                head = self.stack[-1] if self.stack else 0
                self.add_arc(head, s0, "dep" if head != 0 else "root")
        elif move == LEFT:
            dependent = self.stack.pop()
            self.add_arc(self.buffer_position, dependent, label)
        else:
            dependent = self.buffer_position
            self.buffer_position += 1
            self.add_arc(self.stack[-1], dependent, label)
            self.stack.append(dependent)


def eager_move_costs(state, gold_heads, gold_children):
    """Exact dynamic oracle for Arc-Eager (Goldberg & Nivre 2012)."""
    s0 = state.stack_word(0)
    b0 = state.buffer_word(0)
    stack_set = set(state.stack)

    shift_cost = 1000
    if state.can_shift():
        shift_cost = 0
        if gold_heads[b0] in stack_set:
            shift_cost += 1
        for c in gold_children[b0]:
            if c in stack_set and state.heads[c] == NONE:
                shift_cost += 1

    reduce_cost = 1000
    if state.can_reduce():
        reduce_cost = 0
        for c in gold_children[s0]:
            if state.in_buffer(c):
                reduce_cost += 1

    left_cost = 1000
    if state.can_left():
        left_cost = 0
        if gold_heads[s0] != b0 and state.in_buffer(gold_heads[s0]):
            left_cost += 1
        for c in gold_children[s0]:
            if state.in_buffer(c):
                left_cost += 1

    right_cost = 1000
    if state.can_right():
        right_cost = 0
        if gold_heads[b0] != s0 and (
            state.in_buffer(gold_heads[b0])
            or (gold_heads[b0] in stack_set and state.heads[b0] == NONE)
        ):
            right_cost += 1
        for c in gold_children[b0]:
            if c in stack_set and state.heads[c] == NONE:
                right_cost += 1

    return [shift_cost, reduce_cost, left_cost, right_cost]


class ArcEagerParser:
    """
    Arc-Eager Transition Parser (top-down right attachments).
    Exposes parent & already-assigned label features for `s0` and `s1`.
    """

    def __init__(self, label_names, lexicon=None, n_rows=2 ** 21):
        self.label_names = label_names
        self.n_labels = len(label_names)
        self.label_number = {name: k for k, name in enumerate(label_names)}
        self.lexicon = lexicon
        # Classes: 0 = SHIFT, 1 = REDUCE, 2..L+1 = LEFT(label), L+2..2L+1 = RIGHT(label)
        self.model = AveragedPerceptron(2 + 2 * self.n_labels, n_rows)

    def class_of(self, move, label_number=0):
        if move == SHIFT:
            return 0
        if move == REDUCE:
            return 1
        if move == LEFT:
            return 2 + label_number
        return 2 + self.n_labels + label_number

    def move_of(self, class_number):
        if class_number == 0:
            return SHIFT, None
        if class_number == 1:
            return REDUCE, None
        if class_number < 2 + self.n_labels:
            return LEFT, self.label_names[class_number - 2]
        return RIGHT, self.label_names[class_number - 2 - self.n_labels]

    def get_features(self, ctx, state):
        f = _extract_shared_features(ctx, state)
        # Add Arc-Eager parent / already-attached label features on s0 and s1
        s0 = state.stack_word(0)
        s1 = state.stack_word(1)
        n0 = state.buffer_word(0)
        n0t = "<NONE>" if n0 == NONE else ctx.tags[n0]
        s0t = "<NONE>" if s0 == NONE else ctx.tags[s0]

        if s0 != NONE and state.heads[s0] != NONE:
            h0 = state.heads[s0]
            h0w = ctx.words[h0]
            h0t = ctx.tags[h0]
            s0l = state.labels[s0]
        else:
            h0w = h0t = s0l = "<UNATTACHED>"

        if s1 != NONE and state.heads[s1] != NONE:
            s1l = state.labels[s1]
        else:
            s1l = "<UNATTACHED>"

        f.extend([
            "ae_s0l=" + s0l,
            "ae_s0l+s0t=" + s0l + "+" + s0t,
            "ae_s0l+n0t=" + s0l + "+" + n0t,
            "ae_h0t+s0l+n0t=" + h0t + "+" + s0l + "+" + n0t,
            "ae_h0w+s0l=" + h0w + "+" + s0l,
            "ae_s1l+s0l=" + s1l + "+" + s0l,
        ])
        return f

    def allowed_scores(self, scores, state):
        scores = scores.copy()
        L = self.n_labels
        if not state.can_shift():
            scores[0] = VERY_LOW
        if not state.can_reduce():
            scores[1] = VERY_LOW
        if not state.can_left():
            scores[2:2 + L] = VERY_LOW
        if not state.can_right():
            scores[2 + L:] = VERY_LOW
        return scores

    def gold_scores(self, scores, state, gold_heads, gold_labels, gold_children):
        L = self.n_labels
        costs = eager_move_costs(state, gold_heads, gold_children)
        best_cost = min(costs)
        result = np.full(len(scores), VERY_LOW, dtype=np.float32)

        if costs[0] == best_cost:
            result[0] = scores[0]
        if costs[1] == best_cost:
            result[1] = scores[1]

        s0 = state.stack_word(0)
        b0 = state.buffer_word(0)
        if costs[2] == best_cost:
            if gold_heads[s0] == b0:
                k = self.class_of(LEFT, self.label_number[gold_labels[s0]])
                result[k] = scores[k]
            else:
                result[2:2 + L] = scores[2:2 + L]

        if costs[3] == best_cost:
            if gold_heads[b0] == s0:
                k = self.class_of(RIGHT, self.label_number[gold_labels[b0]])
                result[k] = scores[k]
            else:
                result[2 + L:] = scores[2 + L:]
        return result

    def parse(self, words, tags, use_average=False, supertags=None):
        ctx = SentenceContext(words, tags, supertags, self.lexicon)
        state = ArcEagerState(len(words) - 1, ctx.words, ctx.is_punct)
        while not state.is_finished():
            features = self.get_features(ctx, state)
            rows = self.model.feature_rows(features)
            scores = self.allowed_scores(self.model.scores(rows, use_average), state)
            move, label = self.move_of(int(scores.argmax()))
            state.apply(move, label)

        # Ensure any remaining unattached tokens get attached cleanly
        root_children = state.right_children[0]
        root_tok = root_children[0] if root_children else 1
        for i in range(1, state.n + 1):
            if state.heads[i] == NONE:
                if not root_children and i == root_tok:
                    state.heads[i] = 0
                    state.labels[i] = "root"
                    root_children.append(i)
                else:
                    state.heads[i] = root_tok if root_tok != i else 0
                    state.labels[i] = base_label(tags[i]) if state.heads[i] != 0 else "root"
        return state.heads, state.labels

    def train_one_sentence(
        self,
        words,
        tags,
        gold_heads,
        gold_labels,
        supertags=None,
        explore_prob=0.90,
        rng=None,
    ):
        ctx = SentenceContext(words, tags, supertags, self.lexicon)
        n = len(words) - 1

        gold_children = [[] for _ in range(n + 1)]
        for i in range(1, n + 1):
            gold_children[gold_heads[i]].append(i)

        state = ArcEagerState(n, ctx.words, ctx.is_punct)
        while not state.is_finished():
            features = self.get_features(ctx, state)
            rows = self.model.feature_rows(features)
            scores = self.model.scores(rows)

            allowed = self.allowed_scores(scores, state)
            gold = self.gold_scores(scores, state, gold_heads, gold_labels, gold_children)
            best_correct = int(gold.argmax())

            augmented = allowed.copy()
            non_gold_mask = gold <= (VERY_LOW * 0.5)
            augmented[non_gold_mask] += 1.0
            predicted = int(augmented.argmax())

            if predicted != best_correct:
                self.model.update(rows, best_correct, predicted)

            model_choice = int(allowed.argmax())
            if rng is not None and rng.random() > explore_prob:
                chosen = best_correct
            else:
                chosen = model_choice
            move, label = self.move_of(chosen)
            state.apply(move, label)
        self.model.next_step()

    def train_one_epoch(self, sentences, tags_list, random_generator, supertags_list=None, epoch=0):
        order = list(range(len(sentences)))
        random_generator.shuffle(order)
        explore_prob = 0.50 if epoch == 0 else 0.90
        for k in order:
            sentence = sentences[k]
            stags = supertags_list[k] if supertags_list is not None else None
            self.train_one_sentence(
                sentence["words"],
                tags_list[k],
                sentence["heads"],
                sentence["labels"],
                stags,
                explore_prob=explore_prob,
                rng=random_generator,
            )


def label_set(state, children):
    labels = sorted(set(state.labels[c] for c in children))
    return "+".join(labels)


def normalize_all(words):
    return [normalize_word(word) for word in words]


def parse_all(parser, sentences, tags_list, use_average=False, supertags_list=None):
    all_heads = []
    all_labels = []
    for k in range(len(sentences)):
        stags = supertags_list[k] if supertags_list is not None else None
        heads, labels = parser.parse(sentences[k]["words"], tags_list[k], use_average, stags)
        all_heads.append(heads)
        all_labels.append(labels)
    return all_heads, all_labels


# ----- reading sentences backwards (right-to-left parser) -----

def flip_position(i, n):
    if i <= 0:
        return i
    return n + 1 - i


def flip_list(values):
    result = [values[0]]
    for i in range(len(values) - 1, 0, -1):
        result.append(values[i])
    return result


def flip_tag(tag):
    if not tag:
        return tag
    if tag[-1] == "<":
        return tag[:-1] + ">"
    if tag[-1] == ">":
        return tag[:-1] + "<"
    return tag


def flip_tags(tags):
    result = [tags[0]]
    for i in range(len(tags) - 1, 0, -1):
        result.append(flip_tag(tags[i]))
    return result


def flip_supertag(stag):
    if ":" not in stag:
        return stag
    off, hcg = stag.split(":", 1)
    if off.startswith("+"):
        off = "-" + off[1:].rstrip("+") + ("-" if off.endswith("+") else "")
    elif off.startswith("-"):
        off = "+" + off[1:].rstrip("-") + ("+" if off.endswith("-") else "")
    return off + ":" + hcg


def flip_supertags(stags):
    result = [stags[0]]
    for i in range(len(stags) - 1, 0, -1):
        result.append(flip_supertag(stags[i]))
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
    n = len(heads) - 1
    old_heads = flip_list(heads)
    new_heads = [old_heads[0]]
    for i in range(1, n + 1):
        new_heads.append(flip_position(old_heads[i], n))
    return new_heads, flip_list(labels)
