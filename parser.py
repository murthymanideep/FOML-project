"""
Dependency parsers built on linear structured perceptrons:
  - chu_liu_edmonds: single-root maximum spanning tree decoder
  - GraphMSTParser: first-order graph parser (48 vectorized edge templates)
  - Parser / ArcEagerParser / ArcStandardParser: three transition systems
  - SecondOrderTreeModel: grandparent, sibling, valency, and crossing scorer
  - FullTreeLabeler: full-tree dependency label classifier
"""

import random
import numpy as np

from data_utils import (
    normalize_word,
    stem_word,
    word_shape,
    word_signature,
    base_label,
    tag_direction,
    coarse_group,
    head_offset_bucket,
    is_punct_word,
    build_lexicon,
    lookup_word_priors,
    lookup_word_clusters,
)
from perceptron import AveragedPerceptron


SHIFT = 0
LEFT = 1
RIGHT = 2
REDUCE = 3
NONE = -1          # "no word here"
VERY_LOW = -1.0e9  # score given to moves that are not allowed


# =====================================================================
# 1. Exact Single-Root Chu-Liu/Edmonds Maximum Spanning Arborescence
# =====================================================================

def _cle_unconstrained(scores, n_nodes):
    """
    Recursive Chu-Liu/Edmonds algorithm returning `par[0..n_nodes-1]`
    that maximizes `sum_{m=1..n_nodes-1} scores[par[m], m]`.
    """
    par = [-1] * n_nodes
    for m in range(1, n_nodes):
        best_h = 0
        best_s = -1.0e18
        for h in range(n_nodes):
            if h != m and scores[h, m] > best_s:
                best_s = float(scores[h, m])
                best_h = h
        par[m] = best_h

    visited = [0] * n_nodes
    visited[0] = 2
    cycle = None
    for start in range(1, n_nodes):
        if visited[start] != 0:
            continue
        path = []
        curr = start
        while curr != 0 and visited[curr] == 0:
            visited[curr] = 1
            path.append(curr)
            curr = par[curr]
        if curr != 0 and visited[curr] == 1:
            idx = path.index(curr)
            cycle = path[idx:]
            break
        for u in path:
            visited[u] = 2

    if cycle is None:
        return par

    in_cycle = [False] * n_nodes
    for u in cycle:
        in_cycle[u] = True

    new_id = [-1] * n_nodes
    orig_of = []
    for u in range(n_nodes):
        if not in_cycle[u]:
            new_id[u] = len(orig_of)
            orig_of.append(u)

    c_id = len(orig_of)
    n_sub = c_id + 1
    sub_scores = np.full((n_sub, n_sub), -1.0e12, dtype=np.float64)

    for u in orig_of:
        iu = new_id[u]
        for v in orig_of:
            if u != v and v != 0:
                sub_scores[iu, new_id[v]] = scores[u, v]

    best_out = {}
    for v in orig_of:
        if v == 0:
            continue
        iv = new_id[v]
        best_u = cycle[0]
        best_val = -1.0e18
        for u in cycle:
            val = float(scores[u, v])
            if val > best_val:
                best_val = val
                best_u = u
        sub_scores[c_id, iv] = best_val
        best_out[iv] = best_u

    best_in = {}
    for u in orig_of:
        iu = new_id[u]
        best_v = cycle[0]
        best_val = -1.0e18
        for v in cycle:
            val = float(scores[u, v] - scores[par[v], v])
            if val > best_val:
                best_val = val
                best_v = v
        sub_scores[iu, c_id] = best_val
        best_in[iu] = best_v

    sub_par = _cle_unconstrained(sub_scores, n_sub)

    for iv in range(1, c_id):
        v = orig_of[iv]
        sp = sub_par[iv]
        if sp == c_id:
            par[v] = best_out[iv]
        else:
            par[v] = orig_of[sp]

    iu_enter = sub_par[c_id]
    u_enter = orig_of[iu_enter]
    v_enter = best_in[iu_enter]
    par[v_enter] = u_enter
    return par


def chu_liu_edmonds(score_matrix):
    """
    Find the exact single-root Maximum Spanning Arborescence rooted at 0.
    `score_matrix` has shape (n + 1, n + 1) where entry [h, m] is the score
    of attaching modifier m (1..n) to head h (0..n).
    Returns `heads` as a list of length n + 1 with `heads[0] = -1`.
    """
    n = score_matrix.shape[0] - 1
    if n <= 0:
        return [-1]
    if n == 1:
        return [-1, 0]

    scores = np.array(score_matrix, dtype=np.float64, copy=True)
    scores[:, 0] = -1.0e12
    for i in range(n + 1):
        scores[i, i] = -1.0e12

    par = _cle_unconstrained(scores, n + 1)
    roots = [m for m in range(1, n + 1) if par[m] == 0]
    if len(roots) == 1:
        return par

    candidates = roots if len(roots) > 0 else [int(np.argmax(scores[0, 1:])) + 1]
    best_par = None
    best_total = -1.0e18
    for r in candidates:
        s_copy = scores.copy()
        s_copy[0, :] = -1.0e12
        s_copy[0, r] = scores[0, r]
        cand_par = _cle_unconstrained(s_copy, n + 1)
        total = sum(float(scores[cand_par[m], m]) for m in range(1, n + 1))
        if total > best_total:
            best_total = total
            best_par = cand_par
    return best_par


# =====================================================================
# 2. Precomputed Sentence Context (with UPOS, Stems, Lookahead & MST)
# =====================================================================

class SentenceContext:
    """Precomputes all token-level strings, UPOS, lookahead tables, and prefix sums."""

    def __init__(
        self,
        raw_words,
        tags,
        supertags=None,
        lexicon=None,
        mst_info=None,
        upos=None,
    ):
        n = len(raw_words) - 1
        self.n = n
        self.raw_words = raw_words
        self.words = [normalize_word(w) for w in raw_words]
        self.stems = [stem_word(w) for w in self.words]
        self.tags = tags
        self.base_tags = [base_label(t) for t in tags]
        self.coarse_tags = [coarse_group(t) for t in tags]
        self.upos = upos if upos is not None else self.coarse_tags
        self.dirs = [tag_direction(t) for t in tags]
        self.shapes = [word_shape(w) for w in raw_words]
        self.sigs = [word_signature(raw_words[i], i) for i in range(n + 1)]
        self.suf3 = [w[-3:] for w in self.words]
        self.wt = [self.words[i] + "/" + tags[i] for i in range(n + 1)]
        self.wc = [self.words[i] + "/" + self.coarse_tags[i] for i in range(n + 1)]
        self.wu = [self.words[i] + "/" + self.upos[i] for i in range(n + 1)]
        self.is_punct = [
            False if i == 0 else (is_punct_word(raw_words[i]) and self.coarse_tags[i] == "PUNCT")
            for i in range(n + 1)
        ]

        if lexicon is not None:
            priors = [
                lookup_word_priors(lexicon, raw_words[i], self.words[i], i)
                for i in range(n + 1)
            ]
            w_clus = [
                lookup_word_clusters(lexicon, raw_words[i], self.words[i], i)
                for i in range(n + 1)
            ]
            self.dom_tags = [p[0] for p in priors]
            self.amb_tags = [p[1] for p in priors]
            self.clusters = [p[3] for p in priors]
            self.fine_clu = [c[0] for c in w_clus]
            self.coarse_clu = [c[1] for c in w_clus]
        else:
            self.dom_tags = ["<NONE>"] * (n + 1)
            self.amb_tags = ["<NONE>"] * (n + 1)
            self.clusters = ["<NONE>"] * (n + 1)
            self.fine_clu = ["FUNK"] * (n + 1)
            self.coarse_clu = ["CUNK"] * (n + 1)

        if supertags is not None:
            self.supertags = supertags
            self.super_offset = [st.split(":")[0] if ":" in st else st for st in supertags]
            self.super_hcg = [st.split(":")[1] if ":" in st else "UNK" for st in supertags]
        else:
            self.supertags = ["<NONE>"] * (n + 1)
            self.super_offset = ["<NONE>"] * (n + 1)
            self.super_hcg = ["<NONE>"] * (n + 1)

        # Global MST guide signals (Nivre & McDonald 2008)
        if mst_info is not None:
            self.mst_h1 = mst_info["heads"]
            self.mst_h2 = mst_info["heads2"]
            self.mst_conf = mst_info["conf"]
            self.mst_max_right_child = [0] * (n + 1)
            self.mst_min_left_child = [n + 1] * (n + 1)
            for m in range(1, n + 1):
                h = self.mst_h1[m]
                if 0 <= h <= n:
                    if m > h and m > self.mst_max_right_child[h]:
                        self.mst_max_right_child[h] = m
                    elif m < h and m < self.mst_min_left_child[h]:
                        self.mst_min_left_child[h] = m
        else:
            self.mst_h1 = None
            self.mst_h2 = None
            self.mst_conf = None
            self.mst_max_right_child = None
            self.mst_min_left_child = None

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
            u = self.upos[i]
            w = self.words[i]
            if cg == "CLAUSE" or u.startswith("VERB"):
                v_cnt += 1
            if w in (",", ";", ":", "--", "-"):
                cm_cnt += 1
            if cg == "CC" or u == "CCONJ":
                cc_cnt += 1
            if cg == "PUNCT" or u == "PUNCT":
                p_cnt += 1
            self.verb_prefix[i] = v_cnt
            self.comma_prefix[i] = cm_cnt
            self.cc_prefix[i] = cc_cnt
            self.punct_prefix[i] = p_cnt
        self.verb_prefix[n + 1] = v_cnt
        self.comma_prefix[n + 1] = cm_cnt
        self.cc_prefix[n + 1] = cc_cnt
        self.punct_prefix[n + 1] = p_cnt

        # Buffer lookahead arrays
        self.next_clause_pos = [0] * (n + 2)
        self.has_root_ahead = ["False"] * (n + 2)
        next_cl = 0
        seen_root = False
        for i in range(n, 0, -1):
            if self.tags[i] == "root":
                seen_root = True
            if self.coarse_tags[i] == "CLAUSE" or self.tags[i] == "root" or self.upos[i].startswith("VERB"):
                next_cl = i
            self.next_clause_pos[i] = next_cl
            self.has_root_ahead[i] = "True" if seen_root else "False"

        self.np_end_pos = [0] * (n + 2)
        for i in range(1, n + 1):
            if self.coarse_tags[i] in ("DET", "MOD", "COMP", "SUBJ", "OBJ", "NMOD") or self.upos[i] in ("DET", "ADJ", "NOUN", "PROPN", "NUM"):
                j = i
                while j < n and (self.coarse_tags[j] in ("DET", "MOD", "COMP") or self.upos[j] in ("DET", "ADJ", "PROPN")) and self.dirs[j] == ">":
                    if self.coarse_tags[j + 1] in ("MOD", "COMP", "SUBJ", "OBJ", "NMOD", "CLAUSE") or self.upos[j + 1] in ("ADJ", "NOUN", "PROPN", "NUM"):
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


# =====================================================================
# 3. Fast Vectorized 42-Template First-Order Graph-Based MST Parser
# =====================================================================

_MASK_64 = (1 << 64) - 1
_P1 = 0x9E3779B97F4A7C15
_P2 = 0xBF58476D1CE4E5B9
_P3 = 0x94D049BB133111EB


def _h64(text):
    """Deterministic 64-bit integer hash for a string token attribute."""
    return hash(text) & _MASK_64


class GraphMSTParser:
    """
    Global First-Order Arc-Factored Graph-Based Dependency Parser trained with
    Margin-Infused Structured Perceptron and Chu-Liu/Edmonds MST decoding.
    Uses 42 vectorized 2D NumPy uint64 feature templates (including UPOS).
    """

    def __init__(self, lexicon=None, n_rows=2 ** 22):
        self.lexicon = lexicon
        self.n_rows = n_rows
        self.mask = np.uint64(n_rows - 1)
        self.weights = np.zeros(n_rows, dtype=np.float32)
        self.totals = np.zeros(n_rows, dtype=np.float64)
        self.timestamps = np.zeros(n_rows, dtype=np.int32)
        self.step = 0

        rng = random.Random(2026)
        self.template_seeds = [
            np.uint64(rng.getrandbits(64) | 1) for _ in range(48)
        ]

    def _extract_edge_feature_cube(self, ctx):
        """
        Return a 3D uint64 array `F` of shape `(48, n + 1, n + 1)` containing
        the hashed feature row indices for every directed edge (h -> m).
        """
        n = ctx.n
        size = n + 1

        hw = np.array([_h64("w:" + ctx.words[i]) for i in range(size)], dtype=np.uint64)
        hst = np.array([_h64("st:" + ctx.stems[i]) for i in range(size)], dtype=np.uint64)
        ht = np.array([_h64("t:" + ctx.tags[i]) for i in range(size)], dtype=np.uint64)
        hbt = np.array([_h64("bt:" + ctx.base_tags[i]) for i in range(size)], dtype=np.uint64)
        hcg = np.array([_h64("cg:" + ctx.coarse_tags[i]) for i in range(size)], dtype=np.uint64)
        hu = np.array([_h64("u:" + ctx.upos[i]) for i in range(size)], dtype=np.uint64)
        hwt = np.array([_h64("wt:" + ctx.wt[i]) for i in range(size)], dtype=np.uint64)
        hwc = np.array([_h64("wc:" + ctx.wc[i]) for i in range(size)], dtype=np.uint64)
        hwu = np.array([_h64("wu:" + ctx.wu[i]) for i in range(size)], dtype=np.uint64)
        hsig = np.array([_h64("sig:" + ctx.sigs[i]) for i in range(size)], dtype=np.uint64)
        hdom = np.array([_h64("dom:" + ctx.dom_tags[i]) for i in range(size)], dtype=np.uint64)
        hsup = np.array([_h64("sup:" + ctx.supertags[i]) for i in range(size)], dtype=np.uint64)
        hdir = np.array([_h64("dir:" + ctx.dirs[i]) for i in range(size)], dtype=np.uint64)
        hfc = np.array([_h64("fc:" + ctx.fine_clu[i]) for i in range(size)], dtype=np.uint64)
        hcc = np.array([_h64("cc:" + ctx.coarse_clu[i]) for i in range(size)], dtype=np.uint64)

        ht_l1 = np.array(
            [_h64("tl1:" + (ctx.tags[i - 1] if i > 1 else "<S>")) for i in range(size)],
            dtype=np.uint64,
        )
        ht_r1 = np.array(
            [_h64("tr1:" + (ctx.tags[i + 1] if 0 < i < n else "<E>")) for i in range(size)],
            dtype=np.uint64,
        )
        hu_l1 = np.array(
            [_h64("ul1:" + (ctx.upos[i - 1] if i > 1 else "<S>")) for i in range(size)],
            dtype=np.uint64,
        )
        hu_r1 = np.array(
            [_h64("ur1:" + (ctx.upos[i + 1] if 0 < i < n else "<E>")) for i in range(size)],
            dtype=np.uint64,
        )
        hw_l1 = np.array(
            [_h64("wl1:" + (ctx.words[i - 1] if i > 1 else "<S>")) for i in range(size)],
            dtype=np.uint64,
        )
        hw_r1 = np.array(
            [_h64("wr1:" + (ctx.words[i + 1] if 0 < i < n else "<E>")) for i in range(size)],
            dtype=np.uint64,
        )

        dd_mat = np.zeros((size, size), dtype=np.uint64)
        dir_mat = np.zeros((size, size), dtype=np.uint64)
        btw_mat = np.zeros((size, size), dtype=np.uint64)
        sup_match_mat = np.zeros((size, size), dtype=np.uint64)

        for h in range(size):
            h_cg = "ROOT" if h == 0 else ctx.coarse_tags[h]
            for m in range(1, size):
                if h == m:
                    continue
                if h == 0:
                    d_str = "ROOT"
                    dr_str = "ROOT"
                else:
                    dr_str = "R" if m > h else "L"
                    dist = abs(m - h)
                    if dist <= 5:
                        db = str(dist)
                    elif dist <= 10:
                        db = "6-10"
                    else:
                        db = "11+"
                    d_str = dr_str + ":" + db
                dd_mat[h, m] = _h64(d_str)
                dir_mat[h, m] = _h64(dr_str)
                btw_mat[h, m] = _h64(d_str + ":" + ctx.between_counts(h, m))

                cand_off = head_offset_bucket(m, h)
                off_ok = ctx.super_offset[m] == cand_off
                cg_ok = ctx.super_hcg[m] == h_cg
                sup_match_mat[h, m] = _h64(f"sm:{off_ok}:{cg_ok}:{ctx.base_tags[m]}")

        p1 = np.uint64(_P1)
        p2 = np.uint64(_P2)
        p3 = np.uint64(_P3)
        seeds = self.template_seeds
        mask = self.mask

        def mix(a_h, b_m, rel, idx):
            return ((a_h[:, None] * p1) ^ (b_m[None, :] * p2) ^ (rel * p3) ^ seeds[idx]) & mask

        zero_vec = np.zeros(size, dtype=np.uint64)
        templates = [
            # Core lexical + tag + distance/direction templates
            mix(hw, hw, dir_mat, 0),
            mix(hw, hw, dd_mat, 1),
            mix(hst, hst, dir_mat, 2),
            mix(hst, hst, dd_mat, 3),
            mix(hw, ht, dir_mat, 4),
            mix(hw, ht, dd_mat, 5),
            mix(ht, hw, dir_mat, 6),
            mix(ht, hw, dd_mat, 7),
            mix(hst, ht, dd_mat, 8),
            mix(ht, hst, dd_mat, 9),
            mix(hwt, hwt, dir_mat, 10),
            mix(hwt, ht, dd_mat, 11),
            mix(ht, hwt, dd_mat, 12),
            mix(hwc, hwc, dd_mat, 13),
            mix(ht, ht, dir_mat, 14),
            mix(ht, ht, dd_mat, 15),
            mix(hbt, hbt, dd_mat, 16),
            mix(hcg, hcg, dd_mat, 17),

            # In-between verb/comma/conjunction structural templates
            mix(ht, ht, btw_mat, 18),
            mix(hcg, hcg, btw_mat, 19),
            mix(hbt, hbt, btw_mat, 20),
            mix(hwc, hcg, btw_mat, 21),
            mix(hcg, hwc, btw_mat, 22),

            # Contextual surrounding tag/word templates
            mix(ht ^ (ht_l1 * p3), ht, dd_mat, 23),
            mix(ht ^ (ht_r1 * p3), ht, dd_mat, 24),
            mix(ht, ht ^ (ht_l1 * p3), dd_mat, 25),
            mix(ht, ht ^ (ht_r1 * p3), dd_mat, 26),
            mix(ht ^ (ht_r1 * p3), ht ^ (ht_l1 * p3), dir_mat, 27),
            mix(hw ^ (hw_r1 * p3), ht, dir_mat, 28),
            mix(ht, hw ^ (hw_l1 * p3), dir_mat, 29),

            # Lexicon prior, signature, direction, & supertag agreement templates
            mix(hdom, hdom, dd_mat, 30),
            mix(hsig, hsig, dd_mat, 31),
            mix(hdir, hdir, dd_mat, 32),
            mix(hsup, hcg, dd_mat, 33),
            mix(ht, hsup, dir_mat, 34),
            mix(zero_vec, ht, sup_match_mat, 35),

            # Induced POS (`upos`) global arc templates
            mix(hu, hu, dd_mat, 36),
            mix(hwu, hwu, dir_mat, 37),
            mix(hu, ht, btw_mat, 38),
            mix(ht, hu, btw_mat, 39),
            mix(hu ^ (hu_r1 * p3), hu ^ (hu_l1 * p3), dd_mat, 40),
            mix(hwu, hu, btw_mat, 41),

            # Unsupervised Bigram Class-HMM word cluster global arc templates
            mix(hfc, hfc, dir_mat, 42),
            mix(hfc, hfc, dd_mat, 43),
            mix(hcc, hcc, btw_mat, 44),
            mix(hfc, ht, dd_mat, 45),
            mix(ht, hfc, dd_mat, 46),
            mix(hfc, hu, btw_mat, 47),
        ]
        return np.stack(templates, axis=0)

    def score_Create_matrix(self, ctx):
        F = self._extract_edge_feature_cube(ctx)
        w = self.weights
        scores = w[F].sum(axis=0, dtype=np.float64)
        n = ctx.n
        scores[:, 0] = -1.0e12
        for i in range(n + 1):
            scores[i, i] = -1.0e12
        for h in range(1, n + 1):
            if ctx.is_punct[h]:
                for m in range(1, n + 1):
                    if not ctx.is_punct[m]:
                        scores[h, m] = -1.0e9
        return scores, F

    def train_one_sentence(self, words, tags, gold_heads, supertags=None, upos=None):
        ctx = SentenceContext(words, tags, supertags, self.lexicon, upos=upos)
        n = ctx.n
        scores, F = self.score_Create_matrix(ctx)

        edge_margin = 4.0
        aug_scores = scores.copy()
        for m in range(1, n + 1):
            gh = gold_heads[m]
            aug_scores[:, m] += edge_margin
            aug_scores[gh, m] -= edge_margin

        pred_heads = chu_liu_edmonds(aug_scores)
        wrong = [m for m in range(1, n + 1) if pred_heads[m] != gold_heads[m]]
        if wrong:
            violation = 0.0
            for m in wrong:
                gh = gold_heads[m]
                ph = pred_heads[m]
                violation += edge_margin + float(scores[ph, m]) - float(scores[gh, m])
            if violation > 0.0:
                tau = violation / (2.0 * F.shape[0] * len(wrong))
                if tau > 1.0:
                    tau = 1.0
                for m in wrong:
                    gh = gold_heads[m]
                    ph = pred_heads[m]
                    self._update_rows(F[:, gh, m], +tau)
                    self._update_rows(F[:, ph, m], -tau)
        self.step += 1

    def _update_rows(self, rows, delta):
        elapsed = self.step - self.timestamps[rows]
        self.totals[rows] += elapsed * self.weights[rows]
        self.weights[rows] += delta
        self.timestamps[rows] = self.step

    def average_weights(self):
        elapsed = self.step - self.timestamps
        self.totals += elapsed * self.weights
        if self.step > 0:
            self.weights = (self.totals / self.step).astype(np.float32)

    def train(self, sentences, tags_list, supertags_list=None, upos_list=None, epochs=8, seed=1):
        rng = random.Random(seed)
        order = list(range(len(sentences)))
        for epoch in range(epochs):
            rng.shuffle(order)
            for k in order:
                stags = supertags_list[k] if supertags_list is not None else None
                utags = upos_list[k] if upos_list is not None else None
                self.train_one_sentence(
                    sentences[k]["words"],
                    tags_list[k],
                    sentences[k]["heads"],
                    stags,
                    utags,
                )
        self.average_weights()

    def predict_mst_info(self, words, tags, supertags=None, upos=None, second_model=None):
        """
        Return a dictionary with:
          - 'heads'  : 1-best Chu-Liu/Edmonds MST heads [0..n]
          - 'heads2' : 2nd-best head per token [0..n]
          - 'heads3' : 3rd-best head per token [0..n]
          - 'conf'   : margin confidence bucket ('HIGH', 'MID', 'LOW') per token
          - 'margins': numeric margin per token
        """
        ctx = SentenceContext(words, tags, supertags, self.lexicon, upos=upos)
        n = ctx.n
        scores, _ = self.score_Create_matrix(ctx)
        if second_model is not None:
            scores2, _ = second_model.score_Create_matrix(ctx)
            scores = 0.5 * (scores + scores2)

        mst_heads = chu_liu_edmonds(scores)

        heads2 = [-1] * (n + 1)
        heads3 = [-1] * (n + 1)
        conf = ["<NONE>"] * (n + 1)
        margins = [0.0] * (n + 1)
        for m in range(1, n + 1):
            h1 = mst_heads[m]
            s1 = float(scores[h1, m])
            col = scores[:, m].copy()
            col[h1] = -1.0e12
            col[m] = -1.0e12
            h2 = int(np.argmax(col))
            s2 = float(col[h2])
            col[h2] = -1.0e12
            h3 = int(np.argmax(col))
            marg = max(0.0, s1 - s2)
            heads2[m] = h2
            heads3[m] = h3
            margins[m] = marg
            if marg >= 6.0:
                conf[m] = "HIGH"
            elif marg >= 2.5:
                conf[m] = "MID"
            else:
                conf[m] = "LOW"

        return {
            "heads": mst_heads,
            "heads2": heads2,
            "heads3": heads3,
            "conf": conf,
            "margins": margins,
        }


def jackknife_mst(sentences, tags_list, supertags_list=None, upos_list=None, epochs=7, n_folds=5, word_clusters=None):
    """
    Produce out-of-fold Global Graph MST predictions for training sentences
    so transition parsers can use them without overfitting.
    """
    mst_infos = [None] * len(sentences)
    for fold in range(n_folds):
        train_sents = []
        train_tags = []
        train_super = []
        train_upos = []
        held_out = []
        for k in range(len(sentences)):
            if k % n_folds == fold:
                held_out.append(k)
            else:
                train_sents.append(sentences[k])
                train_tags.append(tags_list[k])
                if supertags_list is not None:
                    train_super.append(supertags_list[k])
                if upos_list is not None:
                    train_upos.append(upos_list[k])

        fold_tag_sents = [
            {"words": train_sents[i]["words"], "labels": train_tags[i]}
            for i in range(len(train_sents))
        ]
        fold_lexicon = build_lexicon(fold_tag_sents, word_clusters=word_clusters)
        model = GraphMSTParser(lexicon=fold_lexicon)
        model.train(
            train_sents,
            train_tags,
            train_super if supertags_list is not None else None,
            train_upos if upos_list is not None else None,
            epochs=epochs,
            seed=fold + 1,
        )
        for k in held_out:
            stags = supertags_list[k] if supertags_list is not None else None
            utags = upos_list[k] if upos_list is not None else None
            mst_infos[k] = model.predict_mst_info(
                sentences[k]["words"], tags_list[k], stags, utags
            )
        print("  MST fold %d/%d done" % (fold + 1, n_folds))
    return mst_infos


# =====================================================================
# 4. Transition Parser States & Shared Feature Extractor
# =====================================================================

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
        self.margins = [1.0] * size
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
        if self.is_punct is not None and self.is_punct[b0]:
            s0 = self.stack[-1]
            if not self.is_punct[s0] and self.can_shift():
                return False
        return True

    def can_right(self):
        if len(self.stack) < 2:
            return False
        if self.is_punct is not None and self.can_shift():
            s1 = self.stack[-2]
            s0 = self.stack[-1]
            if self.is_punct[s1] and not self.is_punct[s0]:
                return False
        return True

    def add_arc(self, head, dependent, label, margin=1.0):
        self.heads[dependent] = head
        self.labels[dependent] = label
        self.margins[dependent] = margin
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

    def apply(self, move, label, margin=1.0):
        if move == SHIFT:
            self.stack.append(self.buffer_position)
            self.buffer_position += 1
        elif move == LEFT:
            dependent = self.stack.pop()
            self.add_arc(self.buffer_word(0), dependent, label, margin)
        else:
            dependent = self.stack.pop()
            self.add_arc(self.stack[-1], dependent, label, margin)


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
    """Shared feature extractor for Arc-Hybrid, Arc-Eager, and Arc-Standard states."""
    words = ctx.words
    stems = ctx.stems
    tags = ctx.tags
    base_tags = ctx.base_tags
    coarse_tags = ctx.coarse_tags
    upos = ctx.upos
    dirs = ctx.dirs
    wt_arr = ctx.wt
    wc_arr = ctx.wc
    wu_arr = ctx.wu
    suf3 = ctx.suf3
    sigs = ctx.sigs
    dom_tags = ctx.dom_tags
    amb_tags = ctx.amb_tags
    clusters = ctx.clusters
    fine_clu = ctx.fine_clu
    coarse_clu = ctx.coarse_clu
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

    def st(idx):
        return "<NONE>" if idx == NONE else stems[idx]

    def t(idx):
        return "<NONE>" if idx == NONE else tags[idx]

    def bt(idx):
        return "<NONE>" if idx == NONE else base_tags[idx]

    def cg(idx):
        return "<NONE>" if idx == NONE else coarse_tags[idx]

    def u(idx):
        return "<NONE>" if idx == NONE else upos[idx]

    def lab(idx):
        return "<NONE>" if idx == NONE else state.labels[idx]

    s0w = w(s0)
    s0st = st(s0)
    s0t = t(s0)
    s0bt = bt(s0)
    s0cg = cg(s0)
    s0u = u(s0)
    s0wt = "<NONE>" if s0 == NONE else wt_arr[s0]
    s0wc = "<NONE>" if s0 == NONE else wc_arr[s0]
    s0wu = "<NONE>" if s0 == NONE else wu_arr[s0]
    s0d = "<NONE>" if s0 == NONE else dirs[s0]
    s0sig = "<NONE>" if s0 == NONE else sigs[s0]
    s0amb = "<NONE>" if s0 == NONE else amb_tags[s0]
    s0clu = "<NONE>" if s0 == NONE else clusters[s0]
    s0fc = "<NONE>" if s0 == NONE else fine_clu[s0]
    s0cc = "<NONE>" if s0 == NONE else coarse_clu[s0]

    s1w = w(s1)
    s1st = st(s1)
    s1t = t(s1)
    s1bt = bt(s1)
    s1cg = cg(s1)
    s1u = u(s1)
    s1wt = "<NONE>" if s1 == NONE else wt_arr[s1]
    s1wc = "<NONE>" if s1 == NONE else wc_arr[s1]
    s1wu = "<NONE>" if s1 == NONE else wu_arr[s1]
    s1d = "<NONE>" if s1 == NONE else dirs[s1]
    s1amb = "<NONE>" if s1 == NONE else amb_tags[s1]
    s1fc = "<NONE>" if s1 == NONE else fine_clu[s1]
    s1cc = "<NONE>" if s1 == NONE else coarse_clu[s1]

    s2w = w(s2)
    s2t = t(s2)
    s2cg = cg(s2)
    s2u = u(s2)
    s3t = t(s3)

    n0w = w(n0)
    n0st = st(n0)
    n0t = t(n0)
    n0bt = bt(n0)
    n0cg = cg(n0)
    n0u = u(n0)
    n0wt = "<NONE>" if n0 == NONE else wt_arr[n0]
    n0wc = "<NONE>" if n0 == NONE else wc_arr[n0]
    n0wu = "<NONE>" if n0 == NONE else wu_arr[n0]
    n0d = "<NONE>" if n0 == NONE else dirs[n0]
    n0sig = "<NONE>" if n0 == NONE else sigs[n0]
    n0amb = "<NONE>" if n0 == NONE else amb_tags[n0]
    n0clu = "<NONE>" if n0 == NONE else clusters[n0]
    n0fc = "<NONE>" if n0 == NONE else fine_clu[n0]
    n0cc = "<NONE>" if n0 == NONE else coarse_clu[n0]

    n1w = w(n1)
    n1t = t(n1)
    n1cg = cg(n1)
    n1u = u(n1)
    n1cc = "<NONE>" if n1 == NONE else coarse_clu[n1]
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

    # Buffer lookahead features
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

        # Single words, stems, tags, UPOS, signatures, and lexicon priors
        "s0w=" + s0w, "s0st=" + s0st, "s0t=" + s0t, "s0wt=" + s0wt, "s0bt=" + s0bt, "s0cg=" + s0cg, "s0u=" + s0u,
        "s1w=" + s1w, "s1st=" + s1st, "s1t=" + s1t, "s1wt=" + s1wt, "s1bt=" + s1bt, "s1cg=" + s1cg, "s1u=" + s1u,
        "s2w=" + s2w, "s2t=" + s2t, "s2cg=" + s2cg, "s2u=" + s2u, "s3t=" + s3t,
        "n0w=" + n0w, "n0st=" + n0st, "n0t=" + n0t, "n0wt=" + n0wt, "n0bt=" + n0bt, "n0cg=" + n0cg, "n0u=" + n0u,
        "n1w=" + n1w, "n1t=" + n1t, "n1wt=" + n1wt, "n1cg=" + n1cg, "n1u=" + n1u,
        "n2w=" + n2w, "n2t=" + n2t, "n2cg=" + n2cg,
        "s0sig=" + s0sig, "n0sig=" + n0sig,
        "s0amb=" + s0amb, "n0amb=" + n0amb, "s1amb=" + s1amb,
        "s0clu=" + s0clu, "n0clu=" + n0clu,
        "s0p=" + ("<NONE>" if s0 == NONE else dom_tags[s0]),
        "n0p=" + ("<NONE>" if n0 == NONE else dom_tags[n0]),
        "s1p=" + ("<NONE>" if s1 == NONE else dom_tags[s1]),

        # Pairs of stack0 and buffer0 (words, stems, tags, UPOS, coarse groups)
        "s0wt+n0wt=" + s0wt + "+" + n0wt,
        "s0wt+n0w=" + s0wt + "+" + n0w,
        "s0w+n0wt=" + s0w + "+" + n0wt,
        "s0wt+n0t=" + s0wt + "+" + n0t,
        "s0t+n0wt=" + s0t + "+" + n0wt,
        "s0w+n0w=" + s0w + "+" + n0w,
        "s0st+n0st=" + s0st + "+" + n0st,
        "s0st+n0t=" + s0st + "+" + n0t,
        "s0t+n0st=" + s0t + "+" + n0st,
        "s0t+n0t=" + s0t + "+" + n0t,
        "s0bt+n0bt=" + s0bt + "+" + n0bt,
        "s0cg+n0cg=" + s0cg + "+" + n0cg,
        "s0u+n0u=" + s0u + "+" + n0u,
        "s0wu+n0wu=" + s0wu + "+" + n0wu,
        "s0wc+n0wc=" + s0wc + "+" + n0wc,
        "s0sig+n0sig=" + s0sig + "+" + n0sig,
        "s0amb+n0amb=" + s0amb + "+" + n0amb,
        "s0clu+n0clu=" + s0clu + "+" + n0clu,

        # Pairs of stack1 and stack0
        "s1wt+s0wt=" + s1wt + "+" + s0wt,
        "s1wt+s0w=" + s1wt + "+" + s0w,
        "s1w+s0wt=" + s1w + "+" + s0wt,
        "s1wt+s0t=" + s1wt + "+" + s0t,
        "s1t+s0wt=" + s1t + "+" + s0wt,
        "s1w+s0w=" + s1w + "+" + s0w,
        "s1st+s0st=" + s1st + "+" + s0st,
        "s1st+s0t=" + s1st + "+" + s0t,
        "s1t+s0st=" + s1t + "+" + s0st,
        "s1t+s0t=" + s1t + "+" + s0t,
        "s1bt+s0bt=" + s1bt + "+" + s0bt,
        "s1cg+s0cg=" + s1cg + "+" + s0cg,
        "s1u+s0u=" + s1u + "+" + s0u,
        "s1wu+s0wu=" + s1wu + "+" + s0wu,
        "s1wc+s0wc=" + s1wc + "+" + s0wc,
        "s1amb+s0amb=" + s1amb + "+" + s0amb,

        # Other pairs
        "s1t+n0t=" + s1t + "+" + n0t,
        "s1u+n0u=" + s1u + "+" + n0u,
        "s1w+n0w=" + s1w + "+" + n0w,
        "s1st+n0st=" + s1st + "+" + n0st,
        "s1wt+n0t=" + s1wt + "+" + n0t,
        "s1t+n0wt=" + s1t + "+" + n0wt,
        "n0t+n1t=" + n0t + "+" + n1t,
        "n0u+n1u=" + n0u + "+" + n1u,
        "n0wt+n1t=" + n0wt + "+" + n1t,
        "n0t+n1wt=" + n0t + "+" + n1wt,

        # Three-word combinations
        "s1t+s0t+n0t=" + s1t + "+" + s0t + "+" + n0t,
        "s1bt+s0bt+n0bt=" + s1bt + "+" + s0bt + "+" + n0bt,
        "s1cg+s0cg+n0cg=" + s1cg + "+" + s0cg + "+" + n0cg,
        "s1u+s0u+n0u=" + s1u + "+" + s0u + "+" + n0u,
        "s1u+s0t+n0u=" + s1u + "+" + s0t + "+" + n0u,
        "s1w+s0t+n0t=" + s1w + "+" + s0t + "+" + n0t,
        "s1t+s0w+n0t=" + s1t + "+" + s0w + "+" + n0t,
        "s1t+s0t+n0w=" + s1t + "+" + s0t + "+" + n0w,
        "s1w+s0w+n0t=" + s1w + "+" + s0w + "+" + n0t,
        "s1t+s0w+n0w=" + s1t + "+" + s0w + "+" + n0w,
        "s2t+s1t+s0t=" + s2t + "+" + s1t + "+" + s0t,
        "s2cg+s1cg+s0cg=" + s2cg + "+" + s1cg + "+" + s0cg,
        "s2u+s1u+s0u=" + s2u + "+" + s1u + "+" + s0u,
        "s3t+s2t+s1t+s0t=" + s3t + "+" + s2t + "+" + s1t + "+" + s0t,
        "s2w+s1t+s0t=" + s2w + "+" + s1t + "+" + s0t,
        "s0t+n0t+n1t=" + s0t + "+" + n0t + "+" + n1t,
        "s0cg+n0cg+n1cg=" + s0cg + "+" + n0cg + "+" + n1cg,
        "s0u+n0u+n1u=" + s0u + "+" + n0u + "+" + n1u,
        "s0w+n0t+n1t=" + s0w + "+" + n0t + "+" + n1t,
        "s0t+n0w+n1t=" + s0t + "+" + n0w + "+" + n1t,
        "s0t+n0t+n1w=" + s0t + "+" + n0t + "+" + n1w,
        "n0t+n1t+n2t=" + n0t + "+" + n1t + "+" + n2t,
        "s1t+s0t+n0t+n1t=" + s1t + "+" + s0t + "+" + n0t + "+" + n1t,

        # Distance & In-Between Structural Features
        "s0w+d=" + s0w + "+" + distance,
        "s0t+d=" + s0t + "+" + distance,
        "n0w+d=" + n0w + "+" + distance,
        "n0t+d=" + n0t + "+" + distance,
        "s0w+n0w+d=" + s0w + "+" + n0w + "+" + distance,
        "s0t+n0t+d=" + s0t + "+" + n0t + "+" + distance,
        "s0u+n0u+d=" + s0u + "+" + n0u + "+" + distance,
        "s1w+ds1=" + s1w + "+" + distance_s1,
        "s1t+ds1=" + s1t + "+" + distance_s1,
        "s1w+s0w+ds1=" + s1w + "+" + s0w + "+" + distance_s1,
        "s1t+s0t+ds1=" + s1t + "+" + s0t + "+" + distance_s1,
        "s1u+s0u+ds1=" + s1u + "+" + s0u + "+" + distance_s1,
        "btw_s0n0=" + s0t + "+" + n0t + "+" + between_s0_n0,
        "btw_s0n0_cg=" + s0cg + "+" + n0cg + "+" + between_s0_n0,
        "btw_s0n0_u=" + s0u + "+" + n0u + "+" + between_s0_n0,
        "btw_s1s0=" + s1t + "+" + s0t + "+" + between_s1_s0,
        "btw_s1s0_cg=" + s1cg + "+" + s0cg + "+" + between_s1_s0,
        "btw_s1s0_u=" + s1u + "+" + s0u + "+" + between_s1_s0,

        # Children of s0, n0, and s1 (2nd-order siblings & grandchildren)
        "s0_lc1_t=" + t(s0_lc1), "s0_lc1_l=" + lab(s0_lc1), "s0_lc1_w=" + w(s0_lc1),
        "s0_rc1_t=" + t(s0_rc1), "s0_rc1_l=" + lab(s0_rc1), "s0_rc1_w=" + w(s0_rc1),
        "n0_lc1_t=" + t(n0_lc1), "n0_lc1_l=" + lab(n0_lc1), "n0_lc1_w=" + w(n0_lc1),
        "s1_rc1_t=" + t(s1_rc1), "s1_rc1_l=" + lab(s1_rc1), "s1_rc1_w=" + w(s1_rc1),
        "s1_lc1_t=" + t(s1_lc1), "s1_lc1_l=" + lab(s1_lc1), "s1_lc1_w=" + w(s1_lc1),
        "s0_lc2_l=" + lab(s0_lc2), "s0_rc2_l=" + lab(s0_rc2),
        "n0_lc2_l=" + lab(n0_lc2), "s1_rc2_l=" + lab(s1_rc2), "s1_lc2_l=" + lab(s1_lc2),

        # Grandchild features (3rd-order)
        "s0_lc1_lc1_l=" + lab(s0_lc1_lc1) + "+" + s0t,
        "n0_lc1_lc1_l=" + lab(n0_lc1_lc1) + "+" + n0t,
        "s1_rc1_rc1_l=" + lab(s1_rc1_rc1) + "+" + s1t,

        "s0t+s0_lc1_t+n0t=" + s0t + "+" + t(s0_lc1) + "+" + n0t,
        "s0t+s0_rc1_t+n0t=" + s0t + "+" + t(s0_rc1) + "+" + n0t,
        "s0t+n0t+n0_lc1_t=" + s0t + "+" + n0t + "+" + t(n0_lc1),
        "s0t+s0_lc1_l+n0t=" + s0t + "+" + lab(s0_lc1) + "+" + n0t,
        "s0t+s0_rc1_l+n0t=" + s0t + "+" + lab(s0_rc1) + "+" + n0t,
        "s0t+n0t+n0_lc1_l=" + s0t + "+" + n0t + "+" + lab(n0_lc1),
        "s1t+s1_rc1_t+s0t=" + s1t + "+" + t(s1_rc1) + "+" + s0t,
        "s1t+s1_lc1_t+s0t=" + s1t + "+" + t(s1_lc1) + "+" + s0t,
        "s1t+s1_rc1_l+s0t=" + s1t + "+" + lab(s1_rc1) + "+" + s0t,
        "s1t+s1_lc1_l+s0t=" + s1t + "+" + lab(s1_lc1) + "+" + s0t,
        "s1t+s0t+s0_lc1_l=" + s1t + "+" + s0t + "+" + lab(s0_lc1),
        "s1t+s0t+s0_rc1_l=" + s1t + "+" + s0t + "+" + lab(s0_rc1),

        # Coordination (conj / cc) symmetry features
        "coord_s1_s0=" + s1cg + "+" + lab(s0_lc1) + "+" + w(s0_lc1) + "+" + s0cg,
        "coord_s1_s0_u=" + s1u + "+" + lab(s0_lc1) + "+" + w(s0_lc1) + "+" + s0u,
        "coord_s0_n0=" + s0cg + "+" + lab(n0_lc1) + "+" + w(n0_lc1) + "+" + n0cg,
        "coord_s0_n0_u=" + s0u + "+" + lab(n0_lc1) + "+" + w(n0_lc1) + "+" + n0u,

        # Valency & Child Label-Set Features
        "s0w+n_left=" + s0w + "+" + s0_n_left,
        "s0t+n_left=" + s0t + "+" + s0_n_left,
        "s0w+n_right=" + s0w + "+" + s0_n_right,
        "s0t+n_right=" + s0t + "+" + s0_n_right,
        "n0w+n_left=" + n0w + "+" + n0_n_left,
        "n0t+n_left=" + n0t + "+" + n0_n_left,
        "s1w+n_right=" + s1w + "+" + s1_n_right,
        "s1t+n_right=" + s1t + "+" + s1_n_right,
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
        "s1st+s0_case=" + s1st + "+" + s0_case + "+" + s0cg,
        "s1u+s0_case=" + s1u + "+" + s0_case + "+" + s0u,
        "s1t+s0_case=" + s1t + "+" + s0_case + "+" + s0t,
        "s1t+s0_case+s0w=" + s1t + "+" + s0_case + "+" + s0w,
        "n0w+s0_case=" + n0w + "+" + s0_case + "+" + s0cg,
        "n0st+s0_case=" + n0st + "+" + s0_case + "+" + s0cg,
        "n0u+s0_case=" + n0u + "+" + s0_case + "+" + s0u,
        "n0t+s0_case=" + n0t + "+" + s0_case + "+" + s0t,
        "s0w+n0_case=" + s0w + "+" + n0_case + "+" + n0cg,
        "s0st+n0_case=" + s0st + "+" + n0_case + "+" + n0cg,
        "s0u+n0_case=" + s0u + "+" + n0_case + "+" + n0u,
        "s1_aux+s0_case=" + s1_aux + "+" + s0_case + "+" + s0cg,
        "n0_aux+s0t=" + n0_aux + "+" + s0t,
        "s1_aux+s0t=" + s1_aux + "+" + s0t,
        "s1t+s0_mark=" + s1t + "+" + s0_mark + "+" + s0t,
        "s1u+s0_mark=" + s1u + "+" + s0_mark + "+" + s0u,
        "s1w+s0_mark=" + s1w + "+" + s0_mark + "+" + s0cg,
        "s1st+s0_mark=" + s1st + "+" + s0_mark + "+" + s0cg,
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
        "s0sup=" + s0_st,
        "n0sup=" + n0_st,
        "s1sup=" + s1_st,
        "s0sup+n0cg=" + s0_st + "+" + n0cg + "+" + distance,
        "s0sup+s1cg=" + s0_st + "+" + s1cg + "+" + distance_s1,
        "left_off_match=" + left_off_match + "+" + s0bt,
        "right_off_match=" + right_off_match + "+" + s0bt,
        "s0off+d=" + s0_pred_off + "+" + distance + "+" + distance_s1,
        "look_root=" + root_ahead + "+" + s0t + "+" + n0t,
        "look_cl=" + s1t + "+" + s0t + "+" + next_cl_t + "+" + next_cl_dist,
        "look_np=" + s0t + "+" + n0t + "+" + np_end_t + "+" + np_end_dist,

        # Unsupervised Bigram Class-HMM word cluster features (Koo et al. 2008)
        "s0fc=" + s0fc, "s0cc=" + s0cc,
        "s1fc=" + s1fc, "s1cc=" + s1cc,
        "n0fc=" + n0fc, "n0cc=" + n0cc,
        "s0fc+n0fc=" + s0fc + "+" + n0fc,
        "s0cc+n0cc=" + s0cc + "+" + n0cc,
        "s0fc+n0t=" + s0fc + "+" + n0t,
        "s0t+n0fc=" + s0t + "+" + n0fc,
        "s1fc+s0fc=" + s1fc + "+" + s0fc,
        "s1cc+s0cc=" + s1cc + "+" + s0cc,
        "s1fc+s0t=" + s1fc + "+" + s0t,
        "s1t+s0fc=" + s1t + "+" + s0fc,
        "s1cc+s0cc+n0cc=" + s1cc + "+" + s0cc + "+" + n0cc,
        "s0cc+n0cc+n1cc=" + s0cc + "+" + n0cc + "+" + n1cc,
        "s0fc+n0fc+d=" + s0fc + "+" + n0fc + "+" + distance,
        "s1fc+s0fc+ds1=" + s1fc + "+" + s0fc + "+" + distance_s1,
        "s1fc+s0_case=" + s1fc + "+" + s0_case + "+" + s0cc,
        "n0fc+s0_case=" + n0fc + "+" + s0_case + "+" + s0cc,
    ]

    # Global Graph-Based MST Guide Features (Nivre & McDonald 2008)
    if ctx.mst_h1 is not None:
        mst_h1 = ctx.mst_h1
        mst_h2 = ctx.mst_h2
        mst_conf = ctx.mst_conf
        max_rc = ctx.mst_max_right_child

        if s0 != NONE and s0 > 0:
            s0_mh1 = mst_h1[s0]
            s0_mh2 = mst_h2[s0]
            s0_mc = mst_conf[s0]
            s0_m_n0 = "H1" if s0_mh1 == n0 else ("H2" if s0_mh2 == n0 else "NO")
            s0_m_s1 = "H1" if s0_mh1 == s1 else ("H2" if s0_mh2 == s1 else "NO")
            s0_m_in_buf = str(s0_mh1 == 0 or (b_pos <= state.n and s0_mh1 >= b_pos))
            s0_has_buf_child = str(b_pos <= state.n and max_rc[s0] >= b_pos)
        else:
            s0_mc = "<NONE>"
            s0_m_n0 = s0_m_s1 = s0_m_in_buf = s0_has_buf_child = "NONE"

        if n0 != NONE and n0 > 0:
            n0_mh1 = mst_h1[n0]
            n0_mh2 = mst_h2[n0]
            n0_mc = mst_conf[n0]
            n0_m_s0 = "H1" if n0_mh1 == s0 else ("H2" if n0_mh2 == s0 else "NO")
        else:
            n0_mc = "<NONE>"
            n0_m_s0 = "NONE"

        f.extend([
            "mst_s0_n0=" + s0_m_n0 + "+" + s0_mc + "+" + s0bt,
            "mst_s0_s1=" + s0_m_s1 + "+" + s0_mc + "+" + s0bt,
            "mst_n0_s0=" + n0_m_s0 + "+" + n0_mc + "+" + n0bt,
            "mst_s0_state=" + s0_m_n0 + "+" + s0_m_s1 + "+" + s0_m_in_buf + "+" + s0_has_buf_child,
            "mst_buf_child=" + s0_has_buf_child + "+" + s0cg + "+" + n0cg,
        ])

    return f


# =====================================================================
# 5. Arc-Hybrid, Arc-Eager & Arc-Standard Transition Parsers
# =====================================================================

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

    def parse(self, words, tags, use_average=False, supertags=None, mst_info=None, upos=None):
        ctx = SentenceContext(words, tags, supertags, self.lexicon, mst_info, upos)
        state = ParserState(len(words) - 1, ctx.words, ctx.is_punct)
        while not state.is_finished():
            features = self.get_features(ctx, state)
            rows = self.model.feature_rows(features)
            scores = self.allowed_scores(self.model.scores(rows, use_average), state)
            best_idx = int(scores.argmax())
            best_val = float(scores[best_idx])
            scores[best_idx] = VERY_LOW
            second_val = float(scores.max())
            margin = max(0.0, best_val - second_val) if second_val > VERY_LOW * 0.5 else 10.0
            move, label = self.move_of(best_idx)
            state.apply(move, label, margin)
        return state.heads, state.labels, state.margins

    def train_one_sentence(
        self,
        words,
        tags,
        gold_heads,
        gold_labels,
        supertags=None,
        mst_info=None,
        upos=None,
        explore_prob=0.90,
        rng=None,
    ):
        ctx = SentenceContext(words, tags, supertags, self.lexicon, mst_info, upos)
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
            augmented[non_gold_mask] += 10.0
            predicted = int(augmented.argmax())

            if predicted != best_correct:
                self.model.update_pa(rows, best_correct, predicted, scores, margin=10.0, cap=1.0)

            model_choice = int(allowed.argmax())
            if rng is not None and rng.random() > explore_prob:
                chosen = best_correct
            else:
                chosen = model_choice
            move, label = self.move_of(chosen)
            state.apply(move, label)
        self.model.next_step()

    def train_one_epoch(
        self,
        sentences,
        tags_list,
        random_generator,
        supertags_list=None,
        mst_list=None,
        upos_list=None,
        epoch=0,
    ):
        order = list(range(len(sentences)))
        random_generator.shuffle(order)
        explore_prob = 0.50 if epoch == 0 else 0.90
        for k in order:
            sentence = sentences[k]
            stags = supertags_list[k] if supertags_list is not None else None
            minfo = mst_list[k] if mst_list is not None else None
            utags = upos_list[k] if upos_list is not None else None
            self.train_one_sentence(
                sentence["words"],
                tags_list[k],
                sentence["heads"],
                sentence["labels"],
                stags,
                minfo,
                utags,
                explore_prob=explore_prob,
                rng=random_generator,
            )


class ArcEagerState(ParserState):
    """
    Arc-Eager state (Nivre 2003; Goldberg & Nivre 2012).
    ROOT (0) starts at the bottom of the stack; buffer runs 1..n.
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

    def apply(self, move, label, margin=1.0):
        if move == SHIFT:
            self.stack.append(self.buffer_position)
            self.buffer_position += 1
        elif move == REDUCE:
            s0 = self.stack.pop()
            if self.heads[s0] == NONE:
                head = self.stack[-1] if self.stack else 0
                self.add_arc(head, s0, "dep" if head != 0 else "root", margin=0.2)
        elif move == LEFT:
            dependent = self.stack.pop()
            self.add_arc(self.buffer_position, dependent, label, margin)
        else:
            dependent = self.buffer_position
            self.buffer_position += 1
            self.add_arc(self.stack[-1], dependent, label, margin)
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
        s0 = state.stack_word(0)
        s1 = state.stack_word(1)
        n0 = state.buffer_word(0)
        n0t = "<NONE>" if n0 == NONE else ctx.tags[n0]
        s0t = "<NONE>" if s0 == NONE else ctx.tags[s0]

        if s0 != NONE and state.heads[s0] != NONE:
            h0 = state.heads[s0]
            h0w = ctx.words[h0]
            h0t = ctx.tags[h0]
            h0u = ctx.upos[h0]
            s0l = state.labels[s0]
        else:
            h0w = h0t = h0u = s0l = "<UNATTACHED>"

        if s1 != NONE and state.heads[s1] != NONE:
            s1l = state.labels[s1]
        else:
            s1l = "<UNATTACHED>"

        f.extend([
            "ae_s0l=" + s0l,
            "ae_s0l+s0t=" + s0l + "+" + s0t,
            "ae_s0l+n0t=" + s0l + "+" + n0t,
            "ae_h0t+s0l+n0t=" + h0t + "+" + s0l + "+" + n0t,
            "ae_h0u+s0l+n0t=" + h0u + "+" + s0l + "+" + n0t,
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

    def parse(self, words, tags, use_average=False, supertags=None, mst_info=None, upos=None):
        ctx = SentenceContext(words, tags, supertags, self.lexicon, mst_info, upos)
        state = ArcEagerState(len(words) - 1, ctx.words, ctx.is_punct)
        while not state.is_finished():
            features = self.get_features(ctx, state)
            rows = self.model.feature_rows(features)
            scores = self.allowed_scores(self.model.scores(rows, use_average), state)
            best_idx = int(scores.argmax())
            best_val = float(scores[best_idx])
            scores[best_idx] = VERY_LOW
            second_val = float(scores.max())
            margin = max(0.0, best_val - second_val) if second_val > VERY_LOW * 0.5 else 10.0
            move, label = self.move_of(best_idx)
            state.apply(move, label, margin)

        root_children = state.right_children[0]
        root_tok = root_children[0] if root_children else 1
        for i in range(1, state.n + 1):
            if state.heads[i] == NONE:
                if not root_children and i == root_tok:
                    state.heads[i] = 0
                    state.labels[i] = "root"
                    state.margins[i] = 0.2
                    root_children.append(i)
                else:
                    state.heads[i] = root_tok if root_tok != i else 0
                    state.labels[i] = base_label(tags[i]) if state.heads[i] != 0 else "root"
                    state.margins[i] = 0.2
        return state.heads, state.labels, state.margins

    def train_one_sentence(
        self,
        words,
        tags,
        gold_heads,
        gold_labels,
        supertags=None,
        mst_info=None,
        upos=None,
        explore_prob=0.90,
        rng=None,
    ):
        ctx = SentenceContext(words, tags, supertags, self.lexicon, mst_info, upos)
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
            augmented[non_gold_mask] += 10.0
            predicted = int(augmented.argmax())

            if predicted != best_correct:
                self.model.update_pa(rows, best_correct, predicted, scores, margin=10.0, cap=1.0)

            model_choice = int(allowed.argmax())
            if rng is not None and rng.random() > explore_prob:
                chosen = best_correct
            else:
                chosen = model_choice
            move, label = self.move_of(chosen)
            state.apply(move, label)
        self.model.next_step()

    def train_one_epoch(
        self,
        sentences,
        tags_list,
        random_generator,
        supertags_list=None,
        mst_list=None,
        upos_list=None,
        epoch=0,
    ):
        order = list(range(len(sentences)))
        random_generator.shuffle(order)
        explore_prob = 0.50 if epoch == 0 else 0.90
        for k in order:
            sentence = sentences[k]
            stags = supertags_list[k] if supertags_list is not None else None
            minfo = mst_list[k] if mst_list is not None else None
            utags = upos_list[k] if upos_list is not None else None
            self.train_one_sentence(
                sentence["words"],
                tags_list[k],
                sentence["heads"],
                sentence["labels"],
                stags,
                minfo,
                utags,
                explore_prob=explore_prob,
                rng=random_generator,
            )


class ArcStandardState(ParserState):
    """
    Arc-Standard state (Nivre 2004).
    ROOT (0) starts at the bottom of the stack; buffer runs 1..n.
    Moves:
      SHIFT        : push b0 onto stack
      LEFT(label)  : attach s0 -> s1 (head=s0, dep=s1), pop s1
      RIGHT(label) : attach s1 -> s0 (head=s1, dep=s0), pop s0
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

    def can_left(self):
        if len(self.stack) < 2:
            return False
        s1 = self.stack[-2]
        s0 = self.stack[-1]
        if s1 == 0:
            return False
        if self.is_punct is not None and self.is_punct[s0] and not self.is_punct[s1]:
            if self.can_shift() or len(self.stack) > 2:
                return False
        return True

    def can_right(self):
        if len(self.stack) < 2:
            return False
        s1 = self.stack[-2]
        s0 = self.stack[-1]
        if s1 == 0:
            return self.buffer_position > self.n and len(self.stack) == 2
        if self.is_punct is not None and self.is_punct[s1] and not self.is_punct[s0]:
            if self.can_shift():
                return False
        return True

    def apply(self, move, label, margin=1.0):
        if move == SHIFT:
            self.stack.append(self.buffer_position)
            self.buffer_position += 1
        elif move == LEFT:
            s0 = self.stack[-1]
            s1 = self.stack[-2]
            del self.stack[-2]
            self.add_arc(s0, s1, label, margin)
        else:
            s0 = self.stack.pop()
            s1 = self.stack[-1]
            self.add_arc(s1, s0, label, margin)


def standard_move_costs(state, gold_heads, gold_children):
    """Oracle for Arc-Standard (Nivre 2004)."""
    s0 = state.stack_word(0)
    s1 = state.stack_word(1)
    stack_set = set(state.stack)

    left_cost = 1000
    if state.can_left():
        left_cost = 0
        if gold_heads[s1] != s0 and (
            gold_heads[s1] in stack_set or state.in_buffer(gold_heads[s1])
        ):
            left_cost += 1
        for c in gold_children[s1]:
            if c in stack_set or state.in_buffer(c):
                left_cost += 1

    right_cost = 1000
    if state.can_right():
        right_cost = 0
        if gold_heads[s0] != s1 and (
            gold_heads[s0] in stack_set or state.in_buffer(gold_heads[s0])
        ):
            right_cost += 1
        for c in gold_children[s0]:
            if c in stack_set or state.in_buffer(c):
                right_cost += 1

    shift_cost = 1000
    if state.can_shift():
        shift_cost = 0
        if state.can_left() and left_cost == 0 and gold_heads[s1] == s0:
            shift_cost += 1
        if state.can_right() and right_cost == 0 and gold_heads[s0] == s1:
            shift_cost += 1

    return [shift_cost, left_cost, right_cost]


class ArcStandardParser(Parser):
    """
    Arc-Standard Transition Parser (bottom-up stack-stack s1/s0 attachments
    with buffer lookahead n0).
    """

    def get_features(self, ctx, state):
        f = _extract_shared_features(ctx, state)
        s0 = state.stack_word(0)
        s1 = state.stack_word(1)
        n0 = state.buffer_word(0)
        s1_case = state.case_word[s1] if s1 != NONE else "<NONE>"
        s1_mark = state.mark_word[s1] if s1 != NONE else "<NONE>"
        s0w = "<NONE>" if s0 == NONE else ctx.words[s0]
        s0st = "<NONE>" if s0 == NONE else ctx.stems[s0]
        s0u = "<NONE>" if s0 == NONE else ctx.upos[s0]
        s1t = "<NONE>" if s1 == NONE else ctx.tags[s1]
        s1u = "<NONE>" if s1 == NONE else ctx.upos[s1]
        n0t = "<NONE>" if n0 == NONE else ctx.tags[n0]

        if ctx.mst_h1 is not None and s1 != NONE and s1 > 0:
            s1_m_s0 = "H1" if ctx.mst_h1[s1] == s0 else ("H2" if ctx.mst_h2[s1] == s0 else "NO")
        else:
            s1_m_s0 = "NONE"

        f.extend([
            "as_s1case+s0w=" + s1_case + "+" + s0w + "+" + s0u,
            "as_s1case+s0st=" + s1_case + "+" + s0st + "+" + s1u,
            "as_s1mark+s0st=" + s1_mark + "+" + s0st + "+" + s0u,
            "as_mst_s1_s0=" + s1_m_s0 + "+" + s1t + "+" + s0u + "+" + n0t,
        ])
        return f

    def gold_scores(self, scores, state, gold_heads, gold_labels, gold_children):
        L = self.n_labels
        costs = standard_move_costs(state, gold_heads, gold_children)
        best_cost = min(costs)
        result = np.full(len(scores), VERY_LOW, dtype=np.float32)

        if costs[SHIFT] == best_cost:
            result[0] = scores[0]

        s0 = state.stack_word(0)
        s1 = state.stack_word(1)
        if costs[LEFT] == best_cost:
            if s1 > 0 and gold_heads[s1] == s0:
                k = self.class_of(LEFT, self.label_number[gold_labels[s1]])
                result[k] = scores[k]
            else:
                result[1:1 + L] = scores[1:1 + L]

        if costs[RIGHT] == best_cost:
            if s0 > 0 and gold_heads[s0] == s1:
                k = self.class_of(RIGHT, self.label_number[gold_labels[s0]])
                result[k] = scores[k]
            else:
                result[1 + L:] = scores[1 + L:]
        return result

    def parse(self, words, tags, use_average=False, supertags=None, mst_info=None, upos=None):
        ctx = SentenceContext(words, tags, supertags, self.lexicon, mst_info, upos)
        state = ArcStandardState(len(words) - 1, ctx.words, ctx.is_punct)
        while not state.is_finished():
            features = self.get_features(ctx, state)
            rows = self.model.feature_rows(features)
            scores = self.allowed_scores(self.model.scores(rows, use_average), state)
            best_idx = int(scores.argmax())
            best_val = float(scores[best_idx])
            scores[best_idx] = VERY_LOW
            second_val = float(scores.max())
            margin = max(0.0, best_val - second_val) if second_val > VERY_LOW * 0.5 else 10.0
            move, label = self.move_of(best_idx)
            state.apply(move, label, margin)
        return state.heads, state.labels, state.margins

    def train_one_sentence(
        self,
        words,
        tags,
        gold_heads,
        gold_labels,
        supertags=None,
        mst_info=None,
        upos=None,
        explore_prob=0.90,
        rng=None,
    ):
        ctx = SentenceContext(words, tags, supertags, self.lexicon, mst_info, upos)
        n = len(words) - 1

        gold_children = [[] for _ in range(n + 1)]
        for i in range(1, n + 1):
            gold_children[gold_heads[i]].append(i)

        state = ArcStandardState(n, ctx.words, ctx.is_punct)
        while not state.is_finished():
            features = self.get_features(ctx, state)
            rows = self.model.feature_rows(features)
            scores = self.model.scores(rows)

            allowed = self.allowed_scores(scores, state)
            gold = self.gold_scores(scores, state, gold_heads, gold_labels, gold_children)
            best_correct = int(gold.argmax())

            augmented = allowed.copy()
            non_gold_mask = gold <= (VERY_LOW * 0.5)
            augmented[non_gold_mask] += 10.0
            predicted = int(augmented.argmax())

            if predicted != best_correct:
                self.model.update_pa(rows, best_correct, predicted, scores, margin=10.0, cap=1.0)

            model_choice = int(allowed.argmax())
            if rng is not None and rng.random() > explore_prob:
                chosen = best_correct
            else:
                chosen = model_choice
            move, label = self.move_of(chosen)
            state.apply(move, label)
        self.model.next_step()


# =====================================================================
# 6. Second-Order Graph Tree Scorer (Grandparent + Sibling + Valency + Crossing)
# =====================================================================

def _is_ancestor(heads, u, v):
    """Return True if node u is an ancestor of node v in tree `heads`."""
    curr = v
    while curr > 0:
        if curr == u:
            return True
        curr = heads[curr]
    return False


def _count_crossings_of_edge(heads, n, h, m):
    """Count edges (hx, x) in `heads` (x != m) that strictly cross (h, m)."""
    if h <= 0 or m <= 0:
        return 0
    lo = h if h < m else m
    hi = m if h < m else h
    cnt = 0
    for x in range(1, n + 1):
        if x == m:
            continue
        hx = heads[x]
        if hx <= 0:
            continue
        xlo = hx if hx < x else x
        xhi = x if hx < x else hx
        if (xlo < lo < xhi < hi) or (lo < xlo < hi < xhi):
            cnt += 1
    return cnt


def _total_crossing_pairs(heads, n):
    """Total number of unordered crossing edge pairs in tree `heads`."""
    total = 0
    for m in range(1, n + 1):
        total += _count_crossings_of_edge(heads, n, heads[m], m)
    return total // 2


class SecondOrderTreeModel:
    """
    Second-Order Graph-Based Structured Perceptron (McDonald & Pereira 2006)
    scoring full dependency trees on:
      - Grandparent triples (g -> h -> m)
      - Consecutive sibling triples (h, s1, s2) on left and right sides
      - Head valency (left & right child counts)
      - Crossing-arc (non-projectivity) count
    Trained with 1-best MIRA / PA-I and exact coordinate-ascent hill-climbing.
    """

    def __init__(self, lexicon=None, n_rows=2 ** 21):
        self.lexicon = lexicon
        self.n_rows = n_rows
        self.mask = n_rows - 1
        self.weights = np.zeros(n_rows, dtype=np.float32)
        self.totals = np.zeros(n_rows, dtype=np.float64)
        self.timestamps = np.zeros(n_rows, dtype=np.int32)
        self.cross_weight = 0.0
        self.cross_total = 0.0
        self.cross_ts = 0
        self.step = 0

    def _hash_rows(self, features):
        mask = self.mask
        return np.fromiter((hash(f) & mask for f in features), dtype=np.int32, count=len(features))

    def _score_feats(self, features):
        if not features:
            return 0.0
        rows = self._hash_rows(features)
        return float(self.weights[rows].sum())

    def _gp_features(self, ctx, g, h, m):
        mu = ctx.upos[m]
        mt = ctx.tags[m]
        mcg = ctx.coarse_tags[m]
        mw = ctx.words[m]
        mfc = ctx.fine_clu[m]
        if h <= 0:
            return [
                "gp0_u=" + mu,
                "gp0_t=" + mt,
                "gp0_w=" + mw,
                "gp0_fc=" + mfc,
            ]
        hu = ctx.upos[h]
        ht = ctx.tags[h]
        hcg = ctx.coarse_tags[h]
        hw = ctx.words[h]
        hfc = ctx.fine_clu[h]
        gu = "<ROOT>" if g <= 0 else ctx.upos[g]
        gt = "<ROOT>" if g <= 0 else ctx.tags[g]
        gcg = "<ROOT>" if g <= 0 else ctx.coarse_tags[g]
        gfc = "FROOT" if g <= 0 else ctx.fine_clu[g]
        dr_gh = "R0" if g <= 0 else ("R" if h > g else "L")
        dr_hm = "R" if m > h else "L"
        d_hm = distance_text(h, m)
        return [
            "gp_u=" + gu + "+" + hu + "+" + mu + "+" + dr_gh + "+" + dr_hm,
            "gp_t=" + gt + "+" + ht + "+" + mt + "+" + dr_gh + "+" + dr_hm,
            "gp_cg=" + gcg + "+" + hcg + "+" + mcg + "+" + dr_gh + "+" + dr_hm,
            "gp_ud=" + gu + "+" + hu + "+" + mu + "+" + dr_hm + "+" + d_hm,
            "gp_hw=" + gu + "+" + hw + "+" + mu + "+" + dr_gh + "+" + dr_hm,
            "gp_mw=" + gu + "+" + hu + "+" + mw + "+" + dr_gh + "+" + dr_hm,
            "gp_fc=" + gfc + "+" + hfc + "+" + mfc + "+" + dr_gh + "+" + dr_hm,
        ]

    def _head_children_features(self, ctx, h, child_list):
        if h <= 0:
            hu = ht = hcg = hw = hfc = hcc = "<ROOT>"
        else:
            hu = ctx.upos[h]
            ht = ctx.tags[h]
            hcg = ctx.coarse_tags[h]
            hw = ctx.words[h]
            hfc = ctx.fine_clu[h]
            hcc = ctx.coarse_clu[h]

        left_c = [c for c in child_list if c < h]
        right_c = [c for c in child_list if c > h]
        nl = str(min(len(left_c), 4))
        nr = str(min(len(right_c), 4))

        feats = [
            "val_u=" + hu + "+" + nl + "+" + nr,
            "val_t=" + ht + "+" + nl + "+" + nr,
            "val_w=" + hw + "+" + nl + "+" + nr,
            "val_fc=" + hfc + "+" + nl + "+" + nr,
        ]

        # Traverse children outward from h on both sides
        for side, seq in (("L", left_c[::-1]), ("R", right_c)):
            prev_c = -1
            for curr_c in seq:
                if prev_c < 0:
                    u1 = t1 = cg1 = cc1 = "<FIRST>"
                    d12 = "0"
                else:
                    u1 = ctx.upos[prev_c]
                    t1 = ctx.tags[prev_c]
                    cg1 = ctx.coarse_tags[prev_c]
                    cc1 = ctx.coarse_clu[prev_c]
                    d12 = distance_text(prev_c, curr_c)
                u2 = ctx.upos[curr_c]
                t2 = ctx.tags[curr_c]
                cg2 = ctx.coarse_tags[curr_c]
                cc2 = ctx.coarse_clu[curr_c]
                feats.append("sib_u=" + hu + "+" + u1 + "+" + u2 + "+" + side)
                feats.append("sib_t=" + ht + "+" + t1 + "+" + t2 + "+" + side)
                feats.append("sib_cg=" + hcg + "+" + cg1 + "+" + cg2 + "+" + side)
                feats.append("sib_w=" + hw + "+" + u1 + "+" + u2 + "+" + side)
                feats.append("sib_cc=" + hcc + "+" + cc1 + "+" + cc2 + "+" + side)
                feats.append("sib_d=" + hu + "+" + cg1 + "+" + cg2 + "+" + d12 + "+" + side)
                prev_c = curr_c
        return feats

    def tree_features(self, ctx, heads):
        n = ctx.n
        children = [[] for _ in range(n + 1)]
        for m in range(1, n + 1):
            h = heads[m]
            if 0 <= h <= n:
                children[h].append(m)
        feats = []
        for m in range(1, n + 1):
            h = heads[m]
            g = heads[h] if 1 <= h <= n else -1
            feats.extend(self._gp_features(ctx, g, h, m))
        for h in range(0, n + 1):
            feats.extend(self._head_children_features(ctx, h, children[h]))
        return feats

    def score_tree(self, ctx, heads):
        feats = self.tree_features(ctx, heads)
        cross = _total_crossing_pairs(heads, ctx.n)
        return self._score_feats(feats) + self.cross_weight * cross

    def edge_swap_delta(self, ctx, heads, children, m, h_new):
        """
        Exact score difference `score_tree(new) - score_tree(old)` when changing
        only `heads[m]` from `h_old` to `h_new` (both > 0, no cycle).
        """
        h_old = heads[m]
        if h_old == h_new:
            return 0.0
        g_old = heads[h_old] if h_old > 0 else -1
        g_new = heads[h_new] if h_new > 0 else -1

        old_feats = self._gp_features(ctx, g_old, h_old, m)
        new_feats = self._gp_features(ctx, g_new, h_new, m)
        for c in children[m]:
            old_feats.extend(self._gp_features(ctx, h_old, m, c))
            new_feats.extend(self._gp_features(ctx, h_new, m, c))

        old_ch_old = children[h_old]
        new_ch_old = [x for x in old_ch_old if x != m]
        old_ch_new = children[h_new]
        new_ch_new = sorted(old_ch_new + [m])

        old_feats.extend(self._head_children_features(ctx, h_old, old_ch_old))
        old_feats.extend(self._head_children_features(ctx, h_new, old_ch_new))
        new_feats.extend(self._head_children_features(ctx, h_old, new_ch_old))
        new_feats.extend(self._head_children_features(ctx, h_new, new_ch_new))

        delta = self._score_feats(new_feats) - self._score_feats(old_feats)
        cross_delta = (
            _count_crossings_of_edge(heads, ctx.n, h_new, m)
            - _count_crossings_of_edge(heads, ctx.n, h_old, m)
        )
        return delta + self.cross_weight * cross_delta

    def decode_hill_climb(
        self,
        ctx,
        base_scores,
        cands_per_tok,
        init_heads=None,
        alpha=1.0,
        gold_heads=None,
        margin=0.0,
        n_passes=2,
    ):
        """
        Exact coordinate-ascent tree refinement starting from `init_heads`
        (or Chu-Liu/Edmonds on `base_scores`). Preserves single-root tree invariant.
        """
        n = ctx.n
        if n <= 1:
            return [-1] if n <= 0 else [-1, 0]

        if init_heads is None:
            heads = list(chu_liu_edmonds(base_scores))
        else:
            heads = list(init_heads)

        children = [[] for _ in range(n + 1)]
        for m in range(1, n + 1):
            h = heads[m]
            if 0 <= h <= n:
                children[h].append(m)

        for _ in range(n_passes):
            changed = False
            for m in range(1, n + 1):
                h_old = heads[m]
                cands = cands_per_tok[m]
                if not cands or len(cands) <= 1:
                    continue
                if h_old == 0:
                    continue

                base_old = float(base_scores[h_old, m])
                if gold_heads is not None and h_old != gold_heads[m]:
                    base_old += margin

                best_h = h_old
                best_gain = 1.0e-9
                for h_new in cands:
                    if h_new == h_old or h_new == m or h_new <= 0 or h_new > n:
                        continue
                    if _is_ancestor(heads, m, h_new):
                        continue
                    base_new = float(base_scores[h_new, m])
                    if gold_heads is not None and h_new != gold_heads[m]:
                        base_new += margin
                    so_delta = self.edge_swap_delta(ctx, heads, children, m, h_new)
                    gain = (base_new - base_old) + alpha * so_delta
                    if gain > best_gain:
                        best_gain = gain
                        best_h = h_new

                if best_h != h_old:
                    children[h_old].remove(m)
                    children[best_h].append(m)
                    children[best_h].sort()
                    heads[m] = best_h
                    changed = True

            # Also check valid root swaps: swap current root r with any candidate m having 0 in cands[m]
            curr_root = -1
            for m in range(1, n + 1):
                if heads[m] == 0:
                    curr_root = m
                    break
            if curr_root > 0:
                curr_so = self.score_tree(ctx, heads)
                best_swap = None
                best_gain = 1.0e-9
                for m in range(1, n + 1):
                    if m == curr_root or 0 not in cands_per_tok[m]:
                        continue
                    if heads[m] != curr_root:
                        continue
                    # Swapping root r <-> child m (heads[m]=0, heads[r]=m) is always a valid tree
                    new_heads = list(heads)
                    new_heads[m] = 0
                    new_heads[curr_root] = m
                    b_old = float(base_scores[0, curr_root]) + float(base_scores[curr_root, m])
                    b_new = float(base_scores[0, m]) + float(base_scores[m, curr_root])
                    if gold_heads is not None:
                        if gold_heads[curr_root] != 0:
                            b_old += margin
                        if gold_heads[m] != curr_root:
                            b_old += margin
                        if gold_heads[m] != 0:
                            b_new += margin
                        if gold_heads[curr_root] != m:
                            b_new += margin
                    gain = (b_new - b_old) + alpha * (self.score_tree(ctx, new_heads) - curr_so)
                    if gain > best_gain:
                        best_gain = gain
                        best_swap = (m, new_heads)
                if best_swap is not None:
                    heads = best_swap[1]
                    children = [[] for _ in range(n + 1)]
                    for m in range(1, n + 1):
                        children[heads[m]].append(m)
                    changed = True

            if not changed:
                break
        return heads

    def candidate_deltas_for_tree(self, words, tags, supertags, upos, init_heads, cands_per_tok):
        """
        Return a dict `(m, h) -> local_2nd_order_delta` relative to `init_heads`
        for every candidate head h of word m. Used as a structural meta-feature
        in the learned combiner.
        """
        ctx = SentenceContext(words, tags, supertags, self.lexicon, upos=upos)
        n = ctx.n
        children = [[] for _ in range(n + 1)]
        curr_root = -1
        for m in range(1, n + 1):
            h = init_heads[m]
            if 0 <= h <= n:
                children[h].append(m)
            if h == 0:
                curr_root = m

        base_so = None
        deltas = {}
        for m in range(1, n + 1):
            h_old = init_heads[m]
            for h_new in cands_per_tok[m]:
                if h_new == h_old:
                    deltas[(m, h_new)] = 0.0
                elif h_old == 0 or h_new == 0:
                    if h_new == 0 and curr_root > 0 and init_heads[m] == curr_root:
                        if base_so is None:
                            base_so = self.score_tree(ctx, init_heads)
                        nh = list(init_heads)
                        nh[m] = 0
                        nh[curr_root] = m
                        deltas[(m, h_new)] = self.score_tree(ctx, nh) - base_so
                    else:
                        deltas[(m, h_new)] = 0.0
                elif _is_ancestor(init_heads, m, h_new):
                    deltas[(m, h_new)] = -2.0
                else:
                    deltas[(m, h_new)] = self.edge_swap_delta(ctx, init_heads, children, m, h_new)
        return deltas

    def refine_tree(self, words, tags, supertags, upos, base_scores, cands_per_tok, init_heads=None, alpha=0.25, n_passes=2):
        ctx = SentenceContext(words, tags, supertags, self.lexicon, upos=upos)
        return self.decode_hill_climb(
            ctx, base_scores, cands_per_tok, init_heads=init_heads, alpha=alpha, n_passes=n_passes
        )

    def _update_rows(self, rows, delta):
        elapsed = self.step - self.timestamps[rows]
        self.totals[rows] += elapsed * self.weights[rows]
        self.weights[rows] += np.float32(delta)
        self.timestamps[rows] = self.step

    def _update_cross(self, delta):
        elapsed = self.step - self.cross_ts
        self.cross_total += elapsed * self.cross_weight
        self.cross_weight += float(delta)
        self.cross_ts = self.step

    def average_weights(self):
        elapsed = self.step - self.timestamps
        self.totals += elapsed * self.weights
        self.cross_total += (self.step - self.cross_ts) * self.cross_weight
        if self.step > 0:
            self.weights = (self.totals / self.step).astype(np.float32)
            self.cross_weight = self.cross_total / self.step

    def train(self, sentences, tags_list, supertags_list=None, mst_list=None, upos_list=None, epochs=5, seed=1):
        rng = random.Random(seed)
        order = list(range(len(sentences)))
        edge_margin = 2.0

        for _ in range(epochs):
            rng.shuffle(order)
            for k in order:
                s = sentences[k]
                words = s["words"]
                gold_heads = s["heads"]
                n = len(words) - 1
                if n <= 1:
                    continue
                tags = tags_list[k]
                stags = supertags_list[k] if supertags_list is not None else None
                utags = upos_list[k] if upos_list is not None else None
                ctx = SentenceContext(words, tags, stags, self.lexicon, upos=utags)

                # Candidate heads per token: MST top-3 + gold head + immediate neighbors
                base_scores = np.full((n + 1, n + 1), -1.0e4, dtype=np.float64)
                cands_per_tok = [[] for _ in range(n + 1)]
                minfo = mst_list[k] if mst_list is not None else None
                for m in range(1, n + 1):
                    c_list = []
                    if minfo is not None:
                        for key, sc in (("heads", 2.0), ("heads2", 0.5), ("heads3", 0.0)):
                            if key in minfo:
                                h = minfo[key][m]
                                if 0 <= h <= n and h != m:
                                    if h not in c_list:
                                        c_list.append(h)
                                    if sc > base_scores[h, m]:
                                        base_scores[h, m] = sc
                    gh = gold_heads[m]
                    if gh not in c_list:
                        c_list.append(gh)
                    if base_scores[gh, m] < -1.0e3:
                        base_scores[gh, m] = 0.0
                    cands_per_tok[m] = c_list

                init_heads = minfo["heads"] if minfo is not None else gold_heads
                pred_heads = self.decode_hill_climb(
                    ctx,
                    base_scores,
                    cands_per_tok,
                    init_heads=init_heads,
                    alpha=1.0,
                    gold_heads=gold_heads,
                    margin=edge_margin,
                    n_passes=2,
                )

                n_err = sum(1 for m in range(1, n + 1) if pred_heads[m] != gold_heads[m])
                if n_err > 0:
                    g_feats = self.tree_features(ctx, gold_heads)
                    p_feats = self.tree_features(ctx, pred_heads)
                    g_rows = self._hash_rows(g_feats)
                    p_rows = self._hash_rows(p_feats)
                    g_cross = _total_crossing_pairs(gold_heads, n)
                    p_cross = _total_crossing_pairs(pred_heads, n)

                    g_sc = float(self.weights[g_rows].sum()) + self.cross_weight * g_cross
                    p_sc = float(self.weights[p_rows].sum()) + self.cross_weight * p_cross
                    violation = edge_margin * n_err - (g_sc - p_sc)
                    if violation > 0.0:
                        denom = float(len(g_rows) + len(p_rows) + (g_cross - p_cross) ** 2)
                        tau = violation / max(denom, 1.0)
                        if tau > 0.5:
                            tau = 0.5
                        self._update_rows(g_rows, +tau)
                        self._update_rows(p_rows, -tau)
                        if g_cross != p_cross:
                            self._update_cross(tau * (g_cross - p_cross))
                self.step += 1
        self.average_weights()


# =====================================================================
# 7. Full-Tree Second/Third-Order Dependency Label Classifier
# =====================================================================

class FullTreeLabeler:
    """
    Predicts/refines the dependency relation `label[m]` for each edge (h -> m)
    given the COMPLETE dependency tree `heads[1..n]` and Induced POS `upos[1..n]`.
    """

    def __init__(self, label_names, lexicon=None, n_rows=2 ** 20):
        self.label_names = label_names
        self.label_number = {name: i for i, name in enumerate(label_names)}
        self.lexicon = lexicon
        self.model = AveragedPerceptron(len(label_names), n_rows)

    def _prepare_tree_context(self, words, tags, supertags, heads, upos=None):
        ctx = SentenceContext(words, tags, supertags, self.lexicon, upos=upos)
        n = ctx.n
        children = [[] for _ in range(n + 1)]
        for m in range(1, n + 1):
            h = heads[m]
            if 0 <= h <= n:
                children[h].append(m)

        case_w = ["<NONE>"] * (n + 1)
        mark_w = ["<NONE>"] * (n + 1)
        aux_str = ["<NONE>"] * (n + 1)
        has_cop = [False] * (n + 1)
        has_pass = [False] * (n + 1)
        has_subj = [False] * (n + 1)
        has_obj = [False] * (n + 1)
        has_det = [False] * (n + 1)
        has_expl = [False] * (n + 1)
        child_cg_sig = ["<EMPTY>"] * (n + 1)

        for u in range(1, n + 1):
            c_list = children[u]
            if not c_list:
                continue
            cgs = []
            for c in c_list:
                bt = ctx.base_tags[c]
                cg = ctx.coarse_tags[c]
                cw = ctx.words[c]
                cgs.append(cg)
                if bt == "case":
                    case_w[u] = cw
                elif bt == "mark":
                    mark_w[u] = cw
                elif bt in ("aux", "cop"):
                    aux_str[u] = ctx.tags[c].rstrip("<>") + ":" + cw
                    if bt == "cop":
                        has_cop[u] = True
                    if "pass" in ctx.tags[c]:
                        has_pass[u] = True
                elif bt in ("nsubj", "csubj"):
                    has_subj[u] = True
                    if "pass" in ctx.tags[c]:
                        has_pass[u] = True
                elif bt in ("obj", "iobj"):
                    has_obj[u] = True
                elif bt in ("det", "amod") or "poss" in ctx.tags[c]:
                    has_det[u] = True
                elif bt == "expl":
                    has_expl[u] = True
            child_cg_sig[u] = "+".join(sorted(set(cgs)))

        return (
            ctx,
            children,
            case_w,
            mark_w,
            aux_str,
            has_cop,
            has_pass,
            has_subj,
            has_obj,
            has_det,
            has_expl,
            child_cg_sig,
        )

    def get_features(self, tree_ctx, heads, m):
        (
            ctx,
            children,
            case_w,
            mark_w,
            aux_str,
            has_cop,
            has_pass,
            has_subj,
            has_obj,
            has_det,
            has_expl,
            child_cg_sig,
        ) = tree_ctx

        h = heads[m]
        if h <= 0:
            return ["root_only"]

        g = heads[h] if 1 <= h <= ctx.n else -1
        mw = ctx.words[m]
        mst = ctx.stems[m]
        mt = ctx.tags[m]
        mbt = ctx.base_tags[m]
        mcg = ctx.coarse_tags[m]
        mu = ctx.upos[m]
        msig = ctx.sigs[m]
        mdom = ctx.dom_tags[m]
        mamb = ctx.amb_tags[m]
        msup = ctx.supertags[m]
        mfc = ctx.fine_clu[m]
        mcc = ctx.coarse_clu[m]

        hw = ctx.words[h]
        hst = ctx.stems[h]
        ht = ctx.tags[h]
        hbt = ctx.base_tags[h]
        hcg = ctx.coarse_tags[h]
        hu = ctx.upos[h]
        hsig = ctx.sigs[h]
        hfc = ctx.fine_clu[h]
        hcc = ctx.coarse_clu[h]

        gt = "<ROOT>" if g <= 0 else ctx.tags[g]
        gcg = "<ROOT>" if g <= 0 else ctx.coarse_tags[g]
        gu = "<ROOT>" if g <= 0 else ctx.upos[g]

        dr = "L" if m < h else "R"
        dist = distance_text(h, m)
        btw = ctx.between_counts(h, m)

        sibs = children[h]
        pos_in_sibs = sibs.index(m) if m in sibs else -1
        lsib = sibs[pos_in_sibs - 1] if pos_in_sibs > 0 else -1
        rsib = sibs[pos_in_sibs + 1] if (0 <= pos_in_sibs < len(sibs) - 1) else -1
        lsib_t = "<NONE>" if lsib < 0 else ctx.tags[lsib]
        rsib_t = "<NONE>" if rsib < 0 else ctx.tags[rsib]
        lsib_u = "<NONE>" if lsib < 0 else ctx.upos[lsib]
        lsib_w = "<NONE>" if lsib < 0 else ctx.words[lsib]

        m_case = case_w[m]
        m_mark = mark_w[m]
        m_aux = aux_str[m]
        h_case = case_w[h]
        h_aux = aux_str[h]

        h_is_pred = str(hu.startswith("VERB") or hcg == "CLAUSE" or ht == "root" or has_cop[h] or has_subj[h])
        h_is_nom = str((hu in ("NOUN", "PROPN", "PRON", "NUM") or hcg in ("SUBJ", "OBJ", "NMOD", "COMP")) and not has_cop[h] and not has_subj[h])
        m_is_clause = str(mu.startswith("VERB") or mcg == "CLAUSE" or has_cop[m] or has_subj[m] or m_mark != "<NONE>")

        mw_l1 = ctx.words[m - 1] if m > 1 else "<S>"
        mw_r1 = ctx.words[m + 1] if m < ctx.n else "<E>"
        mt_l1 = ctx.tags[m - 1] if m > 1 else "<S>"
        mt_r1 = ctx.tags[m + 1] if m < ctx.n else "<E>"
        mu_l1 = ctx.upos[m - 1] if m > 1 else "<S>"
        mu_r1 = ctx.upos[m + 1] if m < ctx.n else "<E>"

        m_nch = str(min(len(children[m]), 4))
        h_nch = str(min(len(children[h]), 5))
        m_punct_l = mw_l1 if mw_l1 in (",", ":", ";", "(", ")", '"', "--", "-") else "NO"
        m_punct_r = mw_r1 if mw_r1 in (",", ":", ";", "(", ")", '"', "--", "-") else "NO"

        return [
            "bias",
            "mt=" + mt,
            "mbt=" + mbt,
            "mu=" + mu,
            "mu+dr=" + mu + "+" + dr,
            "mbt+dr=" + mbt + "+" + dr,
            "mw+dr=" + mw + "+" + dr,
            "mst+dr=" + mst + "+" + dr,
            "msig+dr=" + msig + "+" + dr,
            "mdom=" + mdom,
            "mamb=" + mamb,
            "msup=" + msup,
            "mfc+dr=" + mfc + "+" + dr,
            "mfc+hfc+dr=" + mfc + "+" + hfc + "+" + dr,
            "mcc+hcc+dr=" + mcc + "+" + hcc + "+" + dr,
            "mfc+hu+dr=" + mfc + "+" + hu + "+" + dr,
            "mu+hfc+dr=" + mu + "+" + hfc + "+" + dr,
            "mt+ht+dr=" + mt + "+" + ht + "+" + dr,
            "mu+hu+dr=" + mu + "+" + hu + "+" + dr,
            "mu+hu+dist=" + mu + "+" + hu + "+" + dr + "+" + dist,
            "mbt+hbt+dr=" + mbt + "+" + hbt + "+" + dr,
            "mbt+hu+dr=" + mbt + "+" + hu + "+" + dr,
            "mu+hbt+dr=" + mu + "+" + hbt + "+" + dr,
            "mcg+hcg+dr=" + mcg + "+" + hcg + "+" + dr + "+" + dist,
            "mw+hw+dr=" + mw + "+" + hw + "+" + dr,
            "mst+hst+dr=" + mst + "+" + hst + "+" + dr,
            "mw+ht+dr=" + mw + "+" + ht + "+" + dr,
            "mw+hu+dr=" + mw + "+" + hu + "+" + dr,
            "mt+hw+dr=" + mt + "+" + hw + "+" + dr,
            "mu+hw+dr=" + mu + "+" + hw + "+" + dr,
            "mst+ht+dr=" + mst + "+" + ht + "+" + dr,
            "mt+hst+dr=" + mt + "+" + hst + "+" + dr,
            "mamb+hcg+dr=" + mamb + "+" + hcg + "+" + dr,
            "mamb+hu+dr=" + mamb + "+" + hu + "+" + dr,
            "mt+ht+gt=" + mt + "+" + ht + "+" + gt,
            "mu+hu+gu=" + mu + "+" + hu + "+" + gu,
            "mcg+hcg+gcg=" + mcg + "+" + hcg + "+" + gcg,

            # Full-tree PP (obl vs nmod vs nmod:poss) disambiguation
            "pp_full=" + m_case + "+" + hu + "+" + hbt + "+" + h_is_pred + "+" + h_is_nom + "+" + dr,
            "pp_lex=" + hst + "+" + m_case + "+" + mu + "+" + hu,
            "pp_head_cop=" + m_case + "+" + str(has_cop[h]) + "+" + hu + "+" + hcg,
            "pp_head_case=" + m_case + "+" + h_case + "+" + hu,

            # Full-tree clausal (ccomp vs xcomp vs advcl vs acl vs acl:relcl) disambiguation
            "cl_full=" + m_is_clause + "+" + m_mark + "+" + str(has_subj[m]) + "+" + hu + "+" + hcg + "+" + dr,
            "cl_lex=" + hst + "+" + m_mark + "+" + str(has_subj[m]) + "+" + str(has_obj[h]) + "+" + str(has_expl[h]),
            "cl_aux=" + m_aux + "+" + m_mark + "+" + str(has_subj[m]) + "+" + hu,

            # Full-tree passive & subject/object disambiguation
            "pass_full=" + str(has_pass[h]) + "+" + h_aux + "+" + mu + "+" + mbt + "+" + dr + "+" + m_case,
            "obj_full=" + str(has_obj[h]) + "+" + str(has_cop[h]) + "+" + mu + "+" + mbt + "+" + dr,

            # Pre-nominal (compound vs amod vs flat vs det vs nmod:poss) disambiguation
            "nom_full=" + mu + "+" + hu + "+" + msig + "+" + hsig + "+" + dr + "+" + dist + "+" + str(has_det[m]),
            "nom_sib=" + lsib_u + "+" + mu + "+" + mt + "+" + rsib_t + "+" + hu,
            "nom_neigh=" + mt_l1 + "+" + mt + "+" + mt_r1 + "+" + dr,
            "upos_neigh=" + mu_l1 + "+" + mu + "+" + mu_r1 + "+" + hu + "+" + dr,
            "mw_neigh=" + mw_l1 + "+" + mw + "+" + mw_r1 + "+" + dr,

            # Punctuation enclosure & valency (appos vs parataxis vs conj vs advcl)
            "punct_enc=" + m_punct_l + "+" + m_punct_r + "+" + mu + "+" + hu + "+" + dr,
            "val_lbl=" + m_nch + "+" + h_nch + "+" + mu + "+" + hu + "+" + dr,

            # Children & sibling structure
            "m_ch_cgs=" + child_cg_sig[m] + "+" + mu + "+" + hu,
            "h_ch_cgs=" + child_cg_sig[h] + "+" + mu + "+" + dr,
            "lsib=" + lsib_t + "+" + lsib_w + "+" + mu + "+" + mbt,
            "btw=" + btw + "+" + mu + "+" + hu,
        ]

    def score_node(self, tree_ctx, heads, m, use_average=True):
        features = self.get_features(tree_ctx, heads, m)
        rows = self.model.feature_rows(features)
        return self.model.scores(rows, use_average), rows

    def train(
        self,
        sentences,
        tags_list,
        supertags_list=None,
        mst_list=None,
        upos_list=None,
        epochs=6,
        seed=1,
    ):
        rng = random.Random(seed)
        order = list(range(len(sentences)))
        lbl_margin = 6.0
        for epoch in range(epochs):
            rng.shuffle(order)
            for k in order:
                s = sentences[k]
                words = s["words"]
                gold_heads = s["heads"]
                gold_labels = s["labels"]
                tags = tags_list[k]
                stags = supertags_list[k] if supertags_list is not None else None
                utags = upos_list[k] if upos_list is not None else None

                if mst_list is not None and rng.random() < 0.25:
                    heads = mst_list[k]["heads"]
                else:
                    heads = gold_heads

                tree_ctx = self._prepare_tree_context(words, tags, stags, heads, utags)
                for m in range(1, len(words)):
                    if heads[m] == 0 or heads[m] != gold_heads[m]:
                        continue
                    scores, rows = self.score_node(tree_ctx, heads, m, use_average=False)
                    correct = self.label_number[gold_labels[m]]
                    augmented = scores.copy()
                    augmented += lbl_margin
                    augmented[correct] -= lbl_margin
                    if "root" in self.label_number:
                        augmented[self.label_number["root"]] = VERY_LOW
                    guess = int(augmented.argmax())
                    if guess != correct:
                        self.model.update_pa(rows, correct, guess, scores, margin=lbl_margin, cap=0.5)
                self.model.next_step()
        self.model.average_weights()

    def score_sentence_labels(self, words, tags, supertags, heads, upos=None, second_model=None):
        """
        Return a list of length n + 1 where entry `m` (for m >= 1) is a dict
        mapping `label_name -> score` from the full-tree labeler (optionally
        averaged with `second_model`).
        """
        tree_ctx = self._prepare_tree_context(words, tags, supertags, heads, upos)
        n = len(words) - 1
        out = [None] * (n + 1)
        for m in range(1, n + 1):
            if heads[m] == 0:
                out[m] = {"root": 10.0}
                continue
            scores, _ = self.score_node(tree_ctx, heads, m, use_average=True)
            if second_model is not None:
                scores2, _ = second_model.score_node(tree_ctx, heads, m, use_average=True)
                scores = 0.5 * (scores + scores2)
            out[m] = {
                self.label_names[idx]: float(scores[idx])
                for idx in range(len(self.label_names))
                if self.label_names[idx] != "root"
            }
        return out


# --- Helpers for right-to-left sentence flipping ---

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


def flip_mst_info(mst_info):
    if mst_info is None:
        return None
    n = len(mst_info["heads"]) - 1
    old_h1 = flip_list(mst_info["heads"])
    new_h1 = [old_h1[0]] + [flip_position(old_h1[i], n) for i in range(1, n + 1)]
    old_h2 = flip_list(mst_info["heads2"])
    new_h2 = [old_h2[0]] + [flip_position(old_h2[i], n) for i in range(1, n + 1)]
    res = {
        "heads": new_h1,
        "heads2": new_h2,
        "conf": flip_list(mst_info["conf"]),
        "margins": flip_list(mst_info["margins"]),
    }
    if "heads3" in mst_info:
        old_h3 = flip_list(mst_info["heads3"])
        res["heads3"] = [old_h3[0]] + [flip_position(old_h3[i], n) for i in range(1, n + 1)]
    return res


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


def flip_result(heads, labels, margins=None):
    n = len(heads) - 1
    old_heads = flip_list(heads)
    new_heads = [old_heads[0]]
    for i in range(1, n + 1):
        new_heads.append(flip_position(old_heads[i], n))
    if margins is not None:
        return new_heads, flip_list(labels), flip_list(margins)
    return new_heads, flip_list(labels)
