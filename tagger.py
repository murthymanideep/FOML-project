"""
Sequence taggers for predicting each word's:
  1. Fine-grained relation + attachment direction (`tag`, e.g. "nsubj>", "obl<", "root")
  2. Relative head-offset + head coarse category (`supertag`, e.g. "+1:OBJ", "-2:CLAUSE", "ROOT:ROOT")

Uses:
  - Fold-specific empirical word/stem/signature priors (`build_lexicon`)
    computed strictly inside each jackknife training fold to prevent leakage.
  - Bidirectional Pass-1 tagging (Left-to-Right and Right-to-Left) with
    global sentence-level clause/root/nominal offset lookahead in Pass-2 and
    Supertagging.
  - Margin-infused structured perceptron updates with scheduled exploration.
"""

import random

from data_utils import (
    normalize_word,
    stem_word,
    word_shape,
    word_signature,
    coarse_group,
    tag_direction,
    head_offset_bucket,
    build_lexicon,
    lookup_word_priors,
)
from perceptron import AveragedPerceptron


START = "<START>"
END = "<END>"


def make_word_tag(sentence, index):
    """Build the tag of word `index`, for example "nsubj>" or "root"."""
    head = sentence["heads"][index]
    label = sentence["labels"][index]
    if head == 0:
        return "root"
    if head < index:
        return label + "<"
    return label + ">"


def make_tag_targets(sentences):
    """Return a copy of the sentences where "labels" are the directional tags."""
    tagged_sentences = []
    for sentence in sentences:
        tags = [sentence["words"][0]]
        for i in range(1, len(sentence["words"])):
            tags.append(make_word_tag(sentence, i))
        new_sentence = {}
        new_sentence["name"] = sentence["name"]
        new_sentence["words"] = sentence["words"]
        new_sentence["heads"] = sentence["heads"]
        new_sentence["labels"] = tags
        tagged_sentences.append(new_sentence)
    return tagged_sentences


def make_supertag_targets(sentences):
    """
    Return a copy of the sentences where "labels" are relative head-offset +
    head coarse syntactic category supertags (e.g., '+1:OBJ', '-2:CLAUSE').
    """
    super_sentences = []
    for sentence in sentences:
        words = sentence["words"]
        heads = sentence["heads"]
        labels = sentence["labels"]
        stags = [words[0]]
        for i in range(1, len(words)):
            h = heads[i]
            offset = head_offset_bucket(i, h)
            h_cg = "ROOT" if h == 0 else coarse_group(labels[h])
            stags.append(offset + ":" + h_cg)
        new_sentence = {}
        new_sentence["name"] = sentence["name"]
        new_sentence["words"] = words
        new_sentence["heads"] = heads
        new_sentence["labels"] = stags
        super_sentences.append(new_sentence)
    return super_sentences


def tag_accuracy(tag_sentences, predicted_tags):
    total = 0
    correct = 0
    for k in range(len(tag_sentences)):
        gold = tag_sentences[k]["labels"]
        pred = predicted_tags[k]
        for i in range(1, len(gold)):
            total += 1
            if gold[i] == pred[i]:
                correct += 1
    return 100.0 * correct / total


def _prepare_global_tag_context(first_tags, backward_tags=None):
    """
    Precompute whole-sentence structural lookahead from Pass-1 tags:
      - relative offset from token i to the nearest predicted `root`
      - relative offset from token i to nearest left/right `CLAUSE`
      - relative offset from token i to nearest left/right `NOMINAL` (`SUBJ`/`OBJ`/`NMOD`)
    """
    n = len(first_tags) - 1
    c_tags = [coarse_group(t) for t in first_tags]

    # Locate predicted roots
    roots = [i for i in range(1, n + 1) if first_tags[i] == "root"]
    if not roots and backward_tags is not None:
        roots = [i for i in range(1, n + 1) if backward_tags[i] == "root"]

    left_cl = [0] * (n + 1)
    left_nom = [0] * (n + 1)
    last_cl = 0
    last_nom = 0
    for i in range(1, n + 1):
        left_cl[i] = last_cl
        left_nom[i] = last_nom
        if c_tags[i] == "CLAUSE" or first_tags[i] == "root":
            last_cl = i
        if c_tags[i] in ("SUBJ", "OBJ", "NMOD", "COMP"):
            last_nom = i

    right_cl = [0] * (n + 1)
    right_nom = [0] * (n + 1)
    next_cl = 0
    next_nom = 0
    for i in range(n, 0, -1):
        right_cl[i] = next_cl
        right_nom[i] = next_nom
        if c_tags[i] == "CLAUSE" or first_tags[i] == "root":
            next_cl = i
        if c_tags[i] in ("SUBJ", "OBJ", "NMOD", "COMP"):
            next_nom = i

    off_root = ["NONE"] * (n + 1)
    off_l_cl = ["NONE"] * (n + 1)
    off_r_cl = ["NONE"] * (n + 1)
    off_l_nom = ["NONE"] * (n + 1)
    off_r_nom = ["NONE"] * (n + 1)

    for i in range(1, n + 1):
        if roots:
            nearest_r = min(roots, key=lambda r: abs(r - i))
            off_root[i] = "SELF" if nearest_r == i else head_offset_bucket(i, nearest_r)
        if left_cl[i] > 0:
            off_l_cl[i] = head_offset_bucket(i, left_cl[i]) + ":" + c_tags[left_cl[i]]
        if right_cl[i] > 0:
            off_r_cl[i] = head_offset_bucket(i, right_cl[i]) + ":" + c_tags[right_cl[i]]
        if left_nom[i] > 0:
            off_l_nom[i] = head_offset_bucket(i, left_nom[i]) + ":" + c_tags[left_nom[i]]
        if right_nom[i] > 0:
            off_r_nom[i] = head_offset_bucket(i, right_nom[i]) + ":" + c_tags[right_nom[i]]

    return (off_root, off_l_cl, off_r_cl, off_l_nom, off_r_nom, str(min(len(roots), 3)))


class Tagger:

    def __init__(self, tag_names, lexicon=None, n_rows=2 ** 20, direction="left-to-right"):
        self.tag_names = tag_names
        self.tag_number = {name: i for i, name in enumerate(tag_names)}
        self.lexicon = lexicon
        self.direction = direction
        self.model = AveragedPerceptron(len(tag_names), n_rows)

    def _prepare_sentence(self, words):
        n = len(words) - 1
        normal_words = [normalize_word(w) for w in words]
        stems = [stem_word(w) for w in normal_words]
        shapes = [word_shape(w) for w in words]
        sigs = [word_signature(words[i], i) for i in range(n + 1)]
        if self.lexicon is not None:
            priors = [
                lookup_word_priors(self.lexicon, words[i], normal_words[i], i)
                for i in range(n + 1)
            ]
        else:
            priors = [("UNK", "UNK", "UNK", "UNK", "OOV")] * (n + 1)
        return normal_words, stems, shapes, sigs, priors

    def get_features(
        self,
        words,
        normal_words,
        stems,
        shapes,
        sigs,
        priors,
        i,
        previous_tag,
        previous_tag_2,
        first_tags=None,
        backward_tags=None,
        global_ctx=None,
    ):
        n = len(words) - 1
        word = words[i]
        normal = normal_words[i]
        stem0 = stems[i]
        shape = shapes[i]
        sig0 = sigs[i]
        dom0, amb0, crs0, clu0, fb0 = priors[i]

        if i > 1:
            w_m1 = normal_words[i - 1]
            st_m1 = stems[i - 1]
            sh_m1 = shapes[i - 1]
            sig_m1 = sigs[i - 1]
            dom_m1, amb_m1, crs_m1, clu_m1, _ = priors[i - 1]
        else:
            w_m1 = st_m1 = sh_m1 = sig_m1 = dom_m1 = amb_m1 = crs_m1 = clu_m1 = START

        if i > 2:
            w_m2 = normal_words[i - 2]
            crs_m2 = priors[i - 2][2]
            dom_m2 = priors[i - 2][0]
        else:
            w_m2 = crs_m2 = dom_m2 = START

        w_m3 = normal_words[i - 3] if i > 3 else START

        if i < n:
            w_p1 = normal_words[i + 1]
            st_p1 = stems[i + 1]
            sh_p1 = shapes[i + 1]
            sig_p1 = sigs[i + 1]
            dom_p1, amb_p1, crs_p1, clu_p1, _ = priors[i + 1]
        else:
            w_p1 = st_p1 = sh_p1 = sig_p1 = dom_p1 = amb_p1 = crs_p1 = clu_p1 = END

        if i + 1 < n:
            w_p2 = normal_words[i + 2]
            crs_p2 = priors[i + 2][2]
            dom_p2 = priors[i + 2][0]
        else:
            w_p2 = crs_p2 = dom_p2 = END

        if i + 2 < n:
            w_p3 = normal_words[i + 3]
            crs_p3 = priors[i + 3][2]
        else:
            w_p3 = crs_p3 = END

        prev_cg = coarse_group(previous_tag)
        prev2_cg = coarse_group(previous_tag_2)

        features = [
            "bias",
            "w=" + normal,
            "raw=" + word,
            "stem=" + stem0,
            "shape=" + shape,
            "sig=" + sig0,
            "p1=" + normal[:1],
            "p2=" + normal[:2],
            "p3=" + normal[:3],
            "p4=" + normal[:4],
            "s1=" + normal[-1:],
            "s2=" + normal[-2:],
            "s3=" + normal[-3:],
            "s4=" + normal[-4:],
            "s5=" + normal[-5:],
            "cap=" + str(word[0].isupper() if word else False),
            "allcap=" + str(word.isupper() if word else False),
            "digit=" + str(any(c.isdigit() for c in word)),
            "hyphen=" + str("-" in word),
            "first_word=" + str(i == 1),
            "last_word=" + str(i == n),

            # Window words, stems, shapes & signatures
            "w-1=" + w_m1,
            "w-2=" + w_m2,
            "w-3=" + w_m3,
            "w+1=" + w_p1,
            "w+2=" + w_p2,
            "w+3=" + w_p3,
            "st-1+st0=" + st_m1 + "+" + stem0,
            "st0+st+1=" + stem0 + "+" + st_p1,
            "sh-1=" + sh_m1,
            "sh+1=" + sh_p1,
            "sig-1=" + sig_m1,
            "sig+1=" + sig_p1,
            "sh-1+sh0+sh+1=" + sh_m1 + "+" + shape + "+" + sh_p1,
            "sig-1+sig0+sig+1=" + sig_m1 + "+" + sig0 + "+" + sig_p1,

            # Word n-grams
            "w-1+w=" + w_m1 + "+" + normal,
            "w+w+1=" + normal + "+" + w_p1,
            "w-1+w+1=" + w_m1 + "+" + w_p1,
            "w-2+w-1=" + w_m2 + "+" + w_m1,
            "w+1+w+2=" + w_p1 + "+" + w_p2,
            "w-2+w-1+w=" + w_m2 + "+" + w_m1 + "+" + normal,
            "w-1+w+w+1=" + w_m1 + "+" + normal + "+" + w_p1,
            "w+w+1+w+2=" + normal + "+" + w_p1 + "+" + w_p2,
            "w+1+w+2+w+3=" + w_p1 + "+" + w_p2 + "+" + w_p3,

            # History transitions
            "t-1=" + previous_tag,
            "t-2=" + previous_tag_2,
            "t-2+t-1=" + previous_tag_2 + "+" + previous_tag,
            "t-1+w=" + previous_tag + "+" + normal,
            "t-1+st=" + previous_tag + "+" + stem0,
            "t-2+t-1+w=" + previous_tag_2 + "+" + previous_tag + "+" + normal,
            "t-1+s3=" + previous_tag + "+" + normal[-3:],
            "t-1+shape=" + previous_tag + "+" + shape,
            "t-1+sig=" + previous_tag + "+" + sig0,
            "cg-1+w=" + prev_cg + "+" + normal,
            "cg-2+cg-1+sig=" + prev2_cg + "+" + prev_cg + "+" + sig0,

            # Empirical lexicon prior features
            "dom0=" + dom0,
            "amb0=" + amb0,
            "crs0=" + crs0,
            "clu0=" + clu0,
            "fb0=" + fb0 + "+" + dom0,
            "dom-1=" + dom_m1,
            "dom+1=" + dom_p1,
            "dom-2=" + dom_m2,
            "dom+2=" + dom_p2,
            "amb-1=" + amb_m1,
            "amb+1=" + amb_p1,
            "amb0+w-1=" + amb0 + "+" + w_m1,
            "amb0+w+1=" + amb0 + "+" + w_p1,
            "amb0+crs-1=" + amb0 + "+" + crs_m1,
            "amb0+crs+1=" + amb0 + "+" + crs_p1,
            "amb0+amb+1=" + amb0 + "+" + amb_p1,
            "amb-1+amb0=" + amb_m1 + "+" + amb0,
            "crs-1+crs0+crs+1=" + crs_m1 + "+" + crs0 + "+" + crs_p1,
            "crs0+crs+1+crs+2=" + crs0 + "+" + crs_p1 + "+" + crs_p2,
            "crs-2+crs-1+crs0=" + crs_m2 + "+" + crs_m1 + "+" + crs0,
            "crs+1+crs+2+crs+3=" + crs_p1 + "+" + crs_p2 + "+" + crs_p3,
            "dom0+dom+1=" + dom0 + "+" + dom_p1,
            "dom-1+dom0=" + dom_m1 + "+" + dom0,
            "dom0+w+1=" + dom0 + "+" + w_p1,
            "w+dom+1=" + normal + "+" + dom_p1,
            "w-1+dom0=" + w_m1 + "+" + dom0,
            "t-1+w+1=" + previous_tag + "+" + w_p1,
            "t-1+w-1=" + previous_tag + "+" + w_m1,
            "t-1+dom+1=" + previous_tag + "+" + dom_p1,
            "t-1+dom-1=" + previous_tag + "+" + dom_m1,
            "t-1+crs0+crs+1=" + previous_tag + "+" + crs0 + "+" + crs_p1,
            "t-1+crs-1+crs0=" + previous_tag + "+" + crs_m1 + "+" + crs0,
            "clu-1+clu0+clu+1=" + clu_m1 + "+" + clu0 + "+" + clu_p1,
            "pos=" + str(min(i, 5)) + "+" + str(min(n - i, 5)),
        ]

        if first_tags is not None:
            p0 = first_tags[i]
            p_l1 = first_tags[i - 1] if i - 1 >= 1 else START
            p_l2 = first_tags[i - 2] if i - 2 >= 1 else START
            p_r1 = first_tags[i + 1] if i + 1 <= n else END
            p_r2 = first_tags[i + 2] if i + 2 <= n else END
            p_r3 = first_tags[i + 3] if i + 3 <= n else END

            c0 = coarse_group(p0)
            c_l1 = coarse_group(p_l1)
            c_r1 = coarse_group(p_r1)
            c_r2 = coarse_group(p_r2)
            d0 = tag_direction(p0)
            d_r1 = tag_direction(p_r1)

            features.extend([
                "p=" + p0,
                "p-1=" + p_l1,
                "p-2=" + p_l2,
                "p+1=" + p_r1,
                "p+2=" + p_r2,
                "p+3=" + p_r3,
                "p+w=" + p0 + "+" + normal,
                "p+st=" + p0 + "+" + stem0,
                "p+1+w=" + p_r1 + "+" + normal,
                "p-1+w=" + p_l1 + "+" + normal,
                "p+w+1=" + p0 + "+" + w_p1,
                "p+p+1=" + p0 + "+" + p_r1,
                "p-1+p=" + p_l1 + "+" + p0,
                "p-1+p+p+1=" + p_l1 + "+" + p0 + "+" + p_r1,
                "p+1+p+2=" + p_r1 + "+" + p_r2,
                "p+p+1+p+2=" + p0 + "+" + p_r1 + "+" + p_r2,
                "p-2+p-1+p=" + p_l2 + "+" + p_l1 + "+" + p0,
                "t-1+p+1=" + previous_tag + "+" + p_r1,
                "t-1+p+p+1=" + previous_tag + "+" + p0 + "+" + p_r1,
                "t-2+t-1+p+1=" + previous_tag_2 + "+" + previous_tag + "+" + p_r1,
                "c-1+c0+c+1=" + c_l1 + "+" + c0 + "+" + c_r1,
                "c0+c+1+c+2=" + c0 + "+" + c_r1 + "+" + c_r2,
                "d0+d+1=" + d0 + "+" + d_r1,
                "p0+amb0=" + p0 + "+" + amb0,
            ])

            if global_ctx is not None:
                off_root, off_l_cl, off_r_cl, off_l_nom, off_r_nom, n_roots = global_ctx
                features.extend([
                    "g_root=" + off_root[i] + "+" + c0,
                    "g_root_p=" + off_root[i] + "+" + p0 + "+" + n_roots,
                    "g_lcl=" + off_l_cl[i] + "+" + c0,
                    "g_rcl=" + off_r_cl[i] + "+" + c0,
                    "g_cl_span=" + off_l_cl[i] + "+" + p0 + "+" + off_r_cl[i],
                    "g_lnom=" + off_l_nom[i] + "+" + c0,
                    "g_rnom=" + off_r_nom[i] + "+" + c0,
                    "g_nom_span=" + off_l_nom[i] + "+" + p0 + "+" + off_r_nom[i],
                ])

            if backward_tags is not None:
                b0 = backward_tags[i]
                b_l1 = backward_tags[i - 1] if i - 1 >= 1 else START
                b_r1 = backward_tags[i + 1] if i + 1 <= n else END
                b_r2 = backward_tags[i + 2] if i + 2 <= n else END
                bc0 = coarse_group(b0)
                bc_r1 = coarse_group(b_r1)
                agree0 = str(p0 == b0)
                agree_r1 = str(p_r1 == b_r1)

                features.extend([
                    "b=" + b0,
                    "b-1=" + b_l1,
                    "b+1=" + b_r1,
                    "b+2=" + b_r2,
                    "p0+b0=" + p0 + "+" + b0,
                    "agree0=" + agree0 + "+" + p0 + "+" + b0,
                    "agree_r1=" + agree_r1 + "+" + p_r1 + "+" + b_r1,
                    "b0+w=" + b0 + "+" + normal,
                    "b-1+b0+b+1=" + b_l1 + "+" + b0 + "+" + b_r1,
                    "b0+b+1+b+2=" + b0 + "+" + b_r1 + "+" + b_r2,
                    "p-1+b0+b+1=" + p_l1 + "+" + b0 + "+" + b_r1,
                    "t-1+b0+b+1=" + previous_tag + "+" + b0 + "+" + b_r1,
                    "c0+bc0+bc_r1=" + c0 + "+" + bc0 + "+" + bc_r1,
                ])
        return features

    def tag(self, words, correct_tags=None, first_tags=None, rng=None, backward_tags=None):
        """
        Tag one sentence. Returns a list of tags, with "<ROOT>" at position 0.
        Supports both left-to-right and right-to-left decoding.
        """
        normal_words, stems, shapes, sigs, priors = self._prepare_sentence(words)
        global_ctx = (
            _prepare_global_tag_context(first_tags, backward_tags)
            if first_tags is not None
            else None
        )
        n = len(words) - 1
        tags = [None] * (n + 1)
        tags[0] = words[0]

        if self.direction == "right-to-left":
            positions = range(n, 0, -1)
            previous_tag = END
            previous_tag_2 = END
        else:
            positions = range(1, n + 1)
            previous_tag = START
            previous_tag_2 = START

        for i in positions:
            features = self.get_features(
                words,
                normal_words,
                stems,
                shapes,
                sigs,
                priors,
                i,
                previous_tag,
                previous_tag_2,
                first_tags,
                backward_tags,
                global_ctx,
            )
            rows = self.model.feature_rows(features)
            scores = self.model.scores(rows)

            if correct_tags is not None:
                correct = self.tag_number[correct_tags[i]]
                augmented = scores.copy()
                augmented += 0.5
                augmented[correct] -= 0.5
                guess = int(augmented.argmax())
                if guess != correct:
                    self.model.update(rows, correct, guess)
                if rng is not None and rng.random() < 0.10:
                    guessed_tag = self.tag_names[int(scores.argmax())]
                else:
                    guessed_tag = correct_tags[i]
            else:
                guess = int(scores.argmax())
                guessed_tag = self.tag_names[guess]

            tags[i] = guessed_tag
            previous_tag_2 = previous_tag
            previous_tag = guessed_tag
        return tags

    def train(self, sentences, epochs, seed=1, first_tags_list=None, backward_tags_list=None):
        """
        Train on sentences; the "labels" of the sentence are the tags to learn.
        """
        random_generator = random.Random(seed)
        order = list(range(len(sentences)))
        for epoch in range(epochs):
            random_generator.shuffle(order)
            for k in order:
                sentence = sentences[k]
                first_tags = first_tags_list[k] if first_tags_list is not None else None
                bwd_tags = backward_tags_list[k] if backward_tags_list is not None else None
                self.tag(
                    sentence["words"],
                    sentence["labels"],
                    first_tags,
                    random_generator,
                    bwd_tags,
                )
                self.model.next_step()
        self.model.average_weights()


def train_tagger(
    sentences,
    tag_names,
    epochs,
    first_tags_list=None,
    lexicon=None,
    direction="left-to-right",
    backward_tags_list=None,
):
    if lexicon is None:
        lexicon = build_lexicon(sentences)
    tagger = Tagger(tag_names, lexicon=lexicon, direction=direction)
    tagger.train(sentences, epochs, 1, first_tags_list, backward_tags_list)
    return tagger


def jackknife_tags(
    sentences,
    tag_names,
    epochs,
    n_folds=5,
    first_tags_list=None,
    direction="left-to-right",
    backward_tags_list=None,
):
    """
    Give tags to the TRAINING sentences using n_folds cross-validation
    so the parser learns with realistic tagger predictions and fold-specific
    lexicons (zero leakage).
    """
    all_tags = [None] * len(sentences)

    for fold in range(n_folds):
        train_part = []
        train_part_first_tags = []
        train_part_bwd_tags = []
        held_out = []
        for k in range(len(sentences)):
            if k % n_folds == fold:
                held_out.append(k)
            else:
                train_part.append(sentences[k])
                if first_tags_list is not None:
                    train_part_first_tags.append(first_tags_list[k])
                if backward_tags_list is not None:
                    train_part_bwd_tags.append(backward_tags_list[k])

        fold_lexicon = build_lexicon(train_part)
        tagger = train_tagger(
            train_part,
            tag_names,
            epochs,
            train_part_first_tags if first_tags_list is not None else None,
            lexicon=fold_lexicon,
            direction=direction,
            backward_tags_list=train_part_bwd_tags if backward_tags_list is not None else None,
        )
        for k in held_out:
            first_tags = first_tags_list[k] if first_tags_list is not None else None
            bwd_tags = backward_tags_list[k] if backward_tags_list is not None else None
            all_tags[k] = tagger.tag(
                sentences[k]["words"],
                first_tags=first_tags,
                backward_tags=bwd_tags,
            )
        print("    jackknife fold", fold + 1, "of", n_folds, "done")
    return all_tags
