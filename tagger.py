"""
The tagger: gives every word a rich syntactic tag (and head-offset supertag)
before parsing.

Key capabilities:
  1. Fold-safe lexicon & distributional word priors (dominant tag, ambiguity
     class, coarse syntactic group, context cluster, and morphological
     signature fallback for OOV/rare words and Proper Nouns).
  2. Bidirectional Pass-1 tagging: trains BOTH a Left-to-Right (forward)
     tagger and a Right-to-Left (backward) tagger so every word gets
     predictions conditioned on both left history and right history.
  3. Pass-2 stacked refinement tagger: combines forward Pass-1 tags,
     backward Pass-1 tags, agreement indicators, and lexical context.
  4. Relative head-offset supertagger (`htags`) that predicts signed
     attachment distance buckets and head coarse categories.
"""

import random

from data_utils import (
    ROOT_WORD,
    normalize_word,
    word_shape,
    word_signature,
    base_label,
    tag_direction,
    coarse_group,
    head_offset_bucket,
    build_lexicon,
    lookup_word_priors,
)
from perceptron import AveragedPerceptron


START = "<START>"
END = "<END>"


class Tagger:

    def __init__(self, tag_names, lexicon=None, direction="left-to-right", n_rows=2 ** 20):
        self.tag_names = tag_names
        self.tag_number = {name: k for k, name in enumerate(tag_names)}
        self.lexicon = lexicon
        self.direction = direction
        self.model = AveragedPerceptron(len(tag_names), n_rows)

    def _prepare_sentence(self, words):
        normal_words = [normalize_word(w) for w in words]
        shapes = [word_shape(w) for w in words]
        sigs = [word_signature(words[i], i) for i in range(len(words))]
        if self.lexicon is not None:
            priors = [
                lookup_word_priors(self.lexicon, words[i], normal_words[i], i)
                for i in range(len(words))
            ]
        else:
            priors = [("UNK", "UNK", "UNK", "UNK", "UNK") for _ in words]
        return normal_words, shapes, sigs, priors

    def get_features(
        self,
        words,
        normal_words,
        shapes,
        sigs,
        priors,
        i,
        previous_tag,
        previous_tag_2,
        first_tags=None,
        backward_tags=None,
    ):
        """Features for the word at position i (positions start at 1)."""
        n = len(words) - 1
        word = words[i]
        normal = normal_words[i]
        shape = shapes[i]
        sig = sigs[i]
        dom0, amb0, crs0, clu0, fb0 = priors[i]

        if i - 1 >= 1:
            w_m1 = normal_words[i - 1]
            sh_m1 = shapes[i - 1]
            dom_m1, amb_m1, crs_m1, clu_m1, _ = priors[i - 1]
        else:
            w_m1 = sh_m1 = dom_m1 = amb_m1 = crs_m1 = clu_m1 = START

        if i - 2 >= 1:
            w_m2 = normal_words[i - 2]
            crs_m2 = priors[i - 2][2]
        else:
            w_m2 = crs_m2 = START

        if i - 3 >= 1:
            w_m3 = normal_words[i - 3]
        else:
            w_m3 = START

        if i + 1 <= n:
            w_p1 = normal_words[i + 1]
            sh_p1 = shapes[i + 1]
            dom_p1, amb_p1, crs_p1, clu_p1, _ = priors[i + 1]
        else:
            w_p1 = sh_p1 = dom_p1 = amb_p1 = crs_p1 = clu_p1 = END

        if i + 2 <= n:
            w_p2 = normal_words[i + 2]
            dom_p2, _, crs_p2, _, _ = priors[i + 2]
        else:
            w_p2 = dom_p2 = crs_p2 = END

        if i + 3 <= n:
            w_p3 = normal_words[i + 3]
            crs_p3 = priors[i + 3][2]
        else:
            w_p3 = crs_p3 = END

        prev_c1 = coarse_group(previous_tag)
        prev_c2 = coarse_group(previous_tag_2)

        features = [
            "bias",
            "w=" + normal,
            "raw=" + word,
            "sig=" + sig,
            "shape=" + shape,
            "dom=" + dom0,
            "amb=" + amb0,
            "crs=" + crs0,
            "clu=" + clu0,
            "fb=" + fb0,
            "dom+w=" + dom0 + "+" + normal,
            "amb+shape=" + amb0 + "+" + shape,
            "suf1=" + normal[-1:],
            "suf2=" + normal[-2:],
            "suf3=" + normal[-3:],
            "suf4=" + normal[-4:],
            "suf5=" + normal[-5:],
            "pre1=" + normal[:1],
            "pre2=" + normal[:2],
            "pre3=" + normal[:3],
            "pre4=" + normal[:4],
            "first_word=" + str(i == 1) + "+" + shape[:1],
            "last_word=" + str(i == n) + "+" + shape[:1],
            "t-1=" + previous_tag,
            "t-2=" + previous_tag_2,
            "t-2+t-1=" + previous_tag_2 + "+" + previous_tag,
            "c-2+c-1=" + prev_c2 + "+" + prev_c1,
            "t-1+w=" + previous_tag + "+" + normal,
            "t-1+dom=" + previous_tag + "+" + dom0,
            "t-1+amb=" + previous_tag + "+" + amb0,
            "t-1+sig=" + previous_tag + "+" + sig,
            "w-1=" + w_m1,
            "w-2=" + w_m2,
            "w-3=" + w_m3,
            "w+1=" + w_p1,
            "w+2=" + w_p2,
            "w+3=" + w_p3,
            "w-1+w=" + w_m1 + "+" + normal,
            "w+w+1=" + normal + "+" + w_p1,
            "w-1+w+1=" + w_m1 + "+" + w_p1,
            "w-2+w-1=" + w_m2 + "+" + w_m1,
            "w+1+w+2=" + w_p1 + "+" + w_p2,
            "w-1+w+w+1=" + w_m1 + "+" + normal + "+" + w_p1,
            "w-1suf3=" + w_m1[-3:],
            "w+1suf3=" + w_p1[-3:],
            "sh-1+sh+sh+1=" + sh_m1 + "+" + shape + "+" + sh_p1,
            "dom-1=" + dom_m1,
            "dom+1=" + dom_p1,
            "dom+2=" + dom_p2,
            "amb-1=" + amb_m1,
            "amb+1=" + amb_p1,
            "amb-1+amb0=" + amb_m1 + "+" + amb0,
            "amb0+amb+1=" + amb0 + "+" + amb_p1,
            "crs-1+crs0=" + crs_m1 + "+" + crs0,
            "crs0+crs+1=" + crs0 + "+" + crs_p1,
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
        normal_words, shapes, sigs, priors = self._prepare_sentence(words)
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
                shapes,
                sigs,
                priors,
                i,
                previous_tag,
                previous_tag_2,
                first_tags,
                backward_tags,
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
        f_list = train_part_first_tags if first_tags_list is not None else None
        b_list = train_part_bwd_tags if backward_tags_list is not None else None
        tagger = train_tagger(
            train_part,
            tag_names,
            epochs,
            f_list,
            fold_lexicon,
            direction=direction,
            backward_tags_list=b_list,
        )

        for k in held_out:
            first_tags = first_tags_list[k] if first_tags_list is not None else None
            bwd_tags = backward_tags_list[k] if backward_tags_list is not None else None
            all_tags[k] = tagger.tag(sentences[k]["words"], None, first_tags, None, bwd_tags)
        print("    jackknife fold", fold + 1, "of", n_folds, "done")
    return all_tags


def tag_accuracy(sentences, tags):
    total = 0
    correct = 0
    for k in range(len(sentences)):
        for i in range(1, len(sentences[k]["words"])):
            total += 1
            if tags[k][i] == sentences[k]["labels"][i]:
                correct += 1
    return 100.0 * correct / total


def make_tag_targets(sentences):
    """
    The tags we teach the tagger: the relation of the word plus the side of
    its head: "<" when the head is on the left, ">" when it is on the right.
    The root word gets "root".
    """
    result = []
    for sentence in sentences:
        tags = [sentence["labels"][0]]
        for i in range(1, len(sentence["words"])):
            head = sentence["heads"][i]
            tag = sentence["labels"][i]
            if head != 0 and head < i:
                tag = tag + "<"
            elif head > i:
                tag = tag + ">"
            tags.append(tag)
        new_sentence = {
            "words": sentence["words"],
            "labels": tags,
        }
        result.append(new_sentence)
    return result


def make_supertag_targets(sentences):
    """
    Supertags for each token: signed relative head-distance bucket + coarse
    role of the head (e.g., '+1:NMOD', '-2:CLAUSE', 'ROOT:ROOT').
    Gives the parser explicit lookahead on how far away a token's head lies.
    """
    result = []
    for sentence in sentences:
        words = sentence["words"]
        heads = sentence["heads"]
        labels = sentence["labels"]
        stags = [ROOT_WORD]
        for i in range(1, len(words)):
            h = heads[i]
            dist_b = head_offset_bucket(i, h)
            head_cg = "ROOT" if h == 0 else coarse_group(labels[h])
            stags.append(dist_b + ":" + head_cg)
        result.append({
            "words": words,
            "labels": stags,
        })
    return result
