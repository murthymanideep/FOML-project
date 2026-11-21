"""
Helper functions for reading the data, writing the submission file,
measuring accuracy, and extracting morphological/lexical word properties.

A sentence is stored as a dictionary with these keys:

    "name"    : the sentence id, for example "train_000001"
    "row_ids" : the ID column of each word (list of strings)
    "words"   : the words of the sentence
    "heads"   : the correct head of each word   (only for training data)
    "labels"  : the correct relation of each word (only for training data)

IMPORTANT: position 0 of "words", "heads" and "labels" is a fake word called
"<ROOT>". The real words start at position 1. This way the word at position i
has token_id i, and a head of 0 means "the root", exactly like in the data.
"""

import csv
import os
import sys


ROOT_WORD = "<ROOT>"

# Groups of relation labels (these label names come from the training data).
# They are used to LEARN word groups from the training trees, so the code
# needs no hand-written lists of English words.
NOMINAL_LABELS = {
    "nsubj", "nsubj:pass", "nsubj:outer", "obj", "iobj", "obl", "obl:unmarked",
    "obl:agent", "nmod", "nmod:poss", "nmod:unmarked", "nmod:desc", "expl",
    "appos", "vocative", "dislocated",
}
MODIFIER_LABELS = {
    "det", "det:predet", "amod", "compound", "nmod:poss", "nummod", "flat", "nmod",
}


def is_punct_word(word):
    """Return True if the token is purely punctuation or an empty quote token."""
    if word == ROOT_WORD:
        return False
    if word == "":
        return True
    for character in word:
        if character.isalnum():
            return False
    return True


def find_punct_forms(sentences, min_share=0.98, min_count=3):
    """
    Return the set of word forms that the training data labels "punct"
    almost always (in at least `min_share` of the cases, and seen at least
    `min_count` times).

    We cannot decide this from the characters alone: in this data "&" is
    "cc", "$" heads the number after it, "%" is a noun and ":)" is
    "discourse". Only these forms may be forced to "punct" or blocked from
    being a head.
    """
    total = {}
    punct = {}
    for sentence in sentences:
        for i in range(1, len(sentence["words"])):
            word = sentence["words"][i]
            total[word] = total.get(word, 0) + 1
            if sentence["labels"][i] == "punct":
                punct[word] = punct.get(word, 0) + 1

    forms = set()
    for word in total:
        if total[word] >= min_count and punct.get(word, 0) >= min_share * total[word]:
            forms.add(word)
    return forms


def make_hashing_repeatable():
    """
    The perceptrons use Python's hash() to find the table row of a feature.
    Python gives different hash numbers in every run unless the setting
    PYTHONHASHSEED is fixed, so two runs of the same code gave slightly
    different models. If the setting is not 0, start this same script again
    with PYTHONHASHSEED=0. Call this before anything else.
    """
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.environ["PYTHONHASHSEED"] = "0"
        os.execv(sys.executable, [sys.executable] + sys.argv)


def make_empty_sentence(name):
    sentence = {}
    sentence["name"] = name
    sentence["row_ids"] = []
    sentence["words"] = [ROOT_WORD]
    sentence["heads"] = [-1]
    sentence["labels"] = [ROOT_WORD]
    return sentence


def read_sentences(file_path):
    """Read train.csv or test.csv and group the rows into sentences."""
    sentences = []
    current_sentence = None

    with open(file_path, newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if current_sentence is None or row["sentence_id"] != current_sentence["name"]:
                current_sentence = make_empty_sentence(row["sentence_id"])
                sentences.append(current_sentence)

            expected_token_id = len(current_sentence["words"])
            if int(row["token_id"]) != expected_token_id:
                raise ValueError("Unexpected token_id in row " + row["ID"])

            current_sentence["row_ids"].append(row["ID"])
            current_sentence["words"].append(row["token"])

            if "target" in row:
                parts = row["target"].split("|")
                current_sentence["heads"].append(int(parts[0]))
                current_sentence["labels"].append(parts[1])

    return sentences


def read_sample_ids(file_path):
    """Return the list of IDs (in order) and the column names of the sample file."""
    ids = []
    with open(file_path, newline="", encoding="utf-8") as csv_file:
        reader = csv.reader(csv_file)
        header = next(reader)
        for row in reader:
            ids.append(row[0])
    return ids, header


def write_submission(file_path, header, ids, answer_of_id):
    """Write one line "ID,HEAD|LABEL" for every ID, in the given order."""
    with open(file_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)
        for row_id in ids:
            writer.writerow([row_id, answer_of_id[row_id]])


def accuracy(sentences, predicted_heads, predicted_labels):
    """
    Return (UAS, LAS) as percentages.

    UAS = share of words with the correct head.
    LAS = share of words with the correct head AND the correct label.
          LAS is the score used in the competition.
    """
    total = 0
    head_correct = 0
    both_correct = 0
    for k in range(len(sentences)):
        sentence = sentences[k]
        for i in range(1, len(sentence["words"])):
            total = total + 1
            if predicted_heads[k][i] == sentence["heads"][i]:
                head_correct = head_correct + 1
                if predicted_labels[k][i] == sentence["labels"][i]:
                    both_correct = both_correct + 1
    uas = 100.0 * head_correct / total
    las = 100.0 * both_correct / total
    return uas, las


def normalize_word(word):
    """
    Lower-case a word, remove apostrophes, and replace numbers, years,
    percentages, amounts of money, times, URLs and e-mail addresses by a
    common symbol.

    Why remove apostrophes: in this data most apostrophes were already
    removed ("Bing s rush", "we ve"), but a few are still there ("n't",
    "’s"). Removing all of them makes both spellings the same word, so they
    share what the models learn about them.
    """
    if word == ROOT_WORD:
        return word
    if word == "":
        return "'"
    if all(c in ('"', "'", "`") for c in word):
        return '"'

    low = word.lower()
    without_apostrophes = low.replace("'", "").replace("’", "")
    if without_apostrophes != "":
        low = without_apostrophes

    if low.startswith("http://") or low.startswith("https://") or low.startswith("www."):
        return "<URL>"
    if "@" in low and "." in low and len(low) > 4:
        return "<EMAIL>"
    if word.isdigit() and len(word) == 4 and (word.startswith("18") or word.startswith("19") or word.startswith("20")):
        return "<YEAR>"
    if len(low) >= 2 and low[-1] == "%" and any(c.isdigit() for c in low[:-1]):
        return "<PERCENT>"
    if len(low) >= 2 and low[0] in ("$", "£", "€", "¥") and any(c.isdigit() for c in low[1:]):
        return "<MONEY>"
    stripped = low.replace(",", "").replace(".", "").replace("/", "").replace("-", "").replace(":", "")
    if stripped.isdigit() and any(c.isdigit() for c in low):
        if ":" in low and len(low) <= 5:
            return "<TIME>"
        return "<NUMBER>"
    if word[0].isdigit():
        return "<NUMBER_MIX>"
    return low


STEM_LENGTH = 5


def stem_word(norm_word):
    """
    A very simple stem that needs no knowledge of English: the first
    STEM_LENGTH letters of the word. Words that start the same way usually
    belong together ("computer", "computers" -> "compu"; "policy",
    "policies" -> "polic"), so they can share their statistics.
    Short words, numbers and special symbols stay as they are.
    """
    if len(norm_word) <= STEM_LENGTH or norm_word[0] == "<" or not norm_word.isalpha():
        return norm_word
    return norm_word[:STEM_LENGTH]


def word_shape(word):
    """
    Describe how a word looks: upper case -> X, lower case -> x, digit -> d,
    other characters stay the same. Repeated symbols are written once.
    """
    if word == "":
        return "'"
    shape = ""
    for character in word:
        if character.isupper():
            symbol = "X"
        elif character.islower():
            symbol = "x"
        elif character.isdigit():
            symbol = "d"
        else:
            symbol = character
        if len(shape) == 0 or shape[-1] != symbol:
            shape = shape + symbol
    return shape


def word_signature(word, position=1):
    """
    How a word looks, for rare or unseen words. Uses only the characters,
    no knowledge of English:
      * capital letters: ALL_CAP ("NASA"), INIT_CAP (capital, first word of
        the sentence), MID_CAP (capital, later in the sentence - often a
        name), MIX_CAP ("iPhone"), LOW (no capitals)
      * digits (DIG) and hyphens (HYP)
      * the last 3 letters ("END_ion", "END_ing", ...) for words of 4 or more
        letters, otherwise the length ("LEN_2"). The models learn from the
        training data what each ending usually means.
    Example: "Mississauga" (not first word) -> "MID_CAP|||END_uga"
    """
    if word == ROOT_WORD:
        return "ROOT"
    if word == "" or is_punct_word(word):
        return "PUNCT"

    has_letter = False
    has_upper = False
    for character in word:
        if character.isalpha():
            has_letter = True
        if character.isupper():
            has_upper = True

    if has_letter and word.isupper():
        cap = "ALL_CAP"
    elif word[0].isupper():
        if position == 1:
            cap = "INIT_CAP"
        else:
            cap = "MID_CAP"
    elif has_upper:
        cap = "MIX_CAP"
    else:
        cap = "LOW"

    has_digit = ""
    for character in word:
        if character.isdigit():
            has_digit = "DIG"
    has_hyphen = ""
    if "-" in word and len(word) > 1:
        has_hyphen = "HYP"

    tail = normalize_word(word)
    if has_hyphen:
        tail = word.lower().rsplit("-", 1)[-1]
    if len(tail) >= 4:
        ending = "END_" + tail[-3:]
    else:
        ending = "LEN_" + str(len(tail))

    return cap + "|" + has_digit + "|" + has_hyphen + "|" + ending


def find_standalone_nominals(sentences, min_count=10):
    """
    LEARN "pronoun-like" words from the training trees: frequent words that
    are mostly used as a nominal (subject, object, possessor, ...) but almost
    never get a determiner, adjective, compound, number or name-part child.

    From this data it learns words like "he", "it", "you", "my", "who",
    "themselves", but also some frequent names ("london", "nasa") and words
    like "today": they all stand alone in a sentence in the same way.
    No hand-made list of English words is used.
    """
    count = {}
    as_nominal = {}
    with_modifier = {}
    for sentence in sentences:
        n = len(sentence["words"]) - 1
        children = []
        for i in range(n + 1):
            children.append([])
        for i in range(1, n + 1):
            children[sentence["heads"][i]].append(i)
        for i in range(1, n + 1):
            word = normalize_word(sentence["words"][i])
            count[word] = count.get(word, 0) + 1
            if sentence["labels"][i] in NOMINAL_LABELS:
                as_nominal[word] = as_nominal.get(word, 0) + 1
            for child in children[i]:
                if sentence["labels"][child] in MODIFIER_LABELS:
                    with_modifier[word] = with_modifier.get(word, 0) + 1
                    break

    words = set()
    for word in count:
        if count[word] < min_count:
            continue
        if as_nominal.get(word, 0) >= 0.6 * count[word] and with_modifier.get(word, 0) <= 0.02 * count[word]:
            words.add(word)
    return words


def find_possessive_markers(sentences, min_count=5):
    """
    LEARN the possessive markers from the training trees: words labelled
    "case" whose head is almost always labelled "nmod:poss" (in this data
    the possessive "s" and the empty apostrophe token).
    """
    as_case = {}
    as_possessive = {}
    for sentence in sentences:
        for i in range(1, len(sentence["words"])):
            if sentence["labels"][i] != "case":
                continue
            word = normalize_word(sentence["words"][i])
            as_case[word] = as_case.get(word, 0) + 1
            head = sentence["heads"][i]
            if head > 0 and sentence["labels"][head] == "nmod:poss":
                as_possessive[word] = as_possessive.get(word, 0) + 1

    markers = set()
    for word in as_case:
        if as_case[word] >= min_count and as_possessive.get(word, 0) >= 0.8 * as_case[word]:
            markers.add(word)
    return markers


def base_label(tag_or_label):
    """Strip direction markers (< or >) and UD subtypes (:pass, :poss, etc.)."""
    if not tag_or_label or tag_or_label in (ROOT_WORD, "<NONE>", "<START>", "<END>"):
        return tag_or_label
    clean = tag_or_label.rstrip("<>")
    return clean.split(":")[0]


def tag_direction(tag):
    """Return '<', '>', or '0' from a directional tag."""
    if not tag:
        return "0"
    if tag[-1] == "<":
        return "<"
    if tag[-1] == ">":
        return ">"
    return "0"


def coarse_group(tag_or_label):
    """Map a UD relation or directional tag into a compact syntactic family."""
    b = base_label(tag_or_label)
    if b in ("nsubj", "csubj"):
        return "SUBJ"
    if b in ("obj", "iobj"):
        return "OBJ"
    if b in ("obl", "nmod", "appos", "vocative", "dislocated"):
        return "NMOD"
    if b in ("compound", "flat", "fixed", "goeswith", "list"):
        return "COMP"
    if b in ("root", "conj", "parataxis", "ccomp", "xcomp", "advcl", "acl"):
        return "CLAUSE"
    if b in ("aux", "cop"):
        return "AUX"
    if b == "det":
        return "DET"
    if b in ("case", "mark"):
        return "ADP"
    if b == "cc":
        return "CC"
    if b in ("amod", "advmod", "nummod", "discourse", "expl"):
        return "MOD"
    if b == "punct":
        return "PUNCT"
    return b


def head_offset_bucket(index, head):
    """Signed relative distance bucket from token index to its head."""
    if head == 0:
        return "ROOT"
    diff = head - index
    if diff == 1:
        return "+1"
    if diff == 2:
        return "+2"
    if diff == 3:
        return "+3"
    if diff >= 4:
        return "+4+"
    if diff == -1:
        return "-1"
    if diff == -2:
        return "-2"
    if diff == -3:
        return "-3"
    return "-4-"


def build_lexicon(sentences):
    """
    Build an empirical word, stem, & morphological signature lexicon from
    training sentences. Also captures left/right neighbor tag context to form
    distributional clusters.
    """
    word_tag_counts = {}
    stem_tag_counts = {}
    sig_tag_counts = {}
    word_totals = {}
    stem_totals = {}
    left_context_counts = {}
    right_context_counts = {}

    for sentence in sentences:
        words = sentence["words"]
        labels = sentence["labels"]
        n = len(words) - 1
        norms = [normalize_word(w) for w in words]
        for i in range(1, n + 1):
            w = norms[i]
            st = stem_word(w)
            t = labels[i]
            word_totals[w] = word_totals.get(w, 0) + 1
            counts = word_tag_counts.get(w)
            if counts is None:
                counts = {}
                word_tag_counts[w] = counts
            counts[t] = counts.get(t, 0) + 1

            stem_totals[st] = stem_totals.get(st, 0) + 1
            st_counts = stem_tag_counts.get(st)
            if st_counts is None:
                st_counts = {}
                stem_tag_counts[st] = st_counts
            st_counts[t] = st_counts.get(t, 0) + 1

            sig = word_signature(words[i], i)
            scounts = sig_tag_counts.get(sig)
            if scounts is None:
                scounts = {}
                sig_tag_counts[sig] = scounts
            scounts[t] = scounts.get(t, 0) + 1

            if i > 1:
                lmap = left_context_counts.get(w)
                if lmap is None:
                    lmap = {}
                    left_context_counts[w] = lmap
                lc = coarse_group(labels[i - 1])
                lmap[lc] = lmap.get(lc, 0) + 1
            if i < n:
                rmap = right_context_counts.get(w)
                if rmap is None:
                    rmap = {}
                    right_context_counts[w] = rmap
                rc = coarse_group(labels[i + 1])
                rmap[rc] = rmap.get(rc, 0) + 1

    dominant_tag = {}
    ambiguity_class = {}
    dominant_coarse = {}
    context_cluster = {}

    for w, counts in word_tag_counts.items():
        total = word_totals[w]
        sorted_tags = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        best_tag = sorted_tags[0][0]
        dominant_tag[w] = best_tag
        dominant_coarse[w] = coarse_group(best_tag)

        if len(sorted_tags) >= 2 and sorted_tags[1][1] >= 0.12 * total:
            ambiguity_class[w] = sorted_tags[0][0] + "/" + sorted_tags[1][0]
        else:
            ambiguity_class[w] = best_tag

        lmap = left_context_counts.get(w)
        best_l = max(lmap.items(), key=lambda kv: kv[1])[0] if lmap else "START"
        rmap = right_context_counts.get(w)
        best_r = max(rmap.items(), key=lambda kv: kv[1])[0] if rmap else "END"
        context_cluster[w] = best_l + "_" + coarse_group(best_tag) + "_" + best_r

    stem_dominant = {}
    stem_ambiguity = {}
    for st, counts in stem_tag_counts.items():
        total = stem_totals[st]
        sorted_tags = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        stem_dominant[st] = sorted_tags[0][0]
        if len(sorted_tags) >= 2 and sorted_tags[1][1] >= 0.15 * total:
            stem_ambiguity[st] = sorted_tags[0][0] + "/" + sorted_tags[1][0]
        else:
            stem_ambiguity[st] = sorted_tags[0][0]

    sig_dominant = {}
    sig_ambiguity = {}
    for sig, counts in sig_tag_counts.items():
        sorted_tags = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        sig_dominant[sig] = sorted_tags[0][0]
        total = sum(counts.values())
        if len(sorted_tags) >= 2 and sorted_tags[1][1] >= 0.18 * total:
            sig_ambiguity[sig] = sorted_tags[0][0] + "/" + sorted_tags[1][0]
        else:
            sig_ambiguity[sig] = sorted_tags[0][0]

    return {
        "word_totals": word_totals,
        "stem_totals": stem_totals,
        "dominant_tag": dominant_tag,
        "ambiguity_class": ambiguity_class,
        "dominant_coarse": dominant_coarse,
        "context_cluster": context_cluster,
        "stem_dominant": stem_dominant,
        "stem_ambiguity": stem_ambiguity,
        "sig_dominant": sig_dominant,
        "sig_ambiguity": sig_ambiguity,
    }


def lookup_word_priors(lexicon, raw_word, normal_word, position):
    """
    Return (prior_tag, ambig_class, coarse_tag, cluster_id, freq_bucket)
    for a word using the training lexicon, with morphological stem and
    signature fallback for rare/unseen words.
    """
    if raw_word == ROOT_WORD:
        return (ROOT_WORD, ROOT_WORD, ROOT_WORD, "ROOT", "HIGH")

    total = lexicon["word_totals"].get(normal_word, 0)
    sig = word_signature(raw_word, position)

    if total >= 2 and not (sig.startswith("MID_CAP") and total <= 3 and not raw_word.islower()):
        dom = lexicon["dominant_tag"][normal_word]
        amb = lexicon["ambiguity_class"][normal_word]
        crs = lexicon["dominant_coarse"][normal_word]
        clu = lexicon["context_cluster"][normal_word]
        if total >= 50:
            fb = "VERY_HIGH"
        elif total >= 10:
            fb = "HIGH"
        elif total >= 4:
            fb = "MID"
        else:
            fb = "LOW"
        return (dom, amb, crs, clu, fb)

    # Fallback 1: check morphological stem if the word is not a mid-sentence proper noun
    st = stem_word(normal_word)
    st_total = lexicon.get("stem_totals", {}).get(st, 0)
    if st != normal_word and st_total >= 4 and not sig.startswith("MID_CAP"):
        dom = lexicon["stem_dominant"][st]
        amb = lexicon["stem_ambiguity"][st]
        crs = coarse_group(dom)
        clu = "STEM_" + crs + "_" + sig.rsplit("|", 1)[-1]
        fb = "STEM_SINGLE" if total == 1 else "STEM_OOV"
        return (dom, amb, crs, clu, fb)

    # Fallback 2: morphological signature
    dom = lexicon["sig_dominant"].get(sig, "UNK")
    amb = lexicon["sig_ambiguity"].get(sig, "UNK")
    crs = coarse_group(dom) if dom != "UNK" else "UNK"
    clu = "SIG_" + sig
    fb = "SINGLE" if total == 1 else "OOV"
    return (dom, amb, crs, clu, fb)
