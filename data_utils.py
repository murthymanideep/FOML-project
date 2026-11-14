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


ROOT_WORD = "<ROOT>"

# Common English function / closed-class word sets for syntactic disambiguation
PASSIVE_AUX_WORDS = {
    "be", "am", "is", "are", "was", "were", "been", "being",
    "get", "gets", "got", "gotten", "getting",
}
POSSESSIVE_PRONOUNS = {
    "my", "your", "his", "her", "its", "our", "their", "whose", "thy",
}
RELATIVE_WORDS = {
    "who", "whom", "whose", "which", "that", "where", "when", "why",
    "whoever", "whomever", "whichever", "whatever",
}
PRT_WORDS = {
    "up", "out", "down", "off", "on", "in", "over", "back", "away",
    "around", "through", "together", "apart", "along", "forward",
    "ahead", "aside", "about", "by", "across", "open", "shut",
}
PUNCT_TOKENS = {
    ".", ",", ":", ";", "!", "?", "-", "--", "---", "(", ")", "[", "]",
    "{", "}", "\"", "'", "''", "``", "/", "\\", "...", "*", "&",
}


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
    """Lower-case a word and normalize numbers, years, ordinals, percentages, URLs."""
    if word == ROOT_WORD:
        return word
    if word == "":
        return "<EMPTY>"
    low = word.lower()
    if low.startswith("http://") or low.startswith("https://") or low.startswith("www."):
        return "<URL>"
    if "@" in low and "." in low and len(low) > 4:
        return "<EMAIL>"
    if word.isdigit() and len(word) == 4 and (word.startswith("18") or word.startswith("19") or word.startswith("20")):
        return "<YEAR>"
    if len(low) >= 3 and low[:-2].isdigit() and low[-2:] in ("st", "nd", "rd", "th"):
        return "<ORDINAL>"
    if len(low) >= 2 and low[-1] == "%" and any(c.isdigit() for c in low[:-1]):
        return "<PERCENT>"
    if len(low) >= 2 and low[0] in ("$", "£", "€", "¥") and any(c.isdigit() for c in low[1:]):
        return "<MONEY>"
    # Pure numbers with optional commas, decimals, slashes, or leading sign
    stripped = low.replace(",", "").replace(".", "").replace("/", "").replace("-", "").replace(":", "")
    if stripped.isdigit() and any(c.isdigit() for c in low):
        if ":" in low and len(low) <= 5:
            return "<TIME>"
        return "<NUMBER>"
    if word[0].isdigit():
        return "<NUMBER_MIX>"
    return low


def word_shape(word):
    """
    Describe how a word looks: upper case -> X, lower case -> x, digit -> d,
    other characters stay the same. Repeated symbols are written once.
    Example: "McDonald's" -> "XxXx'x",  "2026" -> "d",  "U.S." -> "X.X."
    """
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


# Informative English derivational and inflectional suffixes (checked longest first)
_SUFFIX_CLASSES = (
    ("ations", "NOUN_ION"), ("ition", "NOUN_ION"), ("ation", "NOUN_ION"),
    ("ments", "NOUN_MENT"), ("ment", "NOUN_MENT"),
    ("nesses", "NOUN_NESS"), ("ness", "NOUN_NESS"),
    ("ities", "NOUN_ITY"), ("ity", "NOUN_ITY"),
    ("ships", "NOUN_SHIP"), ("ship", "NOUN_SHIP"),
    ("wards", "ADV_WARD"), ("ward", "ADV_WARD"),
    ("fully", "ADV_LY"), ("lessly", "ADV_LY"), ("ably", "ADV_LY"),
    ("ibly", "ADV_LY"), ("ally", "ADV_LY"), ("ively", "ADV_LY"),
    ("ously", "ADV_LY"), ("ly", "ADV_LY"),
    ("able", "ADJ_ABLE"), ("ible", "ADJ_ABLE"),
    ("less", "ADJ_LESS"), ("ful", "ADJ_FUL"),
    ("ious", "ADJ_OUS"), ("eous", "ADJ_OUS"), ("ous", "ADJ_OUS"),
    ("ical", "ADJ_AL"), ("ial", "ADJ_AL"), ("ual", "ADJ_AL"), ("al", "ADJ_AL"),
    ("ive", "ADJ_IVE"), ("ish", "ADJ_ISH"), ("ic", "ADJ_IC"),
    ("est", "ADJ_EST"),
    ("ings", "VERB_ING"), ("ing", "VERB_ING"),
    ("ed", "VERB_ED"), ("en", "VERB_EN"),
    ("izes", "VERB_IZE"), ("ized", "VERB_IZE"), ("ize", "VERB_IZE"),
    ("ises", "VERB_IZE"), ("ised", "VERB_IZE"), ("ise", "VERB_IZE"),
    ("ifies", "VERB_FY"), ("ified", "VERB_FY"), ("ify", "VERB_FY"),
    ("ates", "VERB_ATE"), ("ated", "VERB_ATE"), ("ate", "VERB_ATE"),
    ("ers", "NOUN_ER"), ("ors", "NOUN_ER"), ("er", "ER"), ("or", "OR"),
    ("ists", "NOUN_IST"), ("ist", "NOUN_IST"), ("ism", "NOUN_ISM"),
    ("ion", "NOUN_ION"), ("ies", "PL_IES"), ("es", "PL_ES"), ("s", "PL_S"),
)


def word_signature(word, position=1):
    """
    Morphological & orthographic signature for rare/unseen word generalization.
    Combines capitalization, digit/hyphen properties, and English suffix family.
    """
    if word == ROOT_WORD:
        return "ROOT"
    if not word:
        return "EMPTY"

    # Capitalization
    if word.isupper() and any(c.isalpha() for c in word):
        cap = "ALL_CAP"
    elif word[0].isupper():
        cap = "INIT_CAP" if position == 1 else "MID_CAP"
    elif any(c.isupper() for c in word):
        cap = "MIX_CAP"
    else:
        cap = "LOW"

    has_digit = "DIG" if any(c.isdigit() for c in word) else ""
    has_hyphen = "HYP" if ("-" in word and len(word) > 1) else ""

    low = word.lower()
    # If hyphenated (e.g. "well-known"), inspect the head part after the last hyphen
    tail = low.rsplit("-", 1)[-1] if has_hyphen else low

    suf_class = "NONE"
    if tail in ("'s", "'", "n't", "'re", "'ve", "'ll", "'d", "'m"):
        suf_class = "CLITIC_" + tail
    elif len(tail) >= 3:
        for suf, label in _SUFFIX_CLASSES:
            if tail.endswith(suf) and len(tail) > len(suf) + 1:
                suf_class = label
                break
        if suf_class == "NONE":
            suf_class = "END_" + tail[-2:]
    elif not any(c.isalnum() for c in word):
        suf_class = "PUNCT"

    return cap + "|" + has_digit + "|" + has_hyphen + "|" + suf_class


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
    Build an empirical word & morphological signature lexicon from training sentences.
    Each sentence in `sentences` has `words` and `labels` (the target tags to learn).
    Also captures left/right neighbor tag context to form distributional clusters.
    """
    word_tag_counts = {}
    sig_tag_counts = {}
    word_totals = {}
    left_context_counts = {}
    right_context_counts = {}

    for sentence in sentences:
        words = sentence["words"]
        labels = sentence["labels"]
        n = len(words) - 1
        norms = [normalize_word(w) for w in words]
        for i in range(1, n + 1):
            w = norms[i]
            t = labels[i]
            word_totals[w] = word_totals.get(w, 0) + 1
            counts = word_tag_counts.get(w)
            if counts is None:
                counts = {}
                word_tag_counts[w] = counts
            counts[t] = counts.get(t, 0) + 1

            # For rare words, accumulate morphological signature statistics
            sig = word_signature(words[i], i)
            scounts = sig_tag_counts.get(sig)
            if scounts is None:
                scounts = {}
                sig_tag_counts[sig] = scounts
            scounts[t] = scounts.get(t, 0) + 1

            if i > 1:
                lw = norms[i - 1]
                lmap = left_context_counts.get(w)
                if lmap is None:
                    lmap = {}
                    left_context_counts[w] = lmap
                lc = coarse_group(labels[i - 1])
                lmap[lc] = lmap.get(lc, 0) + 1
            if i < n:
                rw = norms[i + 1]
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

        if len(sorted_tags) >= 2 and sorted_tags[1][1] >= 0.15 * total:
            ambiguity_class[w] = sorted_tags[0][0] + "/" + sorted_tags[1][0]
        else:
            ambiguity_class[w] = best_tag

        lmap = left_context_counts.get(w)
        best_l = max(lmap.items(), key=lambda kv: kv[1])[0] if lmap else "START"
        rmap = right_context_counts.get(w)
        best_r = max(rmap.items(), key=lambda kv: kv[1])[0] if rmap else "END"
        context_cluster[w] = best_l + "_" + coarse_group(best_tag) + "_" + best_r

    sig_dominant = {}
    sig_ambiguity = {}
    for sig, counts in sig_tag_counts.items():
        sorted_tags = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        sig_dominant[sig] = sorted_tags[0][0]
        total = sum(counts.values())
        if len(sorted_tags) >= 2 and sorted_tags[1][1] >= 0.20 * total:
            sig_ambiguity[sig] = sorted_tags[0][0] + "/" + sorted_tags[1][0]
        else:
            sig_ambiguity[sig] = sorted_tags[0][0]

    return {
        "word_totals": word_totals,
        "dominant_tag": dominant_tag,
        "ambiguity_class": ambiguity_class,
        "dominant_coarse": dominant_coarse,
        "context_cluster": context_cluster,
        "sig_dominant": sig_dominant,
        "sig_ambiguity": sig_ambiguity,
    }


def lookup_word_priors(lexicon, raw_word, normal_word, position):
    """
    Return (prior_tag, ambig_class, coarse_tag, cluster_id, freq_bucket)
    for a word using the training lexicon with morphological signature fallback.
    """
    if raw_word == ROOT_WORD:
        return (ROOT_WORD, ROOT_WORD, ROOT_WORD, "ROOT", "HIGH")

    total = lexicon["word_totals"].get(normal_word, 0)
    sig = word_signature(raw_word, position)

    if total >= 2:
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

    # Rare (count == 1) or unseen OOV word: fall back to morphological signature prior
    dom = lexicon["sig_dominant"].get(sig, "UNK")
    amb = lexicon["sig_ambiguity"].get(sig, "UNK")
    crs = coarse_group(dom) if dom != "UNK" else "UNK"
    clu = "SIG_" + sig
    fb = "SINGLE" if total == 1 else "OOV"
    return (dom, amb, crs, clu, fb)
