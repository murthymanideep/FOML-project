# helper functions for reading csv files, saving submission, and word features
# position 0 in every sentence is <ROOT>, real words start at index 1

import csv
import os
import sys


ROOT_WORD = "<ROOT>"

# relation label sets used to find standalone and modifier words from train data
NOMINAL_LABELS = {
    "nsubj", "nsubj:pass", "nsubj:outer", "obj", "iobj", "obl", "obl:unmarked",
    "obl:agent", "nmod", "nmod:poss", "nmod:unmarked", "nmod:desc", "expl",
    "appos", "vocative", "dislocated",
}
MODIFIER_LABELS = {
    "det", "det:predet", "amod", "compound", "nmod:poss", "nummod", "flat", "nmod",
}


def is_punct_word(word):
    if word == ROOT_WORD:
        return False
    if word == "":
        return True
    for character in word:
        if character.isalnum():
            return False
    return True


def find_punct_forms(sentences, min_share=0.98, min_count=3):
    # find tokens that are labeled punct almost every time in train data
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


def fix_hash_seed():
    # fix PYTHONHASHSEED=0 so python hash() gives same numbers every run
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
    # read train.csv or test.csv and group tokens by sentence_id
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
    ids = []
    with open(file_path, newline="", encoding="utf-8") as csv_file:
        reader = csv.reader(csv_file)
        header = next(reader)
        for row in reader:
            ids.append(row[0])
    return ids, header


def write_submission(file_path, header, ids, pred_map):
    with open(file_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)
        for row_id in ids:
            writer.writerow([row_id, pred_map[row_id]])


def accuracy(sentences, predicted_heads, predicted_labels):
    # calculate UAS (head accuracy) and LAS (head + label accuracy)
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
    # lowercase word, remove apostrophes, and group numbers/urls/emails/etc
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
    # take first 5 letters as a simple stem
    if len(norm_word) <= STEM_LENGTH or norm_word[0] == "<" or not norm_word.isalpha():
        return norm_word
    return norm_word[:STEM_LENGTH]


def word_shape(word):
    # convert chars to X (upper), x (lower), d (digit) and collapse repeats
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
    # char pattern for rare or unseen words (caps, digits, hyphen, suffix)
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


def get_standalone_words(sentences, min_count=10):
    # find pronoun-like words from train data that rarely take modifiers
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


def get_possessive_words(sentences, min_count=5):
    # find case tokens whose head is usually nmod:poss
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
    # remove < or > and :subtypes from a tag/label
    if not tag_or_label or tag_or_label in (ROOT_WORD, "<NONE>", "<START>", "<END>"):
        return tag_or_label
    clean = tag_or_label.rstrip("<>")
    return clean.split(":")[0]


def tag_direction(tag):
    if not tag:
        return "0"
    if tag[-1] == "<":
        return "<"
    if tag[-1] == ">":
        return ">"
    return "0"


def coarse_group(tag_or_label):
    # group detailed labels into broader categories
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
    # bucketed signed distance from word to its head
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
    # collect word, stem, signature, and neighbor tag stats from train sentences
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


def get_word_info(lexicon, raw_word, normal_word, position):
    # look up most common tag, top-2 tags, group, cluster, and freq bucket for a word
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

    # try stem if word is rare/unseen
    st = stem_word(normal_word)
    st_total = lexicon.get("stem_totals", {}).get(st, 0)
    if st != normal_word and st_total >= 4 and not sig.startswith("MID_CAP"):
        dom = lexicon["stem_dominant"][st]
        amb = lexicon["stem_ambiguity"][st]
        crs = coarse_group(dom)
        clu = "STEM_" + crs + "_" + sig.rsplit("|", 1)[-1]
        fb = "STEM_SINGLE" if total == 1 else "STEM_OOV"
        return (dom, amb, crs, clu, fb)

    # last fallback: use char signature
    dom = lexicon["sig_dominant"].get(sig, "UNK")
    amb = lexicon["sig_ambiguity"].get(sig, "UNK")
    crs = coarse_group(dom) if dom != "UNK" else "UNK"
    clu = "SIG_" + sig
    fb = "SINGLE" if total == 1 else "OOV"
    return (dom, amb, crs, clu, fb)
