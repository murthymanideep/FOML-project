# FoML 2026 Hackathon: Dependency Parsing

This project solves the Kaggle competition
[IITH Foundations of Machine Learning 2026 Hackathon](https://www.kaggle.com/competitions/iith-foundations-of-machine-learning-2026-hackathon).

For every word in a sentence we must predict:

- its **head**: the number of the word it depends on (0 means the root), and
- its **relation** to the head, for example `nsubj`, `det` or `obj`.

The answer is written as `HEAD|RELATION`, for example `3|det`. A word is
counted as correct only if both parts are correct.

No neural networks are used. Everything is built from classic methods:
**averaged perceptrons** and **structured perceptrons** with hand-made
features, **logistic regression**, plus the **Chu-Liu/Edmonds** algorithm to
build trees.

**No NLP libraries and no hand-written word lists.** The code imports only
numpy and Python's built-in modules. It contains no lists of English words
(pronouns, prepositions, ...) and no English grammar rules for word endings.
Everything about words is either learned from `train.csv` or looks only at
the characters (capital letters, digits, the last letters of a word):

| What | How we get it |
|------|---------------|
| Real punctuation | Word forms labelled `punct` at least 98% of the time in the training data |
| Pronoun-like words ("he", "it", "my", "who", ...) | Frequent words that are mostly nominals but (almost) never get a determiner, adjective, compound, number or name-part child in the training trees |
| Possessive markers ("s", "'") | `case` words whose head is (almost) always `nmod:poss` in the training trees |
| Stem of a word | Its first 5 letters ("computers" -> "compu") |
| Word ending | Its last 3 letters ("running" -> "ing"); the models learn from the data what each ending means |
| Different spellings | Apostrophes are removed, so "n't" and "nt" (both in the data) become the same word |


## How to run

1. Put `train.csv`, `test.csv` and `sample.csv` in the `data/` folder.
2. Install the only library we need:

   ```
   pip install -r requirements.txt
   ```

3. **Check run** (trains on 90% of the training data, tests on the other 10%):

   ```
   python train_and_predict.py --check
   ```

   It prints the score of every parser and of the vote. At the end it
   chooses the best set of rule-based fixes, tests the learned combiner
   fairly (Step 5b), prints an error report, and saves these choices in
   **`best_settings.json`**.

4. **Make the submission** (trains on all the training data):

   ```
   python train_and_predict.py
   ```

   This uses the choices in `best_settings.json` (if the file exists) and
   writes `outputs/submission.csv`. Upload this file to Kaggle.

`best_settings.json` is a small text file in the project folder. Commit it
with the code: then anyone who runs `python train_and_predict.py` later (for
example to compute the private score) gets exactly the same choices, without
running `--check` first. Without the file, the run uses `RULES` at the top of
`train_and_predict.py` and the normal vote.

A full run takes several hours and needs about 2 GB of free memory.

Options for `train_and_predict.py` (they can be combined with `--check`):

| Option | Meaning |
|--------|---------|
| `--one-parser` | Train only the first transition parser instead of 12. |
| `--no-second-pass` | Do not use the second-pass tagger, the supertagger, the MST guide and the tree labeler. |

**Repeatable runs.** The perceptrons use Python's `hash()` to find the table
row of each feature. Python changes these numbers in every run unless
`PYTHONHASHSEED` is fixed, so the script restarts itself with
`PYTHONHASHSEED=0`. All random choices also use fixed seeds, so two runs of
the same code give the same result.

**No overfitting to the leaderboard.** We never tune anything on the Kaggle
score. Every choice (the rules, the combiner) is made on our own check data,
which is separate from the test data, and the combiner is only used if it
wins under cross-validation. The test answers are never used, and no answer
is copied from the training data: every test sentence is parsed by the
models, even if the same sentence appears in `train.csv`.

The public score uses only about 6% of the test words (about 1,500 words), so
it can move by about 1 point just by chance. The private score uses the other
94%. Our check data (about 20,000 words) is a much more reliable guide, so we
make all decisions on it.


## The files

| File | What it does |
|------|--------------|
| `train_and_predict.py` | The main script. Runs all the steps below. Settings (epochs, the list of parsers, the rule-based fixes) are at the top. |
| `combiner.py` | The learned combiner (logistic regression written with numpy). |
| `best_settings.json` | Made by `--check`: the chosen rules and the combiner's weights. Commit it with the code. |
| `data_utils.py` | Reads the csv files, writes the submission, computes the score, word helpers (normalising, stems, word shapes, the word lexicon) and the word groups learned from the training data (punctuation, pronoun-like words, possessive markers). |
| `perceptron.py` | The averaged perceptron used by the taggers, transition parsers and labeler. |
| `tagger.py` | The five taggers. |
| `parser.py` | The graph-based (MST) parser, the three transition parsers and the full-tree labeler. |


## How it works

### Step 1: Five taggers

The data gives us only the words, with no part-of-speech tags. So we first
train taggers that guess extra information for every word:

1. **Word class** (18 classes such as NOUN, VERB, ADP, PUNCT). We make these
   classes ourselves from the training trees: for example a word labelled
   `det` is DET, a word with a `cop` child is a predicate, a learned
   pronoun-like word is PRON, and so on.
2. **Relation + side of the head**, read left to right. Example: `det>` means
   "a determiner whose head comes later".
3. The same, read **right to left**.
4. A **second pass** that sees the guesses of 2 and 3 for the whole sentence.
5. A **supertagger** that guesses where the head is (`+1`, `-2`, `ROOT`, ...)
   and what kind of word the head is.

**Jackknifing.** On the training sentences a tagger would be almost perfect,
because it has seen them. On test sentences it makes mistakes. So we split
the training data into 5 parts and tag each part with a tagger trained on the
other 4 parts (each part also builds its own word lexicon). Now the training
tags have realistic mistakes, like the test tags.

### Step 2: A graph-based (MST) parser

This parser gives a score to every possible pair "head → word" (42 feature
templates: words, stems, tags, distance, direction, words in between, ...)
and then builds the best tree with the **Chu-Liu/Edmonds** algorithm. It is
trained as a structured perceptron. Its answers are used twice: as extra
features ("guides") for the transition parsers, and as extra votes. For the
training sentences the guides are again made with jackknifing.

### Step 3: A full-tree labeler

Once a tree is known, this perceptron chooses the relation of each word from
the whole tree: the head, the grandparent, the siblings, and children such
as `case`, `mark`, `aux`, `cop`.

### Step 4: Twelve transition parsers

A transition parser builds the tree step by step with a stack and a buffer.
We use three different systems (**arc-hybrid**, **arc-eager**,
**arc-standard**), some reading left to right and some right to left, with
different random seeds. They are trained with a "dynamic oracle": they also
learn what to do after their own mistakes. They use the tags, supertags and
MST guides as features.

### Step 5: The vote

Every parser votes for the arcs of its tree. A vote counts more for a better
parser and when the parser was more sure. The MST parser adds extra votes.
Then Chu-Liu/Edmonds picks the best single tree from all the votes, so every
sentence gets exactly one valid tree with one root. For the labels, the
parsers vote and the full-tree labeler adds its scores.

### Step 5b: The learned combiner (optional, `combiner.py`)

The vote above uses hand-set numbers (parser weights, MST bonuses, ...).
The learned combiner learns them from data instead, with **logistic
regression** (an allowed linear model, written with numpy):

- For every word, the candidate heads are all heads that some parser
  proposed. Each candidate is described by a few numbers: which parsers
  chose it, how sure they were, what the MST parser said, the distance, ...
  The logistic regression learns the chance that the candidate is right, and
  Chu-Liu/Edmonds builds the best tree from these chances.
- In the same way, a second logistic regression chooses the label among the
  labels proposed by the parsers and the full-tree labeler.

It learns from the check data: those parser outputs come from models that
never saw the check sentences, so they make realistic mistakes. Its score is
measured fairly with 2-fold cross-validation (learn on one half of the check
data, test on the other half). It is only used for the submission if it beats
the normal vote by at least 0.1 points.

### Step 6: Rule-based fixes

Small rules from the Universal Dependencies guidelines (set `RULES` at the
top of `train_and_predict.py`; `--check` measures each one):

- `punct`: real punctuation always gets the label `punct`.
- `flat`: continue multi-word names.
- `passive`: `nsubj` becomes `nsubj:pass` when the verb has `aux:pass`.
- `obl_nmod`: swap `obl` and `nmod` depending on the kind of head.

**What counts as punctuation?** We do not decide this from the characters.
In this data `&` is `cc`, `$` is the head of the number after it, `%` is a
noun and `:)` is `discourse`. A word form counts as punctuation only if the
training data labels it `punct` at least 98% of the time (47 forms such as
`.` `,` `?` `!` `(` `)` `"`).

## Checking our work

`--check` cuts the training sentences into **groups of 100 neighbouring
sentences** and keeps every 10th group for checking. Neighbouring sentences
usually come from the same document, so the check data has documents the
models never saw, just like the real test data.

Before, we kept every 10th single sentence. That was too easy: the rest of
the same document was in the training data. Share of check words never seen
in training:

| Check data | Unknown words |
|------------|---------------|
| every 10th sentence (old) | 3.9% |
| every 10th group of 100 sentences (now) | 5.0% |
| the real test data | 6.3% |

So check scores are still a bit higher than Kaggle scores. Differences
smaller than about 0.2 points are just noise.


## Results on Kaggle

| Version | Kaggle score |
|---------|--------------|
| 5 arc-hybrid parsers, simple vote | 0.7803 |
| + second-pass tagger, 7 parsers | 0.7811 |
| Word classes, supertagger, MST parser, full-tree labeler, 12 parsers of 3 kinds, Chu-Liu/Edmonds vote, rules | 0.8199 |
| + real punctuation decided from the training data (`&`, `$`, `%`, `:)` are not punctuation) | 0.8319 |

The 0.8199 version treated symbols like `&`, `$`, `%` and `:)` as
punctuation. All 83 test words of this kind got the label `punct`, so all 83
were wrong. Fixing this (see Step 6) gave +1.2 points.

Things we tried that did not help enough:

- **Beam search** for the arc-hybrid parser (keeping the 8 best partial
  trees, global structured perceptron). It gave the same score as one normal
  parser but was about 6 times slower.
- **A bigger weight table** for the parser: no change.


## About the submission file

`data/sample.csv` has the columns `ID,Label` and lists the IDs of **both**
the training rows and the test rows. Our submission has one row for every ID
in `sample.csv`, in the same order. Kaggle only scores the test rows (about 6%
of them for the public score, the other 94% for the private score). The
training rows are not scored, so they are simply filled with their answers
from `train.csv`.
