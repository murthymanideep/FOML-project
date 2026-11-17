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
features, plus the **Chu-Liu/Edmonds** algorithm to build trees.


## How to run

1. Put `train.csv`, `test.csv` and `sample.csv` in the `data/` folder.
2. Install the only library we need:

   ```
   pip install -r requirements.txt
   ```

3. **Make the submission** (trains on all the training data):

   ```
   python train_and_predict.py
   ```

   This writes `outputs/submission.csv` (upload it to Kaggle) and saves all
   parser outputs in `outputs/final_cache.pkl`.

4. **Check the score** on held-out data (trains on 90% of the training data
   and tests on the other 10%):

   ```
   python train_and_predict.py --check
   ```

   It prints the score of every parser, of the vote, and of the vote with
   each rule-based fix on and off. It saves all outputs in
   `outputs/check_cache.pkl`.

5. **Re-vote without training again** (a few minutes instead of hours):

   ```
   python revote.py --check   # compare vote settings on the check data
   python revote.py           # write outputs/submission.csv again
   ```

   `revote.py` only redoes the last step (the vote and the rule-based fixes),
   using the saved outputs from steps 3 and 4.

A full run takes several hours and needs about 2 GB of free memory. The two
saved output files are about 250 MB each.

Options for `train_and_predict.py` (they can be combined with `--check`):

| Option | Meaning |
|--------|---------|
| `--one-parser` | Train only the first transition parser instead of 12. |
| `--no-second-pass` | Do not use the second-pass tagger, the supertagger, the MST guide and the tree labeler. |
| `--no-copy` | Do not copy the answers of sentences seen in training. |

**Repeatable runs.** The perceptrons use Python's `hash()` to find the table
row of each feature. Python changes these numbers in every run unless
`PYTHONHASHSEED` is fixed, so both scripts restart themselves with
`PYTHONHASHSEED=0`. Now two runs of the same code give the same result.


## The files

| File | What it does |
|------|--------------|
| `train_and_predict.py` | The main script. Runs all the steps below. Settings (epochs, the list of parsers, the rule-based fixes) are at the top. |
| `revote.py` | Redoes only the vote from the saved outputs. Its settings are at the top. |
| `data_utils.py` | Reads the csv files, writes the submission, computes the score, word helpers (normalising, stemming, word shapes, the word lexicon, the list of real punctuation). |
| `perceptron.py` | The averaged perceptron used by the taggers, transition parsers and labeler. |
| `tagger.py` | The five taggers. |
| `parser.py` | The graph-based (MST) parser, the three transition parsers and the full-tree labeler. |


## How it works

### Step 1: Five taggers

The data gives us only the words, with no part-of-speech tags. So we first
train taggers that guess extra information for every word:

1. **Word class** (19 classes such as NOUN, VERB, ADP, PUNCT). We make these
   classes ourselves from the training trees: for example a word labelled
   `det` is DET, a word with a `cop` child is a predicate, and so on.
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

### Step 7: Known sentences

If a test sentence has exactly the same words as a training sentence (for
example "Thanks !"), we copy the training answer.


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

The last version still treated symbols like `&`, `$`, `%` and `:)` as
punctuation. All 83 test words of this kind got the label `punct`, so all 83
were wrong. This is now fixed (see Step 6).

Things we tried that did not help enough:

- **Beam search** for the arc-hybrid parser (keeping the 8 best partial
  trees, global structured perceptron). It gave the same score as one normal
  parser but was about 6 times slower.
- **A bigger weight table** for the parser: no change.


## About the submission file

`data/sample.csv` has the columns `ID,Label` and lists the IDs of **both**
the training rows and the test rows. Our submission has one row for every ID
in `sample.csv`, in the same order. Kaggle only scores the test rows, so the
training rows are simply filled with their known answers.
