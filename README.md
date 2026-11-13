# FoML 2026 Hackathon: Dependency Parsing

This project solves the Kaggle competition
[IITH Foundations of Machine Learning 2026 Hackathon](https://www.kaggle.com/competitions/iith-foundations-of-machine-learning-2026-hackathon).

For every word in a sentence we must predict:

- its **head**: the number of the word it depends on (0 means the root), and
- its **relation** to the head, for example `nsubj`, `det` or `obj`.

The answer is written as `HEAD|RELATION`, for example `3|det`. A word is
counted as correct only if both parts are correct.

No neural networks are used. Everything is built from simple, classic
methods: **averaged perceptrons** with hand-made features.


## How to run

1. Put `train.csv`, `test.csv` and `sample.csv` in the `data/` folder.
2. Install the only library we need:

   ```
   pip install -r requirements.txt
   ```

3. Check the score on held-out data (trains on 90% of the training data and
   tests on the other 10%):

   ```
   python train_and_predict.py --check
   ```

4. Make the submission (trains on all the training data):

   ```
   python train_and_predict.py
   ```

   This writes `outputs/submission.csv`. Upload this file to Kaggle.

Each run takes around 15 to 25 minutes on a normal laptop and uses about
1.5 GB of memory.


## The files

| File | What it does |
|------|--------------|
| `train_and_predict.py` | The main script. Runs all the steps below. Settings (number of epochs, number of parsers) are at the top. |
| `data_utils.py` | Reads the csv files, writes the submission, computes the score. |
| `perceptron.py` | The averaged perceptron, the learning algorithm used everywhere. |
| `tagger.py` | The tagger, which guesses a tag for every word before parsing. |
| `parser.py` | The parser, which builds the tree. |


## How it works

### Step 1: The tagger

The data gives us only the words, with no part-of-speech tags (noun, verb,
...). But a parser works much better when it knows what kind of word it is
looking at. So first we train a **tagger** that guesses, for every word:

- its relation (for example `det`), and
- on which side its head is: `<` for left, `>` for right.

For example, in "the dog barks", the tagger should say `det>` for "the",
because "the" is a determiner and its head ("dog") comes after it.

The tagger reads the sentence from left to right. For each word it looks at
the word itself, its last letters, its shape (capital letters, digits, ...),
the nearby words and the tags it already gave to the previous words. It
guesses about 83% of these tags correctly.

**Jackknifing.** On the training data the tagger would be almost perfect,
because it has seen those sentences. On test sentences it makes mistakes.
If the parser only saw perfect tags while training, it would trust them too
much. So we split the training data into 5 parts and tag each part with a
tagger trained on the other 4 parts. Now the training tags have realistic
mistakes too.

### Step 2: The parser

The parser builds the tree step by step (a "transition-based" parser, using
the "arc-hybrid" system). It keeps:

- a **stack** of words that are waiting for their head, and
- a **buffer** of words it has not looked at yet (with a fake ROOT word at the end).

At every step it chooses one move:

- **SHIFT**: move the next buffer word onto the stack.
- **LEFT(label)**: the top stack word gets the next buffer word as its head.
- **RIGHT(label)**: the top stack word gets the word below it in the stack as its head.

After exactly 2 × (number of words) moves, every word has a head and a label.

Choosing the move is a classification problem with 103 classes (SHIFT, plus
LEFT and RIGHT with each of the 51 labels). The perceptron looks at around
90 features: the words and tags on top of the stack and at the front of the
buffer, the children they already have, the labels of those children, the
distance between the words, and so on.

**Training with a dynamic oracle.** For every situation we can compute which
moves still allow the best possible tree (moves with "zero cost"). If the
model picks a costly move, we update the weights. Then we continue with the
move *the model* picked, even if it was wrong. This way the parser also
learns what to do after its own mistakes, which happens all the time on
new sentences.

### Step 3: Voting

We train 5 parsers: 3 read the sentence from left to right and 2 read it
from right to left. They also use different random orders of the training
sentences. Each parser makes slightly different mistakes. For every word we
take the head that most parsers chose, and then the most common label among
the parsers that chose that head.

### The averaged perceptron (`perceptron.py`)

- Each feature (a short text like `s0w=dog`) has one weight for each class.
- The score of a class = the sum of its weights over all features present.
- The prediction is the class with the highest score.
- When the prediction is wrong: add 1 to the correct class, subtract 1 from
  the predicted class (for every feature present).
- At the end we use the **average** of the weights over all of training,
  which works much better than the final weights.

To save memory, the weights live in one big table, and a hash function turns
each feature text into a row number in the table.


## Results

These scores are measured with `--check` (train on 90% of the training data,
test on the other 10%). The competition score is LAS.

| Model | UAS (head correct) | LAS (head and label correct) |
|-------|-----|-----|
| One parser, tags = relation only | 83.4 | 79.3 |
| One parser, tags = relation + head side | 83.4 | 79.6 |
| Vote of 5 parsers (3 left-to-right + 2 right-to-left) | **84.2** | **80.6** |

Single right-to-left parsers are weaker (about 77.3 LAS), but they make
different mistakes, so they still improve the vote.

Things we tried that did not help enough:

- **Beam search** (keeping the 8 best partial trees, with a global
  structured perceptron and "early update"). It reached the same score as
  one normal parser (79.7 LAS) but was about 6 times slower, so we did not
  keep it.


## About the submission file

`data/sample.csv` has the columns `ID,Label` and contains the IDs of **both**
the training rows and the test rows. So our submission does the same: it has
one row for every ID in `sample.csv`, in the same order. The training rows
are also filled with the model's predictions (not copied from the answers).
