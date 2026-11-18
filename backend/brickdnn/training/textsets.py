"""The built-in text datasets: generated sentences of token ids, nothing is downloaded.

Three small classification datasets for networks that read token ids (an Input with data
type ``long`` and an Embedding layer: the LSTM and Transformer examples). Every sample is a
short text of English words (or symbols) from a small vocabulary, written as token ids and
padded to a fixed length; id 0 is the padding ``<pad>``, id 1 ``<unk>`` (a word the
vocabulary does not know, only in texts typed in for predictions).

``sentiment``  Short reviews ("the soup was not very tasty but the staff was friendly"),
               positive or negative. A negation in front of the adjective flips it, and in
               "… but …" the second clause decides: the same words in another order can
               mean the opposite, so a bag of words is clearly worse than a network that
               reads the order.
``lookup``     Eight key-value pairs (one token each, a letter and a digit) and a question:
               ``c7 x2 m4 … ? x`` is answered by the digit of the asked key (2). Retrieval:
               attention finds the pair of the key and reads its digit; an average of the
               embeddings knows which pairs occur but not which one is asked for, and stays
               near chance.
``topics``     Texts about science, sports, cooking or travel: topic words in shared filler
               words, with a few words of another topic. The order does not matter; an easy
               first dataset.

Sample ``index`` of a dataset depends only on its own seed, so the data is the same on
every machine and for every run. Below the imports, the module uses only the standard
library: the generated PyTorch script of a network embeds it (see codegen.py) and trains
on the same data without Brick DNN.
"""

from __future__ import annotations

import random
import re

TEXT_PAD = "<pad>"  # id 0: fills a sequence up to its length
TEXT_UNK = "<unk>"  # id 1: a word that is not in the vocabulary


def _words(text: str) -> list[str]:
    return text.split()


# --------------------------------------------------------------------------- #
# Sentiment: short reviews with negations, intensifiers and "but"
# --------------------------------------------------------------------------- #

_SUBJECTS = _words(
    "food soup pizza coffee dessert service staff waiter room bed view hotel beach trip "
    "flight train guide museum movie plot ending story actor music song album concert show "
    "book game app phone laptop camera screen battery price course teacher lesson"
)
_DETERMINERS = _words("the our my this their")
_VERBS = _words("was is seemed felt looked")
_POSITIVE = _words(
    "good great excellent wonderful lovely amazing pleasant fantastic superb nice "
    "delightful brilliant charming enjoyable fun perfect solid impressive friendly helpful "
    "clean fresh comfortable beautiful tasty smooth reliable clever gripping cosy"
)
_NEGATIVE = _words(
    "bad awful terrible poor boring dull horrible disappointing bland rude dirty slow noisy "
    "broken ugly weak annoying mediocre stale cold cramped confusing buggy overpriced "
    "tedious lousy sloppy flimsy greasy unpleasant"
)
_NEGATIONS = _words("not never hardly")
_INTENSIFIERS = _words("very really quite extremely truly so rather pretty incredibly fairly")
_OPENERS = (
    "honestly",
    "overall",
    "frankly",
    "i think",
    "we thought",
    "in the end",
    "to be honest",
    "for me",
    "i felt",
    "at first",
)
_ENDINGS = ("this time", "for the price", "today", "last night", "on the whole", "as usual")
SENTIMENT_CLASSES = ("negative", "positive")
SENTIMENT_LENGTH = 24


def _clause(rng: random.Random, polarity: int) -> list[str]:
    """``the soup was not very tasty``: a clause whose meaning has ``polarity`` (1 positive,
    0 negative). A negation flips the adjective, an intensifier keeps it."""
    negated = rng.random() < 0.4
    positive_word = polarity == 1 if not negated else polarity == 0
    words = [rng.choice(_DETERMINERS), rng.choice(_SUBJECTS), rng.choice(_VERBS)]
    if negated:
        words.append(rng.choice(_NEGATIONS))
    if rng.random() < 0.4:
        words.append(rng.choice(_INTENSIFIERS))
    words.append(rng.choice(_POSITIVE if positive_word else _NEGATIVE))
    return words


def _sentiment(rng: random.Random) -> tuple[list[str], int]:
    label = rng.randrange(2)
    form = rng.random()
    if form < 0.35:  # one clause
        words = _clause(rng, label)
    elif form < 0.55:  # two clauses that agree
        words = [*_clause(rng, label), "and", *_clause(rng, label)]
    else:  # "… but …": the second clause decides
        words = [*_clause(rng, 1 - label), "but", *_clause(rng, label)]
    if rng.random() < 0.3:
        words = [*rng.choice(_OPENERS).split(), *words]
    if rng.random() < 0.25:
        words += rng.choice(_ENDINGS).split()
    return words, label


# --------------------------------------------------------------------------- #
# Key lookup: key-value pairs and a question
# --------------------------------------------------------------------------- #

_KEYS = tuple("abcdefghijklmnopqrstuvwxyz")
_VALUES = tuple("0123456789")
_PAIRS = 8
LOOKUP_CLASSES = _VALUES
LOOKUP_LENGTH = _PAIRS + 2


def _lookup(rng: random.Random) -> tuple[list[str], int]:
    """``c7 x2 m4 … ? x``: eight pairs (one token each: a key letter and its digit), then
    the question for one of the keys. The answer is its digit."""
    keys = rng.sample(_KEYS, _PAIRS)
    values = [rng.randrange(len(_VALUES)) for _ in keys]
    asked = rng.randrange(_PAIRS)
    pairs = [key + _VALUES[value] for key, value in zip(keys, values, strict=True)]
    return [*pairs, "?", keys[asked]], values[asked]


# --------------------------------------------------------------------------- #
# Topics: topic words in shared filler words
# --------------------------------------------------------------------------- #

_TOPIC_WORDS = (
    _words(
        "atom cell energy experiment gene laboratory molecule physics planet research "
        "theory chemistry telescope data microscope particle scientist hypothesis "
        "electron study biology measurement climate fossil equation"
    ),
    _words(
        "match team goal player coach league score football tennis race stadium referee "
        "season champion tournament ball sprint medal training fans striker keeper "
        "victory defeat penalty"
    ),
    _words(
        "recipe oven flour butter garlic onion pan sauce salt pepper bake boil kitchen "
        "chef dough spice cheese tomato soup knife roast dinner herbs sugar stew"
    ),
    _words(
        "flight hotel passport airport luggage beach map tourist ticket train journey "
        "island border guide museum backpack visa coast mountain village cruise station "
        "suitcase holiday sightseeing"
    ),
)
_FILLER = _words(
    "the a an and of to in it was we they with on for this that very some new after "
    "before really just about when there then our their many few most also more today "
    "week year last next people time day good great small big long first other every "
    "always often again because while still even almost quite"
)
TOPIC_CLASSES = ("science", "sports", "cooking", "travel")
TOPICS_LENGTH = 64


def _topics(rng: random.Random) -> tuple[list[str], int]:
    label = rng.randrange(len(TOPIC_CLASSES))
    count = rng.randint(40, TOPICS_LENGTH)
    words = []
    for _ in range(count):
        chance = rng.random()
        if chance < 0.2:
            words.append(rng.choice(_TOPIC_WORDS[label]))
        elif chance < 0.26:  # a word of another topic
            other = (label + rng.randrange(1, len(TOPIC_CLASSES))) % len(TOPIC_CLASSES)
            words.append(rng.choice(_TOPIC_WORDS[other]))
        else:
            words.append(rng.choice(_FILLER))
    return words, label


# --------------------------------------------------------------------------- #
# All datasets
# --------------------------------------------------------------------------- #


def _unique(*groups: object) -> tuple[str, ...]:
    words: dict[str, None] = {}
    for group in groups:
        for item in group:  # type: ignore[attr-defined]
            for word in str(item).split():
                words[word] = None
    return (TEXT_PAD, TEXT_UNK, *words)


# id: (sample function, vocabulary, sequence length, class names, seed, train, test)
TEXT_DATASETS = {
    "sentiment": (
        _sentiment,
        _unique(
            _DETERMINERS,
            _SUBJECTS,
            _VERBS,
            _NEGATIONS,
            _INTENSIFIERS,
            _POSITIVE,
            _NEGATIVE,
            ["and", "but"],
            _OPENERS,
            _ENDINGS,
        ),
        SENTIMENT_LENGTH,
        SENTIMENT_CLASSES,
        20_261_117,
        6000,
        1000,
    ),
    "lookup": (
        _lookup,
        _unique([key + value for key in _KEYS for value in _VALUES], ["?"], _KEYS),
        LOOKUP_LENGTH,
        LOOKUP_CLASSES,
        20_261_118,
        12_000,
        2000,
    ),
    "topics": (
        _topics,
        _unique(_FILLER, *_TOPIC_WORDS),
        TOPICS_LENGTH,
        TOPIC_CLASSES,
        20_261_119,
        3000,
        1000,
    ),
}


def text_vocabulary(dataset: str) -> tuple[str, ...]:
    """The words of ``dataset`` in the order of their ids (0 ``<pad>``, 1 ``<unk>``)."""
    return TEXT_DATASETS[dataset][1]


def text_sample(dataset: str, index: int) -> tuple[list[str], int]:
    """The words of sample ``index`` (before padding) and its class."""
    make, *_, seed, _train, _test = TEXT_DATASETS[dataset]
    return make(random.Random(seed * 100_003 + index))


def tokenize(text: str) -> list[str]:
    """Lower-case words, numbers and single punctuation marks: "Not bad!" -> not, bad, !."""
    return re.findall(r"[a-z0-9']+|[^\sa-z0-9']", text.lower())


def encode(words: list[str], vocabulary: tuple[str, ...], length: int) -> list[int]:
    """Token ids of ``words`` (unknown words get the id of ``<unk>``), cut or padded with
    ``<pad>`` to ``length``."""
    index = {word: i for i, word in enumerate(vocabulary)}
    ids = [index.get(word, 1) for word in words[:length]]
    return ids + [0] * (length - len(ids))


def decode(ids: list[int], vocabulary: tuple[str, ...]) -> list[str]:
    """The words of token ids, without the padding."""
    return [vocabulary[i] if 0 <= i < len(vocabulary) else TEXT_UNK for i in ids if i != 0]


def text_examples(dataset: str, count: int = 2) -> list[dict[str, str]]:
    """The first samples of the test split as text with their class (for the data picker)."""
    *_, classes, _seed, train, _test = TEXT_DATASETS[dataset]
    examples = []
    for index in range(train, train + count):
        words, label = text_sample(dataset, index)
        examples.append({"text": " ".join(words), "class": classes[label]})
    return examples


def text_dataset(dataset: str, start: int, count: int) -> tuple[list[list[int]], list[int]]:
    """Samples ``start`` … ``start + count - 1`` as token ids (padded) and classes. The
    first ``train`` samples are the training split, the next ``test`` the test split."""
    _, vocabulary, length, *_ = TEXT_DATASETS[dataset]
    index = {word: i for i, word in enumerate(vocabulary)}
    ids, labels = [], []
    for number in range(start, start + count):
        words, label = text_sample(dataset, number)
        row = [index[word] for word in words[:length]]
        ids.append(row + [0] * (length - len(row)))
        labels.append(label)
    return ids, labels
