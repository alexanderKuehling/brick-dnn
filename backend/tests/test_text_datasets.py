"""The built-in text datasets (Sentiment, Key lookup, Topics): the generator, the dataset
descriptions, the checks and fixes for networks of token ids, the data of a training,
predictions of texts, the generated script, and short real trainings on the CPU."""

from __future__ import annotations

import copy
import json
import re
import subprocess
import sys

import pytest

from brickdnn import EXAMPLES, generate_code
from brickdnn.training import TrainingConfig, builtin_info, check_compatibility, inference
from brickdnn.training.datasets import BUILTIN_DATASETS, DatasetInfo
from brickdnn.training.textsets import (
    TEXT_DATASETS,
    decode,
    encode,
    text_dataset,
    text_examples,
    text_sample,
    text_vocabulary,
    tokenize,
)

from .helpers import chain

try:
    import torch
except ImportError:  # pragma: no cover - PyTorch is optional
    torch = None

needs_torch = pytest.mark.skipif(torch is None, reason="PyTorch is not installed")

# id: (name, vocabulary, length, classes, training texts, test texts)
SPECS = {
    "sentiment": ("Sentiment", 151, 24, ("negative", "positive"), 6000, 1000),
    "lookup": ("Key lookup", 289, 10, tuple("0123456789"), 12_000, 2000),
    "topics": ("Topics", 161, 64, ("science", "sports", "cooking", "travel"), 3000, 1000),
}


def example(name: str) -> dict:
    return copy.deepcopy(EXAMPLES[name]["graph"])


def fixed(graph: dict, dataset: str) -> dict:
    """``graph`` with the fixes of the check for ``dataset`` applied."""
    graph = copy.deepcopy(graph)
    for fix in check_compatibility(graph, builtin_info(dataset)).fixes:
        node = next(n for n in graph["nodes"] if n["id"] == fix["node"])
        node["params"].update(fix["params"])
    return graph


def tiny_transformer(dataset: str, width: int = 16) -> dict:
    _, words, length, classes, *_ = SPECS[dataset]
    return chain(
        ("Input", {"shape": [length], "dtype": "long"}),
        ("Embedding", {"num_embeddings": words, "embedding_dim": width}),
        ("PositionalEmbedding", {"max_len": length}),
        ("TransformerEncoder", {"nhead": 2, "dim_feedforward": 2 * width, "num_layers": 1}),
        ("SequencePool", {"mode": "mean"}),
        ("Linear", {"out_features": len(classes)}),
    )


# --------------------------------------------------------------------------- #
# The generator
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("dataset", list(SPECS))
def test_datasets_are_generated_the_same_every_time(dataset):
    _, words, length, classes, train, test = SPECS[dataset]
    _, vocabulary, size, names, _, n_train, n_test = TEXT_DATASETS[dataset]
    assert (len(vocabulary), size, tuple(names), n_train, n_test) == (
        words,
        length,
        classes,
        train,
        test,
    )
    assert vocabulary[:2] == ("<pad>", "<unk>") and len(set(vocabulary)) == len(vocabulary)
    ids, labels = text_dataset(dataset, 0, train + test)
    assert len(ids) == len(labels) == train + test
    assert all(len(row) == length for row in ids)
    assert all(0 <= i < words for row in ids for i in row)
    # The unknown word never occurs in the generated texts; every class does, about equally.
    assert not any(1 in row for row in ids)
    counts = [labels.count(k) for k in range(len(classes))]
    assert min(counts) > 0.8 * (train + test) / len(classes), counts
    # Every sample has its own seed: a part is the same as the whole.
    assert text_dataset(dataset, train - 3, 6) == (
        ids[train - 3 : train + 3],
        labels[train - 3 : train + 3],
    )
    assert text_sample(dataset, 17) == text_sample(dataset, 17)
    words_of_sample, label = text_sample(dataset, 17)
    assert decode(ids[17], vocabulary) == words_of_sample[:length] and labels[17] == label
    # Different texts, not one text over and over.
    assert len({tuple(row) for row in ids}) > 0.9 * (train + test)


def test_the_texts_say_what_they_should():
    """Read the texts like a person would: the label follows from the words."""
    for index in range(300):
        words, label = text_sample("lookup", index)
        pairs, asked = words[:8], words[-1]
        assert words[8] == "?" and len({pair[0] for pair in pairs}) == 8
        assert int(next(p[1] for p in pairs if p[0] == asked)) == label
        words, label = text_sample("sentiment", index)
        assert 3 <= len(words) <= SPECS["sentiment"][2]
        if "but" in words:  # the second half decides
            words = words[words.index("but") + 1 :]
        elif "and" in words:
            words = words[: words.index("and")]
        positive = {"good", "great", "excellent", "nice", "lovely", "fun", "perfect", "friendly"}
        negative = {"bad", "awful", "terrible", "boring", "dull", "rude", "slow", "noisy"}
        flipped = any(w in ("not", "never", "hardly") for w in words)
        if positive & set(words):
            assert label == (0 if flipped else 1), words
        if negative & set(words):
            assert label == (1 if flipped else 0), words
        words, label = text_sample("topics", index)
        assert 40 <= len(words) <= 64
    # Negations and "but" are common enough to matter.
    texts = [text_sample("sentiment", i)[0] for i in range(1000)]
    assert sum("but" in t for t in texts) > 350
    assert sum(any(w in ("not", "never", "hardly") for w in t) for t in texts) > 400


def test_tokenize_encode_and_decode():
    vocabulary = text_vocabulary("sentiment")
    words = tokenize("Honestly, the SOUP was not tasty!")
    assert words == ["honestly", ",", "the", "soup", "was", "not", "tasty", "!"]
    ids = encode(words, vocabulary, 10)
    assert len(ids) == 10 and ids[1] == ids[7] == 1  # "," and "!" are unknown
    assert ids[-2:] == [0, 0]
    assert decode(ids, vocabulary) == [
        "honestly",
        "<unk>",
        "the",
        "soup",
        "was",
        "not",
        "tasty",
        "<unk>",
    ]
    assert len(encode(["good"] * 30, vocabulary, 24)) == 24  # cut to the length
    assert tokenize("c7 x2 ? x") == ["c7", "x2", "?", "x"]
    examples = text_examples("lookup")
    assert len(examples) == 2 and examples[0]["class"] in SPECS["lookup"][3]
    assert examples[0]["text"] == " ".join(text_sample("lookup", 12_000)[0])


# --------------------------------------------------------------------------- #
# The datasets as Brick DNN describes them
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("dataset", list(SPECS))
def test_built_in_text_datasets(dataset):
    name, words, length, classes, train, test = SPECS[dataset]
    spec = BUILTIN_DATASETS[dataset]
    assert (spec.name, spec.is_text, spec.generated, spec.download_mb) == (name, True, True, 0)
    assert spec.task == "classification" and spec.shape == (length,)
    data = spec.to_dict()
    assert data["input_kind"] == "tokens" and data["sample_shape"] == [length]
    assert (data["vocab_size"], data["sequence_length"]) == (words, length)
    assert data["class_names"] == list(classes) and data["num_classes"] == len(classes)
    assert (data["train_size"], data["test_size"]) == (train, test)
    assert [e["text"] for e in data["examples"]] == [e["text"] for e in text_examples(dataset)]
    # The description names the vocabulary and the length.
    assert f"{words} " in data["description"] and f"{length} " in data["description"]
    info = builtin_info(dataset)
    assert (info.kind, info.builtin, info.task) == ("builtin", dataset, None)
    assert info.tokens == {"vocab_size": words, "length": length}
    assert info.sample_shape == [length] and info.num_classes == len(classes)
    assert DatasetInfo.from_dict(json.loads(json.dumps(info.to_dict()))) == info
    assert BUILTIN_DATASETS["mnist"].to_dict()["input_kind"] == "images"
    assert "vocab_size" not in BUILTIN_DATASETS["mnist"].to_dict()


# --------------------------------------------------------------------------- #
# Checks and fixes
# --------------------------------------------------------------------------- #


def test_the_transformer_example_fits_topics_and_is_adapted_to_the_others():
    graph = example("transformer_classifier")
    topics = check_compatibility(graph, builtin_info("topics"))
    assert topics.ok, topics.errors
    # The Embedding of 10,000 words is far too large: a warning and a fix.
    assert topics.warnings == [
        "The Embedding has 10,000 entries, but Topics has only 161 words: the other 9,839 are "
        "never used. Set its vocabulary size to 161."
    ]
    assert [f["summary"] for f in topics.fixes] == ["Vocabulary size of the Embedding 10000 → 161"]
    plan = topics.plan
    assert plan.task == "classification" and plan.num_classes == 4
    assert plan.class_names == ["science", "sports", "cooking", "travel"]
    assert plan.inputs == [{"shape": [64], "dtype": "long", "vocab": 10000}]
    assert plan.tokens == {"dataset": "topics", "length": 64, "vocab_size": 161}
    assert plan.summary == (
        "Classification into 4 classes (Topics, 3,000 training and 1,000 test texts of 64 tokens)."
    )
    assert any("1,000 test texts are the test set" in note for note in plan.notes)

    lookup = check_compatibility(graph, builtin_info("lookup"))
    assert not lookup.ok
    assert lookup.errors == [
        "Every text of Key lookup has 10 tokens, but the Input expects shape (64). Set the "
        "Input shape to 10.",
        "The network outputs 4 values, but Key lookup has 10 classes. The last layer should "
        "output 10 values (e.g. Linear with 10 outputs).",
    ]
    assert [(f["node"], f["params"]) for f in lookup.fixes] == [
        ("tokens", {"shape": [10]}),
        ("embed", {"num_embeddings": 289}),
        ("fc", {"out_features": 10}),
    ]
    assert [f["summary"] for f in lookup.fixes] == [
        "Input shape 64 → 10",
        "Vocabulary size of the Embedding 10000 → 289",
        "Outputs of the last layer 4 → 10",
    ]
    for dataset in SPECS:
        check = check_compatibility(fixed(graph, dataset), builtin_info(dataset))
        assert check.ok and not check.warnings and not check.fixes, (dataset, check.errors)


def test_the_lstm_example_is_adapted_to_sentiment():
    graph = example("lstm_sentiment")
    check = check_compatibility(graph, builtin_info("sentiment"))
    assert check.errors == [
        "Every text of Sentiment has 24 tokens, but the Input expects shape (128). Set the "
        "Input shape to 24."
    ]
    assert [f["summary"] for f in check.fixes] == [
        "Input shape 128 → 24",
        "Vocabulary size of the Embedding 20000 → 151",
    ]
    assert check_compatibility(fixed(graph, "sentiment"), builtin_info("sentiment")).ok


def test_networks_that_do_not_read_token_ids():
    sentiment = builtin_info("sentiment")
    mlp = check_compatibility(example("mlp_mnist"), sentiment)
    assert mlp.errors == [
        "Sentiment holds texts as token ids: the network needs an Input with data type long "
        "and shape 24, followed by an Embedding layer (e.g. the Transformer or LSTM example)."
    ]
    assert mlp.fixes == []  # a float Input before a Linear layer cannot take token ids
    # Too few words in the Embedding is an error; text networks do not train on images.
    small = chain(
        ("Input", {"shape": [24], "dtype": "long"}),
        ("Embedding", {"num_embeddings": 100, "embedding_dim": 8}),
        ("SequencePool", {"mode": "mean"}),
        ("Linear", {"out_features": 2}),
    )
    check = check_compatibility(small, sentiment)
    assert check.errors == [
        "Sentiment has a vocabulary of 151 words (token ids 0 to 150), but the Embedding knows "
        "only 100. Set its vocabulary size to 151."
    ]
    assert [f["params"] for f in check.fixes] == [{"num_embeddings": 151}]
    images = check_compatibility(small, builtin_info("mnist"))
    assert images.errors[0].startswith("This network expects integer token ids")
    assert "built-in text datasets (Sentiment, Key lookup, Topics)" in images.errors[0]
    # A positional embedding shorter than the texts cannot be fixed by the Input alone.
    short = example("transformer_classifier")
    next(n for n in short["nodes"] if n["id"] == "pos")["params"]["max_len"] = 8
    assert check_compatibility(short, builtin_info("topics")).fixes == []


def test_classical_models_do_not_train_on_texts():
    graph = example("svm_classifier")
    check = check_compatibility(graph, builtin_info("topics"))
    assert not check.ok and "not on texts like Topics" in check.errors[0]
    assert check.fixes == []


# --------------------------------------------------------------------------- #
# Predictions of texts
# --------------------------------------------------------------------------- #


def text_plan_and_preprocessing(dataset: str = "sentiment"):
    graph = fixed(example("transformer_classifier"), dataset)
    plan = check_compatibility(graph, builtin_info(dataset)).plan.to_dict()
    vocabulary = list(text_vocabulary(dataset))
    preprocessing = {
        "kind": "tokens",
        "dataset": dataset,
        "length": SPECS[dataset][2],
        "vocabulary": vocabulary,
        "pad_id": 0,
        "unknown_id": 1,
        "class_names": list(SPECS[dataset][3]),
    }
    return plan, preprocessing


def test_texts_and_token_ids_are_prepared_like_the_training_texts():
    plan, preprocessing = text_plan_and_preprocessing()
    assert inference.kind_of(plan, preprocessing) == "tokens"
    vocabulary = text_vocabulary("sentiment")
    prepared = inference.prepare_inputs(
        plan, preprocessing, ["The soup was not tasty", "the food was good but the staff zzz"]
    )
    assert prepared.kind == "tokens" and prepared.count == 2
    (tensor,) = prepared.tensors
    assert tensor["dtype"] == "long" and tensor["shape"] == [2, 24]
    first = encode(tokenize("the soup was not tasty"), vocabulary, 24)
    assert tensor["values"][:24] == first
    assert prepared.texts[0] == {"tokens": ["the", "soup", "was", "not", "tasty"], "unknown": []}
    assert prepared.texts[1]["tokens"][-1] == "<unk>"
    assert prepared.texts[1]["unknown"] == ["zzz"]
    # One text, one list of token ids (padded), or several of them.
    assert inference.prepare_inputs(plan, preprocessing, "nice").count == 1
    ids = inference.prepare_inputs(plan, preprocessing, [2, 3, 4])
    assert ids.tensors[0]["values"] == [2, 3, 4] + [0] * 21
    assert ids.texts == [{"tokens": list(vocabulary[2:5]), "unknown": []}]
    assert inference.prepare_inputs(plan, preprocessing, [[2], [5, 6]]).count == 2
    for inputs, message in (
        ([[2] * 25], "The sample has 25 token ids, but the network reads 24."),
        ([[2], [2] * 25], "Sample 2 has 25 token ids"),
        ([7, 151], "token ids must be whole numbers from 0 to 150"),
        ([{"text": "good"}], "neither a text nor a list of token ids: the network takes a"),
        ([], "'inputs' must be a list of samples"),
        (["good"] * 65, "65 samples are too many"),
    ):
        with pytest.raises(inference.InvalidInputs, match=re.escape(message)):
            inference.prepare_inputs(plan, preprocessing, inputs)
    description = inference.describe_inputs(plan, preprocessing)
    assert description.startswith("One text per sample, as a string or as a list of up to 24")
    assert "become <unk> (id 1)" in description
    # The outputs are classes like for any classifier.
    (prediction,) = inference.read_outputs(plan, preprocessing, [[[0.2, 1.4]]], 1)
    assert prediction["class"] == "positive"
    # Preprocessing that does not fit the network cannot predict.
    for broken in ({**preprocessing, "length": 32}, {**preprocessing, "vocabulary": ["a"]}):
        with pytest.raises(inference.UnknownPreprocessing):
            inference.kind_of(plan, broken)


def test_the_usage_snippet_encodes_texts_like_the_server():
    plan, preprocessing = text_plan_and_preprocessing("topics")
    snippet = inference.usage_snippet(plan, preprocessing, "my_net", "my-net.pt")
    compile(snippet, "<snippet>", "exec")
    assert "CLASS_NAMES = ['science', 'sports', 'cooking', 'travel']" in snippet
    assert text_examples("topics", 1)[0]["text"] in snippet
    # Its vocabulary and encode() give the same token ids as the training.
    code = snippet[snippet.index("VOCABULARY = (") : snippet.index("x = torch.tensor")]
    namespace: dict = {}
    exec("import re\n" + code, namespace)
    assert namespace["VOCABULARY"] == list(text_vocabulary("topics"))
    text = "The CHEF baked bread, and the team won!"
    assert namespace["encode"](text) == encode(tokenize(text), text_vocabulary("topics"), 64)


# --------------------------------------------------------------------------- #
# The generated script
# --------------------------------------------------------------------------- #


def test_the_script_embeds_the_dataset_the_network_fits():
    code = generate_code(example("transformer_classifier"))
    assert 'DATASET = "topics"' in code and "Train it on Topics" in code
    assert "python this_file.py --train --epochs 10" in code
    assert "def predict_text(" in code and "import csv" not in code
    # The generator is embedded as it is (standard library only) and makes the same texts.
    start = code.index("TEXT_PAD = ")
    end = code.index("# Training on the built-in text dataset")
    namespace: dict = {}
    exec("import random\nimport re\n" + code[start:end], namespace)
    for dataset in SPECS:
        train = SPECS[dataset][4]
        assert namespace["text_dataset"](dataset, train - 5, 10) == text_dataset(
            dataset, train - 5, 10
        )
        assert namespace["text_vocabulary"](dataset) == text_vocabulary(dataset)
    # The network after the fixes for Key lookup gets Key lookup; others get the CSV loop.
    assert 'DATASET = "lookup"' in generate_code(fixed(example("transformer_classifier"), "lookup"))
    assert 'DATASET = "sentiment"' in generate_code(fixed(example("lstm_sentiment"), "sentiment"))
    lstm = generate_code(example("lstm_sentiment"))  # 128 tokens: no dataset fits
    assert "DATASET =" not in lstm and "--csv" not in lstm and "TEXT_PAD" not in lstm
    assert "def read_csv(" in generate_code(example("mlp_mnist"))
    # A dataset can be chosen, if the network fits it.
    graph = fixed(example("transformer_classifier"), "topics")
    assert 'DATASET = "topics"' in generate_code(graph, {"dataset": "topics"})
    with pytest.raises(ValueError, match="does not fit the built-in dataset 'lookup'"):
        generate_code(graph, {"dataset": "lookup"})
    with pytest.raises(ValueError, match="'news' is not a built-in text dataset"):
        generate_code(graph, {"dataset": "news"})
    # Without the training code, nothing of it.
    plain = generate_code(graph, {"include_training": False})
    assert "TEXT_PAD" not in plain and "argparse" not in plain


@needs_torch
def test_the_generated_script_trains_on_the_built_in_dataset(tmp_path):
    graph = tiny_transformer("lookup")
    code = generate_code(graph)
    script = tmp_path / "lookup_net.py"
    script.write_text(code)
    environment = {"OMP_NUM_THREADS": "1", "PATH": ""}
    trained = subprocess.run(
        [sys.executable, str(script), "--train", "--epochs", "1", "--save", "net.pt"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=300,
        env=environment,
    )
    assert trained.returncode == 0, trained.stderr
    assert "Training on cpu: 9600 texts of lookup, 2400 for validation, 2000 for the test" in (
        trained.stdout
    )
    assert "Test accuracy:" in trained.stdout and (tmp_path / "net.pt").is_file()
    predicted = subprocess.run(
        [sys.executable, str(script), "--predict", "c7 x2 m4 ? x", "--save", "net.pt"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
        env=environment,
    )
    assert predicted.returncode == 0, predicted.stderr
    lines = predicted.stdout.splitlines()
    assert lines[0] == "Tokens: c7 x2 m4 ? x" and len(lines) == 11
    assert all(re.fullmatch(r"\d: \d+\.\d%", line) for line in lines[1:])


# --------------------------------------------------------------------------- #
# Training (PyTorch)
# --------------------------------------------------------------------------- #


@needs_torch
def test_text_data_split_and_batches():
    from brickdnn.training.data import prepare_data
    from brickdnn.training.engine import Reporter

    info = builtin_info("sentiment")
    plan = check_compatibility(tiny_transformer("sentiment"), info).plan
    data = prepare_data(
        plan, info, None, TrainingConfig(), None, Reporter(), torch.Generator().manual_seed(0)
    )
    assert (len(data.train), len(data.val), len(data.test)) == (4800, 1200, 1000)
    inputs, targets = data.train.batch(torch.arange(4))
    assert inputs[0].shape == (4, 24) and inputs[0].dtype == torch.long
    assert targets[0].dtype == torch.long and set(targets[0].tolist()) <= {0, 1}
    assert torch.equal(data.test.indices, torch.arange(6000, 7000))
    test_inputs, test_targets = data.test.batch(torch.arange(1))
    ids, labels = text_dataset("sentiment", 6000, 1)
    assert test_inputs[0][0].tolist() == ids[0] and test_targets[0][0].item() == labels[0]
    pre = data.preprocessing
    assert pre["kind"] == "tokens" and pre["dataset"] == "sentiment" and pre["length"] == 24
    assert pre["vocabulary"] == list(text_vocabulary("sentiment"))
    assert (pre["pad_id"], pre["unknown_id"]) == (0, 1)
    assert pre["class_names"] == ["negative", "positive"]
    # Max. samples shrinks the training and the test split.
    small = prepare_data(
        plan,
        info,
        None,
        TrainingConfig.from_dict({"max_samples": 400}),
        None,
        Reporter(),
        torch.Generator().manual_seed(0),
    )
    assert len(small.train) == 400 and len(small.test) == 100


class Recorder:
    """Keeps what the engine reports."""

    def __init__(self):
        self.info, self.history, self.test, self.logs = {}, [], None, []

    def status(self, status, message):
        pass

    def log(self, line):
        self.logs.append(line)

    def setup(self, info):
        self.info.update(info)

    def progress(self, progress):
        pass

    def epoch_end(self, entry, best):
        self.history.append(entry)

    def weights_kept(self, result):
        pass

    def test_end(self, result):
        self.test = result

    def evaluation_end(self, result):
        pass

    def finish(self, status, message, weights):
        pass

    def should_stop(self):
        return False


@needs_torch
@pytest.mark.parametrize(
    ("dataset", "config", "least"),
    [
        # Two epochs of a tiny Transformer on a few hundred texts: the easy Topics is learned,
        # the others at least start to be (chance: 25 %, 50 %, 10 %).
        ("topics", {"epochs": 2, "max_samples": 1500, "learning_rate": 0.005}, 0.6),
        ("sentiment", {"epochs": 2, "max_samples": 600, "learning_rate": 0.003}, None),
        ("lookup", {"epochs": 2, "max_samples": 600, "learning_rate": 0.003}, None),
    ],
)
def test_a_tiny_transformer_trains_on_every_text_dataset(tmp_path, dataset, config, least):
    from brickdnn.training.engine import TrainingJob, train

    torch.set_num_threads(1)
    job = TrainingJob(
        graph=tiny_transformer(dataset),
        dataset=builtin_info(dataset),
        config=TrainingConfig.from_dict({**config, "seed": 1, "device": "cpu"}),
    )
    reporter = Recorder()
    result = train(job, reporter, tmp_path / "weights.pt")
    assert result.status == "completed", result.message
    assert reporter.info["class_names"] == list(SPECS[dataset][3])
    assert reporter.info["preprocessing"]["kind"] == "tokens"
    history = result.history
    assert len(history) == 2 and history[-1]["train_loss"] < history[0]["train_loss"], history
    assert reporter.test is not None and 0 <= reporter.test["accuracy"] <= 1
    if least is not None:
        assert reporter.test["accuracy"] > least, reporter.test
    assert (tmp_path / "weights.pt").is_file()
    saved = json.loads((tmp_path / "weights.preprocessing.json").read_text())
    assert saved["vocabulary"][:2] == ["<pad>", "<unk>"]
