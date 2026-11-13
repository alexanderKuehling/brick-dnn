"""Explain a trained model and look inside it, on demand in the prediction process.

The server sends ``{"explain": {...}}`` to ``python -m brickdnn.training.predict RUN_DIR``
(see :func:`.predict.predict`), which hands it to :func:`explain`. The request::

    {"method": "image",                 # see METHODS
     "plan": {...}, "preprocessing": {...},   # of the run (TrainingRun.plan, state.json)
     "sample": {"index": 3}             # a sample of the run's validation data (data.py;
                                        # without it: of its test or training data),
             | {"tensor": {...}}        # or a prepared input (like a prediction's)
             | null,                    # (activations of a demo run: a random input)
     "target": 2,                       # the class to explain (default: the predicted one)
     "layer": "conv2",                  # Grad-CAM: the brick (default: the last Conv2d)
     "node": "fc1"}                     # activations: only this brick

Methods: ``samples`` (some validation samples with the model's predictions, to pick one),
``image`` (saliency map, Grad-CAM and occlusion of one image), ``tabular`` (Shapley values
and, for networks, integrated gradients of one row), ``permutation`` (permutation feature
importance on validation rows), ``boundary`` (the predictions on a grid over two input
features, with training points), ``activations`` (the output of every brick for one
sample) and ``weights`` (learned filters and weight histograms).

Answers are JSON with pictures as PNG data URLs and numbers rounded (see render.py).
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..job import TrainingJob
from ..reporting import short_error
from ..state import read_job
from . import attribution, render
from .data import ExplainError, load_pool

METHODS = (
    "samples",
    "image",
    "tabular",
    "permutation",
    "boundary",
    "activations",
    "weights",
)
# Optional parts (occlusion, more permutation repeats) are skipped after this many seconds
# of the process, so that the answer comes before the prediction time limit.
SOFT_SECONDS = 18.0
# Samples listed by "samples": the first of the pool, plus some the model gets wrong.
LIST_FIRST = 16
LIST_WRONG = 8
LIST_THUMB = 48
# Largest side of an image shown (bigger images are shrunk).
SHOW_SIDE = 160
# Decision boundary: grid points per side, training points shown.
BOUNDARY_RESOLUTION = 64
BOUNDARY_POINTS = 300
# The coarse grid of a boundary (for assistants): every 4th point, the centre of each block.
GRID_PICK = slice(2, BOUNDARY_RESOLUTION, 4)
# Permutation importance: validation rows and repeats.
PERMUTATION_ROWS = 1000
PERMUTATION_REPEATS = 5
TOP_K = 3

DEMO_NOTE = "Trained on random demo data: what the model learned means nothing."


def run(run_dir: str | Path, request: Any) -> dict[str, Any]:
    """:func:`explain` for the prediction process: errors as its PredictionError."""
    from ..predict import PredictionError

    try:
        return explain(run_dir, request)
    except ExplainError as exc:
        raise PredictionError(str(exc)) from None


def explain(run_dir: str | Path, request: Any) -> dict[str, Any]:
    """Raises :class:`ExplainError` (the message is for the user)."""
    started = time.monotonic()
    if not isinstance(request, dict) or request.get("method") not in METHODS:
        raise ExplainError(f"Unknown method. Methods: {', '.join(METHODS)}.")
    context = _Context(Path(run_dir), request, started + SOFT_SECONDS)
    method = request["method"]
    answer = getattr(context, f"method_{method}")()
    answer["method"] = method
    if context.notes:
        answer["notes"] = context.notes
    answer["seconds"] = round(time.monotonic() - started, 2)
    return answer


class _Context:
    def __init__(self, run_dir: Path, request: dict[str, Any], deadline: float):
        self.run_dir = run_dir
        self.request = request
        self.deadline = deadline
        self.notes: list[str] = []
        raw = read_job(run_dir)
        if raw is None:
            raise ExplainError("The network of this run is missing.")
        try:
            self.job = TrainingJob.from_dict(raw)
        except Exception as exc:
            raise ExplainError(f"The training job of this run is invalid: {exc}") from None
        self.plan = request.get("plan") if isinstance(request.get("plan"), dict) else {}
        pre = request.get("preprocessing")
        self.preprocessing: dict[str, Any] = pre if isinstance(pre, dict) else {"kind": "demo"}
        self.kind = self.preprocessing.get("kind", "demo")
        self.family = self.job.family
        targets = self.plan.get("targets") or [{}]
        self.classification = targets[0].get("kind") == "class" and len(targets) == 1
        self.regression = targets[0].get("kind") == "value" and len(targets) == 1
        self.class_names = self._class_names()
        self.demo = self.kind == "demo"
        self._pool: dict[str, Any] | None = None
        self._model: Any = None

    # ------------------------------------------------------------------ helpers

    def _class_names(self) -> list[str] | None:
        targets = self.plan.get("targets") or []
        if len(targets) != 1 or targets[0].get("kind") != "class":
            return None
        k = int(targets[0].get("num_classes") or 0)
        names = self.preprocessing.get("class_names") or self.plan.get("class_names") or []
        names = [str(n) for n in names][:k]
        return names + [f"class {i}" for i in range(len(names), k)]

    def pool(self) -> dict[str, Any]:
        if self._pool is None:
            self._pool = load_pool(self.run_dir, self.job, self.preprocessing)
        return self._pool

    def option_int(self, name: str) -> int | None:
        value = self.request.get(name)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise ExplainError(f"'{name}' must be a whole number.")
        return value

    def require(self, *, family: str | None = None, kinds: tuple[str, ...] = ()) -> None:
        if family == "neural" and self.family != "neural":
            raise ExplainError(
                "This works with neural networks only: a classical model (scikit-learn) "
                "has no layers."
            )
        if kinds and self.kind not in kinds:
            what = {"image": "image models", "tabular": "models trained on a CSV file"}
            raise ExplainError(
                f"This works with {' and '.join(what.get(k, k) for k in kinds)} only."
                + (f" {DEMO_NOTE}" if self.demo else "")
            )

    # ------------------------------------------------------------------ models

    def neural(self) -> Any:
        """The traced network with the trained weights (see neural.py)."""
        if self._model is None:
            try:
                import torch
            except ImportError as exc:  # pragma: no cover - checked by the server
                raise ExplainError("PyTorch is not installed on the server.") from exc
            from ..predict import THREADS
            from . import neural
            from .neural import build_traced_model

            torch.set_num_threads(max(1, min(THREADS, _cpus())))
            try:
                model, analysis = build_traced_model(self.job.graph)
            except Exception as exc:
                raise ExplainError(f"The network cannot be built: {short_error(exc)}") from exc
            weights = self.run_dir / "weights.pt"
            if not weights.is_file():
                raise ExplainError("This run has no trained weights.")
            try:
                state = torch.load(weights, map_location="cpu", weights_only=True)
                model.load_state_dict(state)
            except Exception as exc:
                raise ExplainError(
                    f"The trained weights do not fit the network of this run: {short_error(exc)}"
                ) from exc
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            self._model = (model, analysis, neural.module_attributes(analysis))
        return self._model

    def classical(self) -> Any:
        """The fitted scikit-learn bundle (see classical_engine.py)."""
        if self._model is None:
            import joblib

            path = self.run_dir / "model.joblib"
            if not path.is_file():
                raise ExplainError("This run has no fitted model.")
            try:
                self._model = joblib.load(path)
            except Exception as exc:
                raise ExplainError(f"The fitted model cannot be loaded: {exc}") from exc
        return self._model

    def tabular_predict(self) -> Any:
        """``predict(rows) -> outputs``: rows of standardised features (raw for classical
        demo runs), outputs are class probabilities or values in the units of the target."""
        if self.family == "classical":
            bundle = self.classical()
            pipeline = bundle["model"]
            # Standardised rows skip the pipeline's StandardScaler (it would do the same).
            model = pipeline[1:] if self.kind == "tabular" else pipeline
            if bundle.get("task") == "classification":
                from ..classical_engine import class_probabilities  # hard voting: vote shares

                indices = bundle.get("class_indices") or []
                k = int(bundle.get("num_classes") or len(indices))

                def predict(rows: np.ndarray) -> np.ndarray:
                    with _thread_limit():
                        proba = class_probabilities(model, rows)
                    full = np.zeros((len(rows), k))
                    full[:, indices or list(range(proba.shape[1]))] = proba
                    return full

                return predict

            def predict_value(rows: np.ndarray) -> np.ndarray:
                with _thread_limit():
                    return np.asarray(model.predict(rows), dtype=np.float64).reshape(-1, 1)

            return predict_value

        import torch

        from .neural import outputs_numpy

        model, _, _ = self.neural()
        t_mean = self.preprocessing.get("target_mean")
        t_std = self.preprocessing.get("target_std")

        def predict_neural(rows: np.ndarray) -> np.ndarray:
            out = outputs_numpy(model, rows)
            if self.classification:
                return torch.softmax(torch.as_tensor(out), dim=1).numpy()
            if t_std is not None:
                return out * t_std + t_mean
            return out

        return predict_neural

    def gp_std(self) -> Any:
        """The predictive standard deviation of a Gaussian process regressor, else None."""
        if self.family != "classical":
            return None
        bundle = self.classical()
        pipeline = bundle["model"]
        if type(pipeline[-1]).__name__ != "GaussianProcessRegressor":
            return None
        model = pipeline[1:] if self.kind == "tabular" else pipeline

        def std(rows: np.ndarray) -> np.ndarray:
            with _thread_limit():
                _, spread = model.predict(rows, return_std=True)
            return np.asarray(spread, dtype=np.float64).reshape(-1)

        return std

    # ------------------------------------------------------------------ samples

    def columns(self) -> list[str]:
        columns = self.preprocessing.get("columns")
        if isinstance(columns, list):
            return [str(c) for c in columns]
        n = int((self.plan.get("inputs") or [{"shape": [0]}])[0]["shape"][0])
        return [f"x{i + 1}" for i in range(n)]

    def stats(self) -> tuple[np.ndarray, np.ndarray]:
        """Mean and std of the features (standardised rows -> raw values)."""
        if self.kind == "tabular":
            return (
                np.asarray(self.preprocessing["mean"], dtype=np.float64),
                np.asarray(self.preprocessing["std"], dtype=np.float64),
            )
        n = len(self.columns())
        return np.zeros(n), np.ones(n)

    def tabular_sample(self) -> dict[str, Any]:
        """One row: ``z`` (standardised, or raw for classical demo runs), ``raw`` and the
        true target if it comes from the validation data."""
        sample = self.request.get("sample")
        mean, std = self.stats()
        if isinstance(sample, dict) and "tensor" in sample:
            values = np.asarray(sample["tensor"].get("values"), dtype=np.float64)
            if values.size != len(mean):
                raise ExplainError(f"A row needs {len(mean)} values.")
            z = values.reshape(-1)
            return {"z": z, "raw": z * std + mean, "source": "input", "index": None, "truth": None}
        index = _index(sample)
        pool = self.pool()
        source = _pool_source(pool)
        if not 0 <= index < len(pool["val_x"]):
            raise ExplainError(
                f"There are {len(pool['val_x'])} {source} rows: 0 to {len(pool['val_x']) - 1}."
            )
        raw = pool["val_x"][index]
        truth = pool["val_y"][index]
        return {
            "z": (raw - mean) / std,
            "raw": raw,
            "source": source,
            "index": index,
            "truth": int(truth) if self.classification else float(truth),
        }

    def image_sample(self) -> dict[str, Any]:
        """One image: ``image`` 1×C×H×W normalised (a torch tensor), ``unit`` C×H×W in
        [0, 1] for showing it, and the true class if it comes from the validation data."""
        import torch

        pre = self.preprocessing
        c, h, w = pre["channels"], pre["height"], pre["width"]
        mean = torch.tensor(pre["mean"], dtype=torch.float32).view(1, -1, 1, 1)
        std = torch.tensor(pre["std"], dtype=torch.float32).view(1, -1, 1, 1)
        sample = self.request.get("sample")
        if isinstance(sample, dict) and "tensor" in sample:
            values = sample["tensor"].get("values") or []
            if len(values) != c * h * w:
                raise ExplainError(f"An image needs {c * h * w} values ({c}×{h}×{w}).")
            image = torch.tensor(values, dtype=torch.float32).view(1, c, h, w)
            unit = (image * std + mean).clamp(0, 1)[0].numpy()
            return {"image": image, "unit": unit, "source": "input", "index": None, "truth": None}
        index = _index(sample)
        pool = self.pool()
        source = _pool_source(pool)
        if not 0 <= index < len(pool["images"]):
            raise ExplainError(
                f"There are {len(pool['images'])} {source} images: 0 to {len(pool['images']) - 1}."
            )
        unit_t = torch.as_tensor(pool["images"][index]).float().div(255.0)[None]
        image = (unit_t - mean) / std
        return {
            "image": image,
            "unit": unit_t[0].numpy(),
            "source": source,
            "index": index,
            "truth": int(pool["labels"][index]),
        }

    def to_net(self) -> Any:
        from ..data import apply_layout

        layout = self.preprocessing.get("layout", "chw")
        return lambda batch: apply_layout(batch, layout)

    def net_input(self) -> Any:
        """The network input for one sample (activations): any kind of run."""
        import torch

        sample = self.request.get("sample")
        if self.kind == "image":
            picked = self.image_sample()
            return self.to_net()(picked["image"]), picked
        if self.kind == "tabular":
            picked = self.tabular_sample()
            return torch.tensor(picked["z"], dtype=torch.float32)[None], picked
        specs = self.plan.get("inputs") or []
        if len(specs) != 1:
            raise ExplainError("Activations need a network with one Input brick.")
        spec = specs[0]
        shape = [int(s) for s in spec["shape"]]
        if isinstance(sample, dict) and "tensor" in sample:
            tensor = sample["tensor"]
            dtype = torch.long if tensor.get("dtype") == "long" else torch.float32
            values = tensor.get("values") or []
            if len(values) != math.prod(shape):
                raise ExplainError(f"The input needs {math.prod(shape)} values.")
            x = torch.tensor(values, dtype=dtype).view(1, *shape)
            source = "input"
        else:
            generator = torch.Generator().manual_seed(int(self.option_int("seed") or 0))
            if spec.get("dtype") == "long":
                vocab = int(spec.get("vocab") or 2)
                x = torch.randint(0, vocab, (1, *shape), generator=generator)
            else:
                x = torch.randn((1, *shape), generator=generator)
            source = "random"
        return x, {"source": source, "index": None, "truth": None}

    def sample_info(self, picked: dict[str, Any]) -> dict[str, Any]:
        info: dict[str, Any] = {"source": picked["source"], "index": picked["index"]}
        truth = picked.get("truth")
        if truth is not None:
            if self.class_names is not None:
                info["label"] = (
                    self.class_names[truth] if truth < len(self.class_names) else str(truth)
                )
                info["label_index"] = truth
            else:
                info["value"] = render.sig(truth)
        if "raw" in picked:
            info["values"] = render.sig_list(picked["raw"], 5)
        return info

    def prediction(self, outputs: np.ndarray, std: float | None = None) -> dict[str, Any]:
        """Top classes with probabilities (``outputs`` are probabilities), or the value."""
        outputs = np.asarray(outputs, dtype=np.float64).ravel()
        if self.class_names is not None:
            order = np.argsort(-outputs)[:TOP_K]
            top = [
                {
                    "class": self.class_names[i],
                    "index": int(i),
                    "probability": render.sig(outputs[i], 4),
                }
                for i in order
            ]
            return {**top[0], "top": top}
        answer: dict[str, Any] = {"value": render.sig(outputs[0], 6)}
        if std is not None:
            answer["std"] = render.sig(std, 4)
        return answer

    def target_class(self, probabilities: np.ndarray) -> int:
        target = self.option_int("target")
        k = len(probabilities)
        if target is None:
            return int(np.argmax(probabilities))
        if not 0 <= target < k:
            raise ExplainError(f"'target' must be a class index from 0 to {k - 1}.")
        return target

    # ------------------------------------------------------------------ methods

    def method_samples(self) -> dict[str, Any]:
        if self.kind == "image":
            self.require(family="neural")
            return self._image_samples()
        self.require(kinds=("tabular",) if self.family == "neural" else ("tabular", "demo"))
        return self._tabular_samples()

    def _image_samples(self) -> dict[str, Any]:
        import torch

        from .neural import forward

        pool = self.pool()
        model, _, _ = self.neural()
        pre = self.preprocessing
        mean = torch.tensor(pre["mean"], dtype=torch.float32).view(1, -1, 1, 1)
        std = torch.tensor(pre["std"], dtype=torch.float32).view(1, -1, 1, 1)
        units = torch.as_tensor(pool["images"]).float().div(255.0)
        with torch.no_grad():
            probs = torch.softmax(
                forward(model, self.to_net()((units - mean) / std)), dim=1
            ).numpy()
        chosen = _chosen(np.argmax(probs, axis=1), pool["labels"])
        samples = []
        for i in chosen:
            pixels = render.shrink(render.unit_to_uint8(units[i].numpy()), LIST_THUMB)
            predicted = int(np.argmax(probs[i]))
            truth = int(pool["labels"][i])
            samples.append(
                {
                    "index": int(i),
                    "png": render.data_url(pixels),
                    "label": self.class_names[truth] if self.class_names else str(truth),
                    "label_index": truth,
                    "predicted": self.class_names[predicted]
                    if self.class_names
                    else str(predicted),
                    "predicted_index": predicted,
                    "probability": render.sig(probs[i, predicted], 3),
                    "correct": predicted == truth,
                }
            )
        accuracy = float((np.argmax(probs, axis=1) == pool["labels"]).mean())
        return {
            "kind": "image",
            "source": _pool_source(pool),
            "total": int(len(pool["images"])),
            "pool_accuracy": render.sig(accuracy, 3),
            "samples": samples,
        }

    def _tabular_samples(self) -> dict[str, Any]:
        pool = self.pool()
        mean, std = self.stats()
        x = pool["val_x"]
        outputs = self.tabular_predict()((x - mean) / std)
        rows = []
        if self.classification:
            predicted = np.argmax(outputs, axis=1)
            chosen = _chosen(predicted, pool["val_y"])
        else:
            errors = np.abs(outputs[:, 0] - pool["val_y"])
            first = list(range(min(LIST_FIRST, len(x))))
            worst = [int(i) for i in np.argsort(-errors) if int(i) not in first][:LIST_WRONG]
            chosen = first + worst
        for i in chosen:
            row: dict[str, Any] = {"index": int(i), "values": render.sig_list(x[i], 5)}
            if self.classification:
                p, t = int(np.argmax(outputs[i])), int(pool["val_y"][i])
                names = self.class_names or []
                row.update(
                    label=names[t] if t < len(names) else str(t),
                    label_index=t,
                    predicted=names[p] if p < len(names) else str(p),
                    predicted_index=p,
                    probability=render.sig(outputs[i, p], 3),
                    correct=p == t,
                )
            else:
                row.update(
                    value=render.sig(pool["val_y"][i]), predicted_value=render.sig(outputs[i, 0])
                )
            rows.append(row)
        answer: dict[str, Any] = {
            "kind": "tabular",
            "source": _pool_source(pool),
            "total": int(len(x)),
            "columns": self.columns(),
            "target": self.preprocessing.get("target"),
            "samples": rows,
        }
        if self.demo:
            self.notes.append(DEMO_NOTE)
        return answer

    def method_image(self) -> dict[str, Any]:
        import torch

        from . import neural

        self.require(family="neural", kinds=("image",))
        if not self.classification:
            raise ExplainError("Image explanations need a classification network.")
        model, analysis, _ = self.neural()
        picked = self.image_sample()
        image, to_net = picked["image"], self.to_net()
        with torch.no_grad():
            probabilities = torch.softmax(neural.forward(model, to_net(image)), dim=1)[0].numpy()
        target = self.target_class(probabilities)
        height, width = image.shape[-2:]
        shown = render.unit_to_uint8(picked["unit"])
        answer: dict[str, Any] = {
            "kind": "image",
            "sample": self.sample_info(picked),
            "image": render.data_url(render.shrink(shown, SHOW_SIDE)),
            "size": [int(height), int(width)],
            "prediction": self.prediction(probabilities),
            "target": {"index": target, "class": self.class_names[target]},
        }

        def overlay(values: np.ndarray) -> dict[str, Any]:
            unit = render.normalise(values)
            return {
                "png": render.data_url(render.shrink(render.heat_rgba(unit), SHOW_SIDE)),
                "grid": render.grid(unit),
            }

        answer["saliency"] = {
            **overlay(neural.saliency(model, image, to_net, target)),
            "samples": neural.SMOOTHGRAD_SAMPLES,
            "noise": neural.SMOOTHGRAD_NOISE,
        }

        layers = neural.gradcam_layers(analysis)
        choices = [
            {
                "id": nid,
                "name": neural.node_label(analysis, nid),
                "type": analysis.nodes[nid].type,
                "shape": list(analysis.nodes[nid].shape or ()),
            }
            for nid in layers
        ]
        requested = self.request.get("layer")
        if requested is not None and requested not in layers:
            raise ExplainError(
                "Grad-CAM needs a brick that puts out feature maps (channels, height, width)."
            )
        layer = requested or neural.default_gradcam_layer(analysis, layers)
        if layer is None:
            answer["gradcam"] = {
                "unavailable": "Grad-CAM needs convolution bricks: this network has no brick "
                "that puts out feature maps (channels, height, width)."
            }
        else:
            coarse, upsampled = neural.gradcam(model, image, to_net, target, layer)
            entry = overlay(upsampled)
            entry["grid"] = render.grid(render.normalise(coarse))
            entry.update(layer=next(c for c in choices if c["id"] == layer), layers=choices)
            if float(coarse.max()) <= 0:
                entry["note"] = (
                    "Nothing in these feature maps speaks for this class (all weighted maps are "
                    "negative), so the map is empty."
                )
            answer["gradcam"] = entry

        result = neural.occlusion(model, image, to_net, target, self.deadline)
        if result is None:
            answer["occlusion"] = {
                "unavailable": "Skipped: it would take too long for this network."
            }
        else:
            drops = np.asarray(result["map"])
            answer["occlusion"] = {
                **overlay(drops),
                "grid": render.grid(np.asarray(result["coarse"]), 14, 3),
                "patch": result["patch"],
                "stride": result["stride"],
                "largest_drop": render.sig(float(np.max(result["coarse"])), 3),
            }
        return answer

    def method_tabular(self) -> dict[str, Any]:
        self.require(kinds=("tabular",) if self.family == "neural" else ("tabular", "demo"))
        if not (self.classification or self.regression):
            raise ExplainError("This works with models that have one output.")
        predict = self.tabular_predict()
        picked = self.tabular_sample()
        z = picked["z"]
        outputs = predict(z[None])[0]
        std_fn = self.gp_std()
        std = float(std_fn(z[None])[0]) if std_fn is not None else None
        baseline = self._baseline()
        output = self.target_class(outputs) if self.classification else 0
        shapley = attribution.shapley_values(predict, z, baseline, output)
        columns = self.columns()
        features = [
            {
                "name": columns[i],
                "value": render.sig(picked["raw"][i], 5),
                "shapley": render.sig(shapley["values"][i]),
            }
            for i in range(len(columns))
        ]
        if self.family == "neural":
            for feature, value in zip(
                features, self._integrated_gradients(z, baseline, output), strict=True
            ):
                feature["integrated_gradients"] = render.sig(value)
        features.sort(key=lambda f: -abs(f["shapley"] or 0))
        if self.classification:
            explained = f"the probability of '{self.class_names[output]}'"
        else:
            explained = f"the predicted {self.preprocessing.get('target') or 'value'}"
        answer: dict[str, Any] = {
            "kind": "tabular",
            "sample": self.sample_info(picked),
            "prediction": self.prediction(outputs, std),
            "explains": explained,
            "target": {"index": output, "class": self.class_names[output]}
            if self.classification
            else None,
            "base_value": render.sig(shapley["base"]),
            "value": render.sig(shapley["prediction"]),
            "baseline": "the mean of the training rows",
            "exact": shapley["exact"],
            "orders": shapley["orders"],
            "features": features,
        }
        if self.demo:
            self.notes.append(DEMO_NOTE)
        return answer

    def _baseline(self) -> np.ndarray:
        if self.kind == "tabular":
            return np.zeros(len(self.columns()))  # standardised: the training mean
        return np.asarray(self.pool()["train_x"], dtype=np.float64).mean(axis=0)

    def _integrated_gradients(self, z: np.ndarray, baseline: np.ndarray, output: int) -> np.ndarray:
        import torch

        from .neural import forward, integrated_gradients

        model, _, _ = self.neural()
        t_std = self.preprocessing.get("target_std")

        def score(rows: Any) -> Any:
            out = forward(model, rows)
            if self.classification:
                return torch.softmax(out, dim=1)[:, output]
            return out[:, 0] * (t_std if t_std is not None else 1.0)

        return integrated_gradients(
            score,
            torch.tensor(z, dtype=torch.float32)[None],
            torch.tensor(baseline, dtype=torch.float32)[None],
        )

    def method_permutation(self) -> dict[str, Any]:
        self.require(kinds=("tabular",) if self.family == "neural" else ("tabular", "demo"))
        if not (self.classification or self.regression):
            raise ExplainError("This works with models that have one output.")
        pool = self.pool()
        mean, std = self.stats()
        x = (pool["val_x"][:PERMUTATION_ROWS] - mean) / std
        y = pool["val_y"][:PERMUTATION_ROWS]
        try:
            result = attribution.permutation_importance(
                self.tabular_predict(),
                x,
                y,
                self.classification,
                PERMUTATION_REPEATS,
                deadline=self.deadline,
            )
        except ValueError as exc:
            raise ExplainError(str(exc)) from None
        columns = self.columns()
        features = [
            {
                "name": columns[i],
                "mean": render.sig(round(float(result["mean"][i]), 9)),
                "std": render.sig(round(float(result["std"][i]), 9)),
            }
            for i in range(len(columns))
        ]
        features.sort(key=lambda f: -(f["mean"] or 0))
        if result["repeats"] < PERMUTATION_REPEATS:
            self.notes.append(f"Only {result['repeats']} repeats fit into the time limit.")
        if self.demo:
            self.notes.append(DEMO_NOTE)
        return {
            "kind": "tabular",
            "metric": "accuracy" if self.classification else "r2",
            "baseline": render.sig(result["baseline"]),
            "rows": int(len(x)),
            "repeats": result["repeats"],
            "features": features,
        }

    def method_boundary(self) -> dict[str, Any]:
        self.require(kinds=("tabular",) if self.family == "neural" else ("tabular", "demo"))
        columns = self.columns()
        if len(columns) != 2:
            raise ExplainError(
                f"A decision boundary can be drawn for models with exactly 2 input features; "
                f"this one has {len(columns)}."
            )
        if not (self.classification or self.regression):
            raise ExplainError("This works with models that have one output.")
        pool = self.pool()
        mean, std = self.stats()
        both = np.concatenate([pool["train_x"], pool["val_x"]])
        low, high = both.min(axis=0), both.max(axis=0)
        span = np.where(high - low > 1e-9, high - low, 1.0)
        low, high = low - 0.1 * span, high + 0.1 * span
        n = BOUNDARY_RESOLUTION
        xs = np.linspace(low[0], high[0], n)
        ys = np.linspace(low[1], high[1], n)
        # Row 0 of the picture is the top: the largest value of the second feature.
        gx, gy = np.meshgrid(xs, ys[::-1])
        raw = np.stack([gx.ravel(), gy.ravel()], axis=1)
        outputs = attribution.predict_chunked(self.tabular_predict(), (raw - mean) / std)
        points = pool["train_x"][:BOUNDARY_POINTS]
        labels = pool["train_y"][:BOUNDARY_POINTS]
        answer: dict[str, Any] = {
            "kind": "tabular",
            "features": columns,
            "x": [render.sig(low[0]), render.sig(high[0])],
            "y": [render.sig(low[1]), render.sig(high[1])],
            "resolution": n,
        }
        if self.classification:
            classes = np.argmax(outputs, axis=1)
            confidence = outputs[np.arange(len(outputs)), classes]
            k = outputs.shape[1]
            colors = [render.CLASS_COLORS[i % len(render.CLASS_COLORS)] for i in range(k)]
            palette = np.array([render.hex_rgb(c) for c in colors], dtype=np.float64)
            # Strong colour where the model is sure, pale where it hesitates.
            strength = np.clip((confidence - 1 / k) / (1 - 1 / k), 0, 1)[:, None] * 0.75 + 0.15
            rgb = 255 * (1 - strength) + palette[classes] * strength
            picture = rgb.reshape(n, n, 3).round().astype(np.uint8)
            answer.update(
                classes=self.class_names,
                colors=colors,
                png=render.data_url(picture),
                grid=classes.reshape(n, n)[GRID_PICK][:, GRID_PICK].tolist(),
                points=[
                    [render.sig(a), render.sig(b), int(c)]
                    for (a, b), c in zip(points, labels, strict=True)
                ],
            )
        else:
            values = outputs[:, 0]
            lo, hi = float(values.min()), float(values.max())
            unit = (values - lo) / (hi - lo) if hi - lo > 1e-12 else np.zeros_like(values)
            answer.update(
                png=render.data_url(render.heat_rgba(unit.reshape(n, n), alpha=False)),
                value_range=[render.sig(lo), render.sig(hi)],
                target=self.preprocessing.get("target"),
                grid=[
                    render.sig_list(row, 3) for row in values.reshape(n, n)[GRID_PICK][:, GRID_PICK]
                ],
                points=[
                    [render.sig(a), render.sig(b), render.sig(c)]
                    for (a, b), c in zip(points, labels, strict=True)
                ],
            )
            std_fn = self.gp_std()
            if std_fn is not None:
                spread = np.concatenate(
                    [std_fn(chunk) for chunk in np.array_split((raw - mean) / std, 4)]
                )
                low_std, high_std = float(spread.min()), float(spread.max())
                unit_std = (spread - low_std) / max(high_std - low_std, 1e-12)
                answer["std"] = {
                    "png": render.data_url(render.heat_rgba(unit_std.reshape(n, n), alpha=False)),
                    "range": [render.sig(spread.min()), render.sig(spread.max())],
                }
        if self.demo:
            self.notes.append(DEMO_NOTE)
        return answer

    def method_activations(self) -> dict[str, Any]:
        import torch

        from . import neural

        self.require(family="neural")
        model, analysis, _ = self.neural()
        x, picked = self.net_input()
        node = self.request.get("node")
        if node is not None and node not in analysis.nodes:
            raise ExplainError("There is no brick with this id in the network.")
        nodes = neural.activations(model, analysis, x, node)
        answer: dict[str, Any] = {
            "kind": self.kind,
            "sample": self.sample_info(picked),
            "nodes": nodes,
        }
        if self.kind == "image":
            answer["image"] = render.data_url(
                render.shrink(render.unit_to_uint8(picked["unit"]), SHOW_SIDE)
            )
        if self.classification or self.regression:
            with torch.no_grad():
                out = neural.forward(model, x)
            if self.classification:
                answer["prediction"] = self.prediction(torch.softmax(out, dim=1)[0].numpy())
            elif self.kind == "tabular" and self.preprocessing.get("target_std") is not None:
                value = (
                    float(out[0, 0]) * self.preprocessing["target_std"]
                    + self.preprocessing["target_mean"]
                )
                answer["prediction"] = {"value": render.sig(value, 6)}
        if self.demo:
            self.notes.append(DEMO_NOTE)
        return answer

    def method_weights(self) -> dict[str, Any]:
        from . import neural

        self.require(family="neural")
        model, analysis, attrs = self.neural()
        return {
            "kind": self.kind,
            "filters": neural.first_conv_filters(model, analysis, attrs),
            "layers": neural.weight_histograms(model, analysis, attrs),
        }


def _pool_source(pool: dict[str, Any]) -> str:
    """Where the samples of the pool come from: "validation", or "test" or "training" for a
    run without validation data (see data.py)."""
    meta = pool.get("meta")
    source = meta.get("source") if isinstance(meta, dict) else None
    return source if source in ("validation", "test", "training") else "validation"


def _index(sample: Any) -> int:
    index = sample.get("index") if isinstance(sample, dict) else None
    if isinstance(index, bool) or not isinstance(index, int):
        raise ExplainError("Pick a sample: {'index': n} of the validation samples, or an input.")
    return index


def _chosen(predicted: np.ndarray, truth: np.ndarray) -> list[int]:
    """The first samples, then some more that the model gets wrong."""
    first = list(range(min(LIST_FIRST, len(truth))))
    wrong = [int(i) for i in np.flatnonzero(predicted != truth) if int(i) not in first]
    return first + wrong[:LIST_WRONG]


def _cpus() -> int:
    import os

    return os.cpu_count() or 1


def _thread_limit():
    from ..predict import _thread_limit as limit

    return limit()
