"""Layer catalogue: every brick that can be placed on the canvas.

Each layer knows

* which parameters it takes (rendered as a form in the UI),
* how to infer its output shape from its input shape(s),
* how many trainable parameters it has, and
* which PyTorch code it turns into.

Shapes never include the batch dimension: an MNIST image is ``(1, 28, 28)``,
a sequence of 32 token ids is ``(32,)``.

Besides the neural layers there are *classical* bricks (scikit-learn): kernels for Gaussian
processes, whose output is a kernel instead of a tensor, and classical models (Gaussian
processes, support vector machines) that take the features straight from the Input layer.
See the end of this module and ``docs/graph-format.md``.

To add a new layer, subclass :class:`Layer`, fill in the class attributes and
decorate it with :func:`register`. See ``docs/adding-a-layer.md``.
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from typing import Any, ClassVar

Shape = tuple[int, ...]


class LayerError(ValueError):
    """Raised when a layer's parameters or inputs are invalid."""


# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #

PARAM_KINDS = ("int", "float", "bool", "select", "ints")


@dataclass(frozen=True)
class Param:
    """Description of a single layer parameter.

    ``kind`` is one of ``int``, ``float``, ``bool``, ``select`` (one of
    ``options``) or ``ints`` (a list of integers, e.g. a kernel size).
    ``auto`` parameters may be left empty and are then inferred from the
    input shape. ``show_if`` hides the field in the UI unless another
    parameter has the given value.
    """

    name: str
    label: str
    kind: str
    default: Any = None
    help: str = ""
    options: tuple[str, ...] = ()
    minimum: float | None = None
    maximum: float | None = None
    auto: bool = False
    length: int | None = None
    show_if: tuple[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "kind": self.kind,
            "default": list(self.default) if isinstance(self.default, tuple) else self.default,
            "help": self.help,
            "options": list(self.options),
            "min": self.minimum,
            "max": self.maximum,
            "auto": self.auto,
            "length": self.length,
            "show_if": (
                {"param": self.show_if[0], "value": self.show_if[1]} if self.show_if else None
            ),
        }

    # -- value coercion ---------------------------------------------------- #

    def coerce(self, value: Any) -> Any:
        """Validate ``value`` and convert it to the canonical Python type."""
        if value is None or value == "" or value == []:
            if self.auto:
                return None
            if self.default is None:
                raise LayerError(f"{self.label} is required.")
            value = self.default

        if self.kind == "int":
            return self._check_range(_to_int(value, self.label))
        if self.kind == "float":
            return self._check_range(_to_float(value, self.label))
        if self.kind == "bool":
            return _to_bool(value, self.label)
        if self.kind == "select":
            if value not in self.options:
                allowed = ", ".join(self.options)
                raise LayerError(f"{self.label} must be one of: {allowed}.")
            return value
        if self.kind == "ints":
            return self._coerce_ints(value)
        raise AssertionError(f"unknown param kind {self.kind!r}")  # pragma: no cover

    def _coerce_ints(self, value: Any) -> tuple[int, ...]:
        if isinstance(value, (int, float, str)) and not isinstance(value, bool):
            value = [value] * (self.length or 1)
        if not isinstance(value, (list, tuple)):
            raise LayerError(f"{self.label} must be a list of integers.")
        values = tuple(self._check_range(_to_int(v, self.label)) for v in value)
        if self.length is not None and len(values) != self.length:
            raise LayerError(f"{self.label} must have exactly {self.length} values.")
        if not 1 <= len(values) <= 6:
            raise LayerError(f"{self.label} must have between 1 and 6 values.")
        return values

    def _check_range(self, value: float) -> Any:
        if self.minimum is not None and value < self.minimum:
            raise LayerError(f"{self.label} must be at least {_fmt_num(self.minimum)}.")
        if self.maximum is not None and value > self.maximum:
            raise LayerError(f"{self.label} must be at most {_fmt_num(self.maximum)}.")
        return value


def _fmt_num(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


MAX_INT = 10**9


def _to_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise LayerError(f"{label} must be an integer.")
    result: int | None = None
    if isinstance(value, int):
        result = value
    elif isinstance(value, float) and value.is_integer():
        result = int(value)
    elif isinstance(value, str) and len(value) < 20:
        with contextlib.suppress(ValueError):
            result = int(value.strip())
    if result is None:
        raise LayerError(f"{label} must be an integer.")
    if abs(result) > MAX_INT:
        raise LayerError(f"{label} is too large.")
    return result


def _to_float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise LayerError(f"{label} must be a number.")
    if isinstance(value, (int, float)):
        result = float(value)
    elif isinstance(value, str):
        try:
            result = float(value.strip())
        except ValueError as exc:
            raise LayerError(f"{label} must be a number.") from exc
    else:
        raise LayerError(f"{label} must be a number.")
    if not math.isfinite(result):
        raise LayerError(f"{label} must be a finite number.")
    return result


def _to_bool(value: Any, label: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in ("true", "false"):
        return value.lower() == "true"
    raise LayerError(f"{label} must be true or false.")


# --------------------------------------------------------------------------- #
# Formatting helpers for generated code
# --------------------------------------------------------------------------- #


def fmt_size(values: tuple[int, ...]) -> str:
    """``(3, 3)`` -> ``"3"``; ``(3, 5)`` -> ``"(3, 5)"``."""
    if len(set(values)) == 1:
        return str(values[0])
    return "(" + ", ".join(str(v) for v in values) + ")"


def fmt_shape(shape: Shape) -> str:
    return "(" + ", ".join(str(s) for s in shape) + ")"


def fmt_float(value: float) -> str:
    return repr(float(value))


# --------------------------------------------------------------------------- #
# Layer base class and registry
# --------------------------------------------------------------------------- #

REGISTRY: dict[str, type[Layer]] = {}

CATEGORIES = (
    "Input",
    "Basic",
    "Convolution",
    "Normalization",
    "Sequence",
    "Merge",
    "Kernels",
    "Classical models",
)
# What flows out of a brick (and so along each of its connections): data, or a kernel of a
# Gaussian process.
OUTPUT_KINDS = ("tensor", "kernel")
# Neural layers (PyTorch), classical bricks (scikit-learn), or both (the Input layer).
FAMILIES = ("neural", "classical", "shared")


def register(cls: type[Layer]) -> type[Layer]:
    if cls.type in REGISTRY:
        raise ValueError(f"duplicate layer type {cls.type!r}")
    if cls.category not in CATEGORIES:
        raise ValueError(f"unknown category {cls.category!r}")
    REGISTRY[cls.type] = cls
    return cls


def get_layer_class(layer_type: str) -> type[Layer]:
    try:
        return REGISTRY[layer_type]
    except KeyError:
        raise LayerError(f"Unknown layer type '{layer_type}'.") from None


class Layer:
    """Base class for all layers.

    Subclasses override :meth:`infer` (shape inference + validation),
    :meth:`init_code` (module construction) and, where needed,
    :meth:`forward_lines` and :meth:`num_params`.
    """

    type: ClassVar[str]
    label: ClassVar[str]
    category: ClassVar[str]
    description: ClassVar[str]
    params: ClassVar[tuple[Param, ...]] = ()
    min_inputs: ClassVar[int] = 1
    max_inputs: ClassVar[int | None] = 1
    has_module: ClassVar[bool] = True
    input_dtypes: ClassVar[tuple[str, ...]] = ("float",)
    name_prefix: ClassVar[str] = "layer"
    output_kind: ClassVar[str] = "tensor"
    family: ClassVar[str] = "neural"

    def __init__(self, raw_params: dict[str, Any] | None = None):
        raw_params = raw_params or {}
        if not isinstance(raw_params, dict):
            raise LayerError("Parameters must be an object.")
        self.p: dict[str, Any] = {}
        for param in self.params:
            self.p[param.name] = param.coerce(raw_params.get(param.name))

    # -- metadata --------------------------------------------------------- #

    @classmethod
    def spec(cls) -> dict[str, Any]:
        return {
            "type": cls.type,
            "label": cls.label,
            "category": cls.category,
            "description": cls.description,
            "params": [p.to_dict() for p in cls.params],
            "min_inputs": cls.min_inputs,
            "max_inputs": cls.max_inputs,
            "family": cls.family,
            "output_kind": cls.output_kind,
            "input_slots": cls.input_slots(),
        }

    @classmethod
    def input_slots(cls) -> list[dict[str, Any]]:
        """What the brick takes: one entry per kind of incoming connection, with how many
        connections of that kind it needs (``min``) and accepts (``max``, None: any)."""
        if cls.max_inputs == 0:
            return []
        return [
            {
                "name": "input",
                "label": "Input",
                "kind": "tensor",
                "min": cls.min_inputs,
                "max": cls.max_inputs,
            }
        ]

    @classmethod
    def param(cls, name: str) -> Param:
        return next(p for p in cls.params if p.name == name)

    def name_hint(self) -> str:
        return self.name_prefix

    def param_visible(self, name: str) -> bool:
        """False if the parameter's ``show_if`` condition is not met (it has no effect then)."""
        param = next((p for p in self.params if p.name == name), None)
        if param is None or param.show_if is None:
            return True
        other, value = param.show_if
        return self.p.get(other) == value

    @property
    def output_dtype(self) -> str:
        return "float"

    # -- behaviour -------------------------------------------------------- #

    def infer(self, inputs: list[Shape]) -> Shape:
        """Validate the input shapes, fill in ``auto`` params, return the output shape."""
        raise NotImplementedError

    def num_params(self) -> int | None:
        return 0

    def init_code(self) -> str:
        """Python expression that constructs the ``nn.Module`` (may span lines)."""
        raise NotImplementedError

    def forward_lines(self, attr: str, inputs: list[str], out: str) -> list[str]:
        return [f"{out} = self.{attr}({inputs[0]})"]

    # -- helpers for subclasses ------------------------------------------- #

    def _auto(self, name: str, inferred: int, source: str) -> int:
        """Fill an ``auto`` parameter, or check that a user-given value matches."""
        value = self.p.get(name)
        if value is None:
            self.p[name] = inferred
            return inferred
        if value != inferred:
            label = self.param(name).label
            raise LayerError(
                f"{label} is set to {value}, but the incoming tensor has {inferred} {source}. "
                f"Clear the field to infer it automatically."
            )
        return value

    def _single(self, inputs: list[Shape], ranks: tuple[int, ...] | None = None) -> Shape:
        shape = inputs[0]
        if ranks is not None and len(shape) not in ranks:
            expected = " or ".join(_rank_name(r) for r in ranks)
            raise LayerError(
                f"{self.label} expects a {expected} input, but got shape {fmt_shape(shape)}."
            )
        return shape


def _rank_name(rank: int) -> str:
    return {
        1: "1-D (features)",
        2: "2-D (channels/sequence, features)",
        3: "3-D (channels, height, width)",
    }.get(rank, f"{rank}-D")


# --------------------------------------------------------------------------- #
# Input
# --------------------------------------------------------------------------- #


@register
class Input(Layer):
    type = "Input"
    label = "Input"
    category = "Input"
    description = (
        "Entry point of the network. The shape excludes the batch dimension, "
        "e.g. [784] for flat features, [3, 32, 32] for RGB images or [64] for 64 token ids."
    )
    params = (
        Param(
            "shape",
            "Shape",
            "ints",
            default=(784,),
            minimum=1,
            help="Tensor shape without the batch dimension.",
        ),
        Param(
            "dtype",
            "Data type",
            "select",
            default="float",
            options=("float", "long"),
            help="Use 'long' for integer token ids that feed an Embedding layer.",
        ),
    )
    min_inputs = 0
    max_inputs = 0
    has_module = False
    name_prefix = "x"
    family = "shared"

    def infer(self, inputs: list[Shape]) -> Shape:
        if len(self.p["shape"]) > 4:
            raise LayerError("Input shape can have at most 4 dimensions (excluding the batch).")
        return tuple(self.p["shape"])

    @property
    def output_dtype(self) -> str:
        return self.p["dtype"]


# --------------------------------------------------------------------------- #
# Basic layers
# --------------------------------------------------------------------------- #


@register
class Linear(Layer):
    type = "Linear"
    label = "Linear"
    category = "Basic"
    description = "Fully connected layer: y = xWᵀ + b. Applied to the last dimension."
    params = (
        Param(
            "in_features",
            "Input features",
            "int",
            auto=True,
            minimum=1,
            help="Leave empty to infer it from the input shape.",
        ),
        Param("out_features", "Output features", "int", default=64, minimum=1),
        Param("bias", "Bias", "bool", default=True),
    )
    name_prefix = "linear"

    def infer(self, inputs: list[Shape]) -> Shape:
        shape = self._single(inputs)
        self._auto("in_features", shape[-1], "features in its last dimension")
        return shape[:-1] + (self.p["out_features"],)

    def num_params(self) -> int:
        n_in, n_out = self.p["in_features"], self.p["out_features"]
        return n_in * n_out + (n_out if self.p["bias"] else 0)

    def init_code(self) -> str:
        bias = "" if self.p["bias"] else ", bias=False"
        return f"nn.Linear({self.p['in_features']}, {self.p['out_features']}{bias})"


@register
class Flatten(Layer):
    type = "Flatten"
    label = "Flatten"
    category = "Basic"
    description = "Flattens all dimensions except the batch into one feature vector."
    name_prefix = "flatten"

    def infer(self, inputs: list[Shape]) -> Shape:
        return (math.prod(self._single(inputs)),)

    def init_code(self) -> str:
        return "nn.Flatten()"


@register
class Reshape(Layer):
    type = "Reshape"
    label = "Reshape"
    category = "Basic"
    description = (
        "Reshapes the tensor (batch dimension excluded). One entry may be -1 to infer it, "
        "e.g. [784] → [1, 28, 28]."
    )
    params = (
        Param(
            "shape",
            "Target shape",
            "ints",
            default=(1, 28, 28),
            minimum=-1,
            help="Shape without the batch dimension. Use -1 once to infer a dimension.",
        ),
    )
    has_module = False
    name_prefix = "reshape"

    def infer(self, inputs: list[Shape]) -> Shape:
        total = math.prod(self._single(inputs))
        target = list(self.p["shape"])
        if 0 in target:
            raise LayerError("Target shape entries must be positive (or -1).")
        if target.count(-1) > 1:
            raise LayerError("Only one entry of the target shape may be -1.")
        known = math.prod(v for v in target if v != -1)
        if -1 in target:
            if total % known:
                raise LayerError(f"Cannot reshape {total} values into {fmt_shape(tuple(target))}.")
            target[target.index(-1)] = total // known
        if math.prod(target) != total:
            raise LayerError(
                f"Cannot reshape {fmt_shape(inputs[0])} ({total} values) "
                f"into {fmt_shape(tuple(target))} ({math.prod(target)} values)."
            )
        return tuple(target)

    def forward_lines(self, attr: str, inputs: list[str], out: str) -> list[str]:
        dims = ", ".join(str(v) for v in self.p["shape"])
        return [f"{out} = {inputs[0]}.reshape({inputs[0]}.size(0), {dims})"]


@register
class Dropout(Layer):
    type = "Dropout"
    label = "Dropout"
    category = "Basic"
    description = "Randomly zeroes elements with probability p during training (regularisation)."
    params = (Param("p", "Drop probability", "float", default=0.5, minimum=0.0, maximum=1.0),)
    name_prefix = "dropout"

    def infer(self, inputs: list[Shape]) -> Shape:
        return self._single(inputs)

    def init_code(self) -> str:
        return f"nn.Dropout(p={fmt_float(self.p['p'])})"


ACTIVATIONS: dict[str, str] = {
    "relu": "nn.ReLU()",
    "leaky_relu": "nn.LeakyReLU(0.01)",
    "gelu": "nn.GELU()",
    "elu": "nn.ELU()",
    "selu": "nn.SELU()",
    "silu": "nn.SiLU()",
    "mish": "nn.Mish()",
    "sigmoid": "nn.Sigmoid()",
    "tanh": "nn.Tanh()",
    "softplus": "nn.Softplus()",
    "softmax": "nn.Softmax(dim=-1)",
    "log_softmax": "nn.LogSoftmax(dim=-1)",
}


@register
class Activation(Layer):
    type = "Activation"
    label = "Activation"
    category = "Basic"
    description = "Element-wise non-linearity such as ReLU, GELU or Sigmoid."
    params = (Param("function", "Function", "select", default="relu", options=tuple(ACTIVATIONS)),)

    def name_hint(self) -> str:
        return self.p["function"]

    def infer(self, inputs: list[Shape]) -> Shape:
        return self._single(inputs)

    def init_code(self) -> str:
        return ACTIVATIONS[self.p["function"]]


# --------------------------------------------------------------------------- #
# Convolution & pooling
# --------------------------------------------------------------------------- #


class _ConvNd(Layer):
    dims: ClassVar[int]
    category = "Convolution"

    def infer(self, inputs: list[Shape]) -> Shape:
        shape = self._single(inputs, (self.dims + 1,))
        channels, spatial = shape[0], shape[1:]
        self._auto("in_channels", channels, "channels")
        kernel, stride = self.p["kernel_size"], self.p["stride"]

        if self.p["padding"] == "same":
            if any(s != 1 for s in stride):
                raise LayerError("'same' padding only works with stride 1. Use custom padding.")
            return (self.p["out_channels"], *spatial)

        pad = (0,) * self.dims if self.p["padding"] == "valid" else self.p["padding_size"]
        out = []
        for size, k, s, p in zip(spatial, kernel, stride, pad, strict=True):
            if size + 2 * p < k:
                raise LayerError(
                    f"Kernel size {k} is larger than the (padded) input size {size + 2 * p}."
                )
            out.append((size + 2 * p - k) // s + 1)
        return (self.p["out_channels"], *out)

    def num_params(self) -> int:
        weights = self.p["out_channels"] * self.p["in_channels"] * math.prod(self.p["kernel_size"])
        return weights + (self.p["out_channels"] if self.p["bias"] else 0)

    def init_code(self) -> str:
        args = [
            str(self.p["in_channels"]),
            str(self.p["out_channels"]),
            f"kernel_size={fmt_size(self.p['kernel_size'])}",
        ]
        if any(s != 1 for s in self.p["stride"]):
            args.append(f"stride={fmt_size(self.p['stride'])}")
        if self.p["padding"] == "same":
            args.append("padding='same'")
        elif self.p["padding"] == "custom" and any(self.p["padding_size"]):
            args.append(f"padding={fmt_size(self.p['padding_size'])}")
        if not self.p["bias"]:
            args.append("bias=False")
        return f"nn.Conv{self.dims}d({', '.join(args)})"


def _conv_params(dims: int, out_channels: int) -> tuple[Param, ...]:
    return (
        Param(
            "in_channels",
            "Input channels",
            "int",
            auto=True,
            minimum=1,
            help="Leave empty to infer it from the input shape.",
        ),
        Param("out_channels", "Output channels (filters)", "int", default=out_channels, minimum=1),
        Param("kernel_size", "Kernel size", "ints", default=(3,) * dims, minimum=1, length=dims),
        Param("stride", "Stride", "ints", default=(1,) * dims, minimum=1, length=dims),
        Param(
            "padding",
            "Padding",
            "select",
            default="same",
            options=("same", "valid", "custom"),
            help="'same' keeps the spatial size, 'valid' adds no padding.",
        ),
        Param(
            "padding_size",
            "Padding size",
            "ints",
            default=(1,) * dims,
            minimum=0,
            length=dims,
            show_if=("padding", "custom"),
        ),
        Param("bias", "Bias", "bool", default=True),
    )


@register
class Conv1d(_ConvNd):
    type = "Conv1d"
    label = "Conv1d"
    dims = 1
    description = "1-D convolution over (channels, length), e.g. for audio or time series."
    params = _conv_params(1, 32)
    name_prefix = "conv1d"


@register
class Conv2d(_ConvNd):
    type = "Conv2d"
    label = "Conv2d"
    dims = 2
    description = "2-D convolution over (channels, height, width), the workhorse of image models."
    params = _conv_params(2, 32)
    name_prefix = "conv2d"


class _PoolNd(Layer):
    dims: ClassVar[int]
    mode: ClassVar[str]
    category = "Convolution"

    def infer(self, inputs: list[Shape]) -> Shape:
        shape = self._single(inputs, (self.dims + 1,))
        kernel = self.p["kernel_size"]
        if self.p["stride"] is None:
            self.p["stride"] = kernel
        out = []
        for size, k, s in zip(shape[1:], kernel, self.p["stride"], strict=True):
            if size < k:
                raise LayerError(f"Kernel size {k} is larger than the input size {size}.")
            out.append((size - k) // s + 1)
        return (shape[0], *out)

    def init_code(self) -> str:
        args = [f"kernel_size={fmt_size(self.p['kernel_size'])}"]
        if self.p["stride"] != self.p["kernel_size"]:
            args.append(f"stride={fmt_size(self.p['stride'])}")
        return f"nn.{self.mode}Pool{self.dims}d({', '.join(args)})"


def _pool_params(dims: int) -> tuple[Param, ...]:
    return (
        Param("kernel_size", "Kernel size", "ints", default=(2,) * dims, minimum=1, length=dims),
        Param(
            "stride",
            "Stride",
            "ints",
            auto=True,
            minimum=1,
            length=dims,
            help="Leave empty to use the kernel size.",
        ),
    )


@register
class MaxPool1d(_PoolNd):
    type = "MaxPool1d"
    label = "MaxPool1d"
    dims = 1
    mode = "Max"
    description = "Downsamples (channels, length) by taking the maximum in each window."
    params = _pool_params(1)
    name_prefix = "maxpool1d"


@register
class MaxPool2d(_PoolNd):
    type = "MaxPool2d"
    label = "MaxPool2d"
    dims = 2
    mode = "Max"
    description = "Downsamples (channels, height, width) by taking the maximum in each window."
    params = _pool_params(2)
    name_prefix = "maxpool2d"


@register
class AvgPool2d(_PoolNd):
    type = "AvgPool2d"
    label = "AvgPool2d"
    dims = 2
    mode = "Avg"
    description = "Downsamples (channels, height, width) by averaging each window."
    params = _pool_params(2)
    name_prefix = "avgpool2d"


@register
class AdaptiveAvgPool2d(Layer):
    type = "AdaptiveAvgPool2d"
    label = "AdaptiveAvgPool2d"
    category = "Convolution"
    description = (
        "Averages each channel down to a fixed output size regardless of the input size. "
        "Output size 1×1 gives global average pooling."
    )
    params = (Param("output_size", "Output size", "ints", default=(1, 1), minimum=1, length=2),)
    name_prefix = "avgpool"

    def infer(self, inputs: list[Shape]) -> Shape:
        shape = self._single(inputs, (3,))
        return (shape[0], *self.p["output_size"])

    def init_code(self) -> str:
        return f"nn.AdaptiveAvgPool2d({fmt_size(self.p['output_size'])})"


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #


class _BatchNorm(Layer):
    category = "Normalization"
    ranks: ClassVar[tuple[int, ...]]
    params = (
        Param(
            "num_features",
            "Features / channels",
            "int",
            auto=True,
            minimum=1,
            help="Leave empty to infer it from the input shape.",
        ),
    )

    def infer(self, inputs: list[Shape]) -> Shape:
        shape = self._single(inputs, self.ranks)
        self._auto("num_features", shape[0], "channels")
        return shape

    def num_params(self) -> int:
        return 2 * self.p["num_features"]

    def init_code(self) -> str:
        return f"nn.{self.type}({self.p['num_features']})"


@register
class BatchNorm1d(_BatchNorm):
    type = "BatchNorm1d"
    label = "BatchNorm1d"
    description = "Batch normalisation for (features) or (channels, length) inputs."
    ranks = (1, 2)
    name_prefix = "batchnorm1d"


@register
class BatchNorm2d(_BatchNorm):
    type = "BatchNorm2d"
    label = "BatchNorm2d"
    description = "Batch normalisation for (channels, height, width) inputs."
    ranks = (3,)
    name_prefix = "batchnorm2d"


@register
class LayerNorm(Layer):
    type = "LayerNorm"
    label = "LayerNorm"
    category = "Normalization"
    description = "Normalises over the last dimension. Standard in Transformers."
    params = (
        Param(
            "normalized_shape",
            "Features",
            "int",
            auto=True,
            minimum=1,
            help="Leave empty to infer it from the input shape.",
        ),
    )
    name_prefix = "layernorm"

    def infer(self, inputs: list[Shape]) -> Shape:
        shape = self._single(inputs)
        self._auto("normalized_shape", shape[-1], "features in its last dimension")
        return shape

    def num_params(self) -> int:
        return 2 * self.p["normalized_shape"]

    def init_code(self) -> str:
        return f"nn.LayerNorm({self.p['normalized_shape']})"


# --------------------------------------------------------------------------- #
# Sequence models
# --------------------------------------------------------------------------- #


@register
class Embedding(Layer):
    type = "Embedding"
    label = "Embedding"
    category = "Sequence"
    description = "Maps integer token ids to dense vectors. Needs an Input with data type 'long'."
    params = (
        Param("num_embeddings", "Vocabulary size", "int", default=10000, minimum=1),
        Param("embedding_dim", "Embedding size", "int", default=128, minimum=1),
    )
    input_dtypes = ("long",)
    name_prefix = "embedding"

    def infer(self, inputs: list[Shape]) -> Shape:
        return self._single(inputs) + (self.p["embedding_dim"],)

    def num_params(self) -> int:
        return self.p["num_embeddings"] * self.p["embedding_dim"]

    def init_code(self) -> str:
        return f"nn.Embedding({self.p['num_embeddings']}, {self.p['embedding_dim']})"


@register
class PositionalEmbedding(Layer):
    type = "PositionalEmbedding"
    label = "PositionalEmbedding"
    category = "Sequence"
    description = (
        "Adds a learned position vector to every step of a (sequence, features) tensor, "
        "so that Transformers know the order of the tokens."
    )
    params = (
        Param("max_len", "Max. sequence length", "int", default=512, minimum=1),
        Param(
            "embedding_dim",
            "Embedding size",
            "int",
            auto=True,
            minimum=1,
            help="Leave empty to infer it from the input shape.",
        ),
    )
    name_prefix = "pos_embedding"

    def infer(self, inputs: list[Shape]) -> Shape:
        shape = self._single(inputs, (2,))
        if shape[0] > self.p["max_len"]:
            raise LayerError(
                f"Sequence length {shape[0]} exceeds the maximum length {self.p['max_len']}."
            )
        self._auto("embedding_dim", shape[1], "features")
        return shape

    def num_params(self) -> int:
        return self.p["max_len"] * self.p["embedding_dim"]

    def init_code(self) -> str:
        return f"nn.Embedding({self.p['max_len']}, {self.p['embedding_dim']})"

    def forward_lines(self, attr: str, inputs: list[str], out: str) -> list[str]:
        x = inputs[0]
        return [f"{out} = {x} + self.{attr}(torch.arange({x}.size(1), device={x}.device))"]


class _Recurrent(Layer):
    category = "Sequence"
    gates: ClassVar[int]
    params = (
        Param(
            "input_size",
            "Input features",
            "int",
            auto=True,
            minimum=1,
            help="Leave empty to infer it from the input shape.",
        ),
        Param("hidden_size", "Hidden size", "int", default=128, minimum=1),
        Param("num_layers", "Stacked layers", "int", default=1, minimum=1, maximum=32),
        Param("bidirectional", "Bidirectional", "bool", default=False),
        Param(
            "return_sequences",
            "Return sequences",
            "bool",
            default=False,
            help="Output every time step (sequence, features) instead of only the last one.",
        ),
    )

    @property
    def _out_features(self) -> int:
        return self.p["hidden_size"] * (2 if self.p["bidirectional"] else 1)

    def infer(self, inputs: list[Shape]) -> Shape:
        shape = self._single(inputs, (2,))
        self._auto("input_size", shape[1], "features")
        if self.p["return_sequences"]:
            return (shape[0], self._out_features)
        return (self._out_features,)

    def num_params(self) -> int:
        hidden, directions = self.p["hidden_size"], 2 if self.p["bidirectional"] else 1
        total, n_in = 0, self.p["input_size"]
        for _ in range(self.p["num_layers"]):
            per_direction = self.gates * hidden * (n_in + hidden) + 2 * self.gates * hidden
            total += directions * per_direction
            n_in = hidden * directions
        return total

    def init_code(self) -> str:
        args = [str(self.p["input_size"]), str(self.p["hidden_size"])]
        if self.p["num_layers"] != 1:
            args.append(f"num_layers={self.p['num_layers']}")
        args.append("batch_first=True")
        if self.p["bidirectional"]:
            args.append("bidirectional=True")
        return f"nn.{self.type}({', '.join(args)})"

    def forward_lines(self, attr: str, inputs: list[str], out: str) -> list[str]:
        if self.p["return_sequences"]:
            return [f"{out}, _ = self.{attr}({inputs[0]})"]
        # Use the final hidden state of the last layer. For bidirectional layers this
        # combines the forward direction (after the last step) with the backward direction
        # (after the first step), which `output[:, -1]` would not.
        state = "(hidden, _)" if self.type == "LSTM" else "hidden"
        lines = [f"_, {state} = self.{attr}({inputs[0]})"]
        if self.p["bidirectional"]:
            lines.append(f"{out} = torch.cat([hidden[-2], hidden[-1]], dim=-1)")
        else:
            lines.append(f"{out} = hidden[-1]")
        return lines


@register
class RNN(_Recurrent):
    type = "RNN"
    label = "RNN"
    description = "Vanilla recurrent layer over a (sequence, features) input."
    gates = 1
    name_prefix = "rnn"


@register
class LSTM(_Recurrent):
    type = "LSTM"
    label = "LSTM"
    description = "Long short-term memory layer over a (sequence, features) input."
    gates = 4
    name_prefix = "lstm"


@register
class GRU(_Recurrent):
    type = "GRU"
    label = "GRU"
    description = "Gated recurrent unit layer over a (sequence, features) input."
    gates = 3
    name_prefix = "gru"


@register
class TransformerEncoder(Layer):
    type = "TransformerEncoder"
    label = "TransformerEncoder"
    category = "Sequence"
    description = "Stack of self-attention encoder blocks over a (sequence, features) input."
    params = (
        Param(
            "d_model",
            "Model dimension",
            "int",
            auto=True,
            minimum=1,
            help="Leave empty to infer it from the input shape.",
        ),
        Param("nhead", "Attention heads", "int", default=8, minimum=1),
        Param("dim_feedforward", "Feed-forward size", "int", default=512, minimum=1),
        Param("num_layers", "Stacked layers", "int", default=2, minimum=1, maximum=64),
        Param("dropout", "Dropout", "float", default=0.1, minimum=0.0, maximum=1.0),
    )
    name_prefix = "encoder"

    def infer(self, inputs: list[Shape]) -> Shape:
        shape = self._single(inputs, (2,))
        d_model = self._auto("d_model", shape[1], "features")
        if d_model % self.p["nhead"]:
            raise LayerError(
                f"Model dimension {d_model} must be divisible by the number of "
                f"attention heads ({self.p['nhead']})."
            )
        return shape

    def num_params(self) -> int:
        d, ff = self.p["d_model"], self.p["dim_feedforward"]
        per_layer = 4 * d * d + 2 * d * ff + 9 * d + ff
        return self.p["num_layers"] * per_layer

    def init_code(self) -> str:
        p = self.p
        return (
            "nn.TransformerEncoder(\n"
            "    nn.TransformerEncoderLayer(\n"
            f"        d_model={p['d_model']}, nhead={p['nhead']}, "
            f"dim_feedforward={p['dim_feedforward']},\n"
            f"        dropout={fmt_float(p['dropout'])}, batch_first=True,\n"
            "    ),\n"
            f"    num_layers={p['num_layers']},\n"
            ")"
        )


SEQUENCE_POOL = {
    "mean": "{x}.mean(dim=1)",
    "max": "{x}.amax(dim=1)",
    "first": "{x}[:, 0]",
    "last": "{x}[:, -1]",
}


@register
class SequencePool(Layer):
    type = "SequencePool"
    label = "SequencePool"
    category = "Sequence"
    description = (
        "Reduces a (sequence, features) tensor to (features) by taking the mean, max, "
        "first or last step. Typical before a classification head."
    )
    params = (Param("mode", "Mode", "select", default="mean", options=tuple(SEQUENCE_POOL)),)
    has_module = False
    name_prefix = "pooled"

    def infer(self, inputs: list[Shape]) -> Shape:
        shape = self._single(inputs)
        if len(shape) < 2:
            raise LayerError(
                f"SequencePool expects a (sequence, features) input, got {fmt_shape(shape)}."
            )
        return shape[1:]

    def forward_lines(self, attr: str, inputs: list[str], out: str) -> list[str]:
        return [f"{out} = " + SEQUENCE_POOL[self.p["mode"]].format(x=inputs[0])]


# --------------------------------------------------------------------------- #
# Merge layers (multiple inputs)
# --------------------------------------------------------------------------- #


@register
class Add(Layer):
    type = "Add"
    label = "Add"
    category = "Merge"
    description = "Element-wise sum of two or more tensors of equal shape (residual connections)."
    min_inputs = 2
    max_inputs = None
    has_module = False
    name_prefix = "added"

    def infer(self, inputs: list[Shape]) -> Shape:
        if len(set(inputs)) != 1:
            shapes = ", ".join(fmt_shape(s) for s in inputs)
            raise LayerError(f"Add needs inputs of identical shape, got {shapes}.")
        return inputs[0]

    def forward_lines(self, attr: str, inputs: list[str], out: str) -> list[str]:
        return [f"{out} = " + " + ".join(inputs)]


@register
class Concat(Layer):
    type = "Concat"
    label = "Concat"
    category = "Merge"
    description = (
        "Concatenates two or more tensors along one dimension "
        "(batch excluded; -1 = last dimension). Inputs are joined in connection order."
    )
    params = (Param("dim", "Dimension", "int", default=-1, minimum=-6, maximum=5),)
    min_inputs = 2
    max_inputs = None
    has_module = False
    name_prefix = "concat"

    def _axis(self, rank: int) -> int:
        dim = self.p["dim"]
        if not -rank <= dim < rank:
            raise LayerError(f"Dimension {dim} is out of range for {rank}-D inputs.")
        return dim % rank

    def infer(self, inputs: list[Shape]) -> Shape:
        ranks = {len(s) for s in inputs}
        if len(ranks) != 1:
            raise LayerError("Concat needs inputs with the same number of dimensions.")
        axis = self._axis(ranks.pop())
        for shape in inputs[1:]:
            if shape[:axis] + shape[axis + 1 :] != inputs[0][:axis] + inputs[0][axis + 1 :]:
                shapes = ", ".join(fmt_shape(s) for s in inputs)
                raise LayerError(
                    f"Concat inputs must match in every dimension except {self.p['dim']}, "
                    f"got {shapes}."
                )
        total = sum(s[axis] for s in inputs)
        return inputs[0][:axis] + (total,) + inputs[0][axis + 1 :]

    def forward_lines(self, attr: str, inputs: list[str], out: str) -> list[str]:
        dim = self.p["dim"]
        torch_dim = dim + 1 if dim >= 0 else dim  # skip the batch dimension
        return [f"{out} = torch.cat([{', '.join(inputs)}], dim={torch_dim})"]


# --------------------------------------------------------------------------- #
# Kernels of Gaussian processes (scikit-learn)
# --------------------------------------------------------------------------- #

# scikit-learn optimises kernel hyperparameters within these bounds unless told otherwise.
DEFAULT_BOUNDS = (1e-5, 1e5)
HYPERPARAMETER_MODES = ("optimize", "custom bounds", "fixed")


@dataclass(frozen=True)
class KernelExpr:
    """A scikit-learn kernel as code: a call such as ``RBF(length_scale=1.0)`` (``name``
    and ``args``, a class of ``sklearn.gaussian_process.kernels``) or the sum or product
    (``op`` "+" or "*") of other kernels (``terms``)."""

    name: str = ""
    args: tuple[str, ...] = ()
    op: str = ""
    terms: tuple[KernelExpr, ...] = ()
    hint: str = "kernel"  # variable name for this kernel in generated code
    hyperparameters: int = 0  # hyperparameters of this call that the GP optimises

    @property
    def precedence(self) -> int:
        """3 for a call, 2 for a product, 1 for a sum (like Python's operators)."""
        return {"": 3, "*": 2, "+": 1}[self.op]

    def leaves(self) -> list[KernelExpr]:
        """The calls, in the order in which :meth:`code` writes them."""
        if not self.op:
            return [self]
        return [leaf for term in self.terms for leaf in term.leaves()]

    def call(self) -> str:
        return f"{self.name}({', '.join(self.args)})"

    def code(self, names: list[str] | None = None) -> str:
        """Python code for the kernel; ``names`` replace the calls (one per leaf, in order)."""
        return self._code(list(names) if names is not None else None)

    def _code(self, names: list[str] | None) -> str:
        if not self.op:
            return names.pop(0) if names is not None else self.call()
        parts = []
        for term in self.terms:
            text = term._code(names)
            parts.append(f"({text})" if term.precedence < self.precedence else text)
        return f" {self.op} ".join(parts)

    def split(self, names: list[str] | None = None) -> list[str]:
        """The code of the top-level terms with their operators, for line breaks:
        ``["a * b", "+ c"]``."""
        if not self.op:
            return [self.code(names)]
        queue = list(names) if names is not None else None
        parts = []
        for index, term in enumerate(self.terms):
            text = term._code(queue)
            text = f"({text})" if term.precedence < self.precedence else text
            parts.append(text if index == 0 else f"{self.op} {text}")
        return parts

    def classes(self) -> list[str]:
        """The kernel classes the code uses (sorted), for the import line."""
        return sorted({leaf.name for leaf in self.leaves()})

    def count(self) -> int:
        """Hyperparameters of the whole kernel that the GP optimises."""
        return sum(leaf.hyperparameters for leaf in self.leaves())


def combine(op: str, terms: list[KernelExpr]) -> KernelExpr:
    return KernelExpr(op=op, terms=tuple(terms))


def _bounds_params(name: str, label: str) -> tuple[Param, ...]:
    return (
        Param(
            f"{name}_min",
            f"{label}: lower bound",
            "float",
            default=DEFAULT_BOUNDS[0],
            show_if=("hyperparameters", "custom bounds"),
        ),
        Param(
            f"{name}_max",
            f"{label}: upper bound",
            "float",
            default=DEFAULT_BOUNDS[1],
            show_if=("hyperparameters", "custom bounds"),
        ),
    )


HYPERPARAMETERS_PARAM = Param(
    "hyperparameters",
    "Hyperparameters",
    "select",
    default="optimize",
    options=HYPERPARAMETER_MODES,
    help=(
        "optimize: the Gaussian process fits them to the training data, starting from the "
        "values above (within 1e-05 … 1e+05); custom bounds: within the bounds below; "
        "fixed: they keep these values."
    ),
)


class Kernel(Layer):
    """A kernel brick: no data input, its output is a kernel for a Gaussian process.

    ``hyper`` lists the names of the kernel's hyperparameters, which are also the names of
    their parameters (initial values) and of their scikit-learn arguments.
    """

    category = "Kernels"
    family = "classical"
    output_kind = "kernel"
    min_inputs = 0
    max_inputs = 0
    has_module = False
    sklearn_name: ClassVar[str]
    hyper: ClassVar[tuple[str, ...]] = ()

    def __init__(self, raw_params: dict[str, Any] | None = None):
        super().__init__(raw_params)
        mode = self.p.get("hyperparameters", "fixed")
        for name in self.hyper:
            label = self.param(name).label
            value = self.p[name]
            if value <= 0:
                raise LayerError(f"{label} must be greater than 0.")
            if mode == "optimize" and not DEFAULT_BOUNDS[0] <= value <= DEFAULT_BOUNDS[1]:
                raise LayerError(
                    f"{label} {value:g} is outside the bounds 1e-05 … 1e+05 within which the "
                    "Gaussian process optimises it. Choose 'custom bounds' or 'fixed'."
                )
            if mode == "custom bounds":
                low, high = self.p[f"{name}_min"], self.p[f"{name}_max"]
                if not 0 < low < high:
                    raise LayerError(
                        f"The bounds of {label} must be positive, the lower one below the "
                        "upper one."
                    )
                if not low <= value <= high:
                    raise LayerError(f"{label} {value:g} is outside its bounds {low:g} … {high:g}.")

    def infer(self, inputs: list[Shape]) -> Shape:  # pragma: no cover - kernels have no shape
        raise LayerError(f"{self.label} is a kernel, not a layer.")

    def num_params(self) -> int:
        """The hyperparameters the Gaussian process optimises (none if they are fixed)."""
        return 0 if self.p.get("hyperparameters") == "fixed" else len(self.hyper)

    def extra_args(self) -> list[str]:
        """Arguments besides the hyperparameters, e.g. ``nu=1.5``."""
        return []

    def kernel(self, terms: list[KernelExpr]) -> KernelExpr:
        """The kernel of this brick (``terms``: the kernels connected to it)."""
        args = [f"{name}={fmt_float(self.p[name])}" for name in self.hyper]
        args += self.extra_args()
        mode = self.p.get("hyperparameters")
        for name in self.hyper:
            if mode == "fixed":
                args.append(f'{name}_bounds="fixed"')
            elif mode == "custom bounds":
                low, high = fmt_float(self.p[f"{name}_min"]), fmt_float(self.p[f"{name}_max"])
                args.append(f"{name}_bounds=({low}, {high})")
        return KernelExpr(
            name=self.sklearn_name,
            args=tuple(args),
            hint=self.name_prefix,
            hyperparameters=self.num_params(),
        )


def _length_scale() -> Param:
    return Param(
        "length_scale",
        "Length scale",
        "float",
        default=1.0,
        help="How far apart (in standardised feature units) inputs still look alike. "
        "Initial value: the Gaussian process optimises it unless it is fixed.",
    )


@register
class RBF(Kernel):
    type = "RBF"
    label = "RBF kernel"
    description = (
        "Radial basis function (squared exponential) kernel for smooth functions. "
        "The most common kernel of Gaussian processes."
    )
    sklearn_name = "RBF"
    hyper = ("length_scale",)
    params = (
        _length_scale(),
        HYPERPARAMETERS_PARAM,
        *_bounds_params("length_scale", "Length scale"),
    )
    name_prefix = "rbf"


@register
class Matern(Kernel):
    type = "Matern"
    label = "Matérn kernel"
    description = (
        "Like RBF, but for rougher functions: ν = 0.5 is the exponential kernel, "
        "1.5 gives once and 2.5 twice differentiable functions."
    )
    sklearn_name = "Matern"
    hyper = ("length_scale",)
    params = (
        _length_scale(),
        Param(
            "nu",
            "Smoothness ν",
            "select",
            default="1.5",
            options=("0.5", "1.5", "2.5"),
            help="Smaller values give rougher functions (ν is not optimised).",
        ),
        HYPERPARAMETERS_PARAM,
        *_bounds_params("length_scale", "Length scale"),
    )
    name_prefix = "matern"

    def extra_args(self) -> list[str]:
        return [f"nu={self.p['nu']}"]


@register
class RationalQuadratic(Kernel):
    type = "RationalQuadratic"
    label = "Rational quadratic kernel"
    description = "A mixture of RBF kernels of different length scales; alpha weights the mixture."
    sklearn_name = "RationalQuadratic"
    hyper = ("length_scale", "alpha")
    params = (
        _length_scale(),
        Param(
            "alpha",
            "Alpha",
            "float",
            default=1.0,
            help="Weighting of large and small length scales (large values: like RBF).",
        ),
        HYPERPARAMETERS_PARAM,
        *_bounds_params("length_scale", "Length scale"),
        *_bounds_params("alpha", "Alpha"),
    )
    name_prefix = "rational_quadratic"


@register
class ExpSineSquared(Kernel):
    type = "ExpSineSquared"
    label = "Periodic kernel"
    description = (
        "Exp-sine-squared kernel for periodic functions: the periodicity is the distance "
        "after which the function repeats."
    )
    sklearn_name = "ExpSineSquared"
    hyper = ("length_scale", "periodicity")
    params = (
        _length_scale(),
        Param(
            "periodicity",
            "Periodicity",
            "float",
            default=1.0,
            help="Period in standardised feature units (the features are standardised "
            "before the kernel sees them).",
        ),
        HYPERPARAMETERS_PARAM,
        *_bounds_params("length_scale", "Length scale"),
        *_bounds_params("periodicity", "Periodicity"),
    )
    name_prefix = "periodic"


@register
class DotProduct(Kernel):
    type = "DotProduct"
    label = "Dot product kernel"
    description = (
        "Linear kernel σ₀² + x·x′ (Bayesian linear regression). Multiply it with itself "
        "for polynomial trends."
    )
    sklearn_name = "DotProduct"
    hyper = ("sigma_0",)
    params = (
        Param("sigma_0", "Sigma₀", "float", default=1.0, help="Inhomogeneity (offset)."),
        HYPERPARAMETERS_PARAM,
        *_bounds_params("sigma_0", "Sigma₀"),
    )
    name_prefix = "dot_product"


@register
class WhiteKernel(Kernel):
    type = "WhiteKernel"
    label = "White noise kernel"
    description = (
        "Independent noise on every sample. Add it (KernelSum) to learn how noisy the targets are."
    )
    sklearn_name = "WhiteKernel"
    hyper = ("noise_level",)
    params = (
        Param(
            "noise_level",
            "Noise level",
            "float",
            default=0.1,
            help="Noise variance (of the standardised targets if Normalize targets is on).",
        ),
        HYPERPARAMETERS_PARAM,
        *_bounds_params("noise_level", "Noise level"),
    )
    name_prefix = "white"


@register
class ConstantKernel(Kernel):
    type = "ConstantKernel"
    label = "Constant kernel"
    description = (
        "A constant: multiply another kernel with it (KernelProduct) to scale that kernel, "
        "add it (KernelSum) for a constant offset."
    )
    sklearn_name = "ConstantKernel"
    hyper = ("constant_value",)
    params = (
        Param("constant_value", "Constant", "float", default=1.0),
        HYPERPARAMETERS_PARAM,
        *_bounds_params("constant_value", "Constant"),
    )
    name_prefix = "constant"


class _KernelCombination(Kernel):
    min_inputs = 2
    max_inputs = None
    op: ClassVar[str]

    @classmethod
    def input_slots(cls) -> list[dict[str, Any]]:
        return [{"name": "kernels", "label": "Kernels", "kind": "kernel", "min": 2, "max": None}]

    def num_params(self) -> int:
        return 0

    def kernel(self, terms: list[KernelExpr]) -> KernelExpr:
        return combine(self.op, terms)


@register
class KernelSum(_KernelCombination):
    type = "KernelSum"
    label = "Kernel sum"
    description = (
        "Adds two or more kernels, e.g. RBF + WhiteKernel for a smooth function with noise."
    )
    op = "+"
    name_prefix = "kernel_sum"


@register
class KernelProduct(_KernelCombination):
    type = "KernelProduct"
    label = "Kernel product"
    description = (
        "Multiplies two or more kernels, e.g. ConstantKernel × RBF to scale a kernel, or "
        "RBF × ExpSineSquared for a slowly changing periodic pattern."
    )
    op = "*"
    name_prefix = "kernel_product"


# --------------------------------------------------------------------------- #
# Classical models (scikit-learn)
# --------------------------------------------------------------------------- #


def fmt_str(value: str) -> str:
    return '"' + value + '"'


class ClassicalModel(Layer):
    """A scikit-learn estimator. It takes a flat feature vector straight from the Input
    layer (Gaussian processes optionally also a kernel) and ends the network: its output is
    the prediction, one value (regression) or one class (classification) per sample.

    The analysis calls :meth:`setup` instead of :meth:`infer`.
    """

    category = "Classical models"
    family = "classical"
    has_module = False
    task: ClassVar[str]  # regression | classification
    estimator: ClassVar[str]  # the scikit-learn class
    module: ClassVar[str]  # where it is imported from
    takes_kernel: ClassVar[bool] = False
    # Which training row limit applies (brickdnn.training.limits): gaussian_process | svm
    scale: ClassVar[str]

    def __init__(self, raw_params: dict[str, Any] | None = None):
        super().__init__(raw_params)
        self.kernel: KernelExpr | None = None
        self.kernel_connected = False

    @classmethod
    def input_slots(cls) -> list[dict[str, Any]]:
        slots: list[dict[str, Any]] = [
            {"name": "features", "label": "Features", "kind": "tensor", "min": 1, "max": 1}
        ]
        if cls.takes_kernel:
            slots.append(
                {"name": "kernel", "label": "Kernel", "kind": "kernel", "min": 0, "max": 1}
            )
        return slots

    def infer(self, inputs: list[Shape]) -> Shape:  # pragma: no cover - see setup()
        raise LayerError(f"{self.label} is a classical model.")

    def setup(self, kernel: KernelExpr | None) -> Shape | None:
        """Remember the kernel (Gaussian processes) and return the output shape: ``(1,)``
        for regression, None for classification (one class per sample; the number of
        classes comes from the data)."""
        if self.takes_kernel:
            self.kernel_connected = kernel is not None
            self.kernel = kernel or self.default_kernel()
        return (1,) if self.task == "regression" else None

    def default_kernel(self) -> KernelExpr | None:
        return None

    def num_params(self) -> int | None:
        """Unknown before training (e.g. the support vectors of an SVM)."""
        return None

    def estimator_args(self) -> list[str]:
        """Keyword arguments of the estimator as code (a GP's kernel is ``kernel=kernel``)."""
        raise NotImplementedError


def _gp_default_kernel(noise: bool) -> KernelExpr:
    scaled = combine("*", [ConstantKernel().kernel([]), RBF().kernel([])])
    return combine("+", [scaled, WhiteKernel().kernel([])]) if noise else scaled


_RESTARTS = Param(
    "n_restarts_optimizer",
    "Optimizer restarts",
    "int",
    default=0,
    minimum=0,
    maximum=20,
    help="Fit the kernel hyperparameters again from this many random starting points and "
    "keep the best fit (slower, but less likely to get stuck).",
)


class _GaussianProcess(ClassicalModel):
    module = "sklearn.gaussian_process"
    takes_kernel = True
    max_inputs = 2
    scale = "gaussian_process"

    def num_params(self) -> int:
        """Hyperparameters of the default kernel (a connected kernel counts its own)."""
        if self.kernel is None or self.kernel_connected:
            return 0
        return self.kernel.count()

    def _restarts(self) -> list[str]:
        restarts = self.p["n_restarts_optimizer"]
        if not restarts:
            return []
        # The restarts start from random hyperparameters: a fixed seed keeps runs repeatable.
        return [f"n_restarts_optimizer={restarts}", "random_state=0"]


@register
class GaussianProcessRegressor(_GaussianProcess):
    type = "GaussianProcessRegressor"
    label = "Gaussian process regression"
    description = (
        "Predicts a number with its uncertainty. Connect the Input (features) and optionally "
        "a kernel; without one it uses ConstantKernel × RBF + WhiteKernel. Fitting takes "
        "O(n³): for up to a few thousand training rows."
    )
    task = "regression"
    estimator = "GaussianProcessRegressor"
    params = (
        Param(
            "alpha",
            "Alpha",
            "float",
            default=1e-10,
            minimum=0.0,
            help="Added to the diagonal of the kernel matrix: keeps the fit numerically "
            "stable. To learn the noise of the targets, add a WhiteKernel instead.",
        ),
        Param(
            "normalize_y",
            "Normalize targets",
            "bool",
            default=True,
            help="Standardise the targets before fitting (recommended unless they are "
            "already centred around 0 with variance 1).",
        ),
        _RESTARTS,
    )
    name_prefix = "gp"

    def default_kernel(self) -> KernelExpr:
        return _gp_default_kernel(noise=True)

    def estimator_args(self) -> list[str]:
        return [
            "kernel=kernel",
            f"alpha={fmt_float(self.p['alpha'])}",
            f"normalize_y={self.p['normalize_y']}",
            *self._restarts(),
        ]


@register
class GaussianProcessClassifier(_GaussianProcess):
    type = "GaussianProcessClassifier"
    label = "Gaussian process classification"
    description = (
        "Classifies with class probabilities (Laplace approximation; with more than two "
        "classes one Gaussian process per class). Connect the Input (features) and optionally "
        "a kernel; without one it uses ConstantKernel × RBF. The classes come from the data. "
        "Fitting takes O(n³): for up to a few thousand training rows."
    )
    task = "classification"
    estimator = "GaussianProcessClassifier"
    params = (_RESTARTS,)
    name_prefix = "gp"

    def default_kernel(self) -> KernelExpr:
        return _gp_default_kernel(noise=False)

    def estimator_args(self) -> list[str]:
        return ["kernel=kernel", *self._restarts()]


SVM_KERNELS = ("linear", "poly", "rbf", "sigmoid")


def _svm_params() -> tuple[Param, ...]:
    return (
        Param(
            "kernel",
            "Kernel",
            "select",
            default="rbf",
            options=SVM_KERNELS,
            help="The SVM's own kernel (kernel bricks only work with Gaussian processes).",
        ),
        Param(
            "C",
            "C (regularisation)",
            "float",
            default=1.0,
            help="Larger values fit the training data more closely, smaller ones regularise more.",
        ),
        Param(
            "gamma",
            "Gamma",
            "select",
            default="scale",
            options=("scale", "auto", "value"),
            help="Coefficient of the rbf, poly and sigmoid kernels: scale = 1 / (features × "
            "variance), auto = 1 / features, or a value.",
        ),
        Param("gamma_value", "Gamma value", "float", default=0.1, show_if=("gamma", "value")),
        Param(
            "degree",
            "Degree",
            "int",
            default=3,
            minimum=1,
            maximum=10,
            show_if=("kernel", "poly"),
        ),
        Param(
            "coef0",
            "Coef0",
            "float",
            default=0.0,
            help="Independent term of the poly and sigmoid kernels.",
        ),
    )


class _SupportVectorMachine(ClassicalModel):
    module = "sklearn.svm"
    scale = "svm"

    def __init__(self, raw_params: dict[str, Any] | None = None):
        super().__init__(raw_params)
        if self.p["C"] <= 0:
            raise LayerError("C must be greater than 0.")
        if self.p["gamma"] == "value" and self.p["gamma_value"] <= 0:
            raise LayerError("Gamma value must be greater than 0.")

    def svm_args(self) -> list[str]:
        kernel = self.p["kernel"]
        args = [f"kernel={fmt_str(kernel)}", f"C={fmt_float(self.p['C'])}"]
        if kernel != "linear":
            gamma = self.p["gamma"]
            value = fmt_float(self.p["gamma_value"]) if gamma == "value" else fmt_str(gamma)
            args.append(f"gamma={value}")
        if kernel == "poly":
            args.append(f"degree={self.p['degree']}")
        if kernel in ("poly", "sigmoid"):
            args.append(f"coef0={fmt_float(self.p['coef0'])}")
        return args


@register
class SVR(_SupportVectorMachine):
    type = "SVR"
    label = "Support vector regression"
    description = (
        "Predicts a number with an ε-insensitive support vector machine. Connect the Input "
        "(features). Fitting takes O(n²) or more: for up to tens of thousands of training "
        "rows."
    )
    task = "regression"
    estimator = "SVR"
    params = (
        *_svm_params(),
        Param(
            "epsilon",
            "Epsilon",
            "float",
            default=0.1,
            minimum=0.0,
            help="Errors smaller than this (in the units of the target) are not penalised.",
        ),
    )
    name_prefix = "svr"

    def estimator_args(self) -> list[str]:
        return [*self.svm_args(), f"epsilon={fmt_float(self.p['epsilon'])}"]


@register
class SVC(_SupportVectorMachine):
    type = "SVC"
    label = "Support vector classification"
    description = (
        "Classifies with a support vector machine; Platt scaling turns its scores into class "
        "probabilities. Connect the Input (features); the classes come from the data. "
        "Fitting takes O(n²) or more: for up to tens of thousands of training rows."
    )
    task = "classification"
    estimator = "SVC"
    params = (
        *_svm_params(),
        Param(
            "class_weight",
            "Class weights",
            "select",
            default="none",
            options=("none", "balanced"),
            help="balanced: weight the classes inversely to their frequency (for imbalanced data).",
        ),
    )
    name_prefix = "svc"

    def estimator_args(self) -> list[str]:
        args = self.svm_args()
        if self.p["class_weight"] == "balanced":
            args.append('class_weight="balanced"')
        return args


def layer_specs() -> list[dict[str, Any]]:
    """JSON-serialisable description of all layers, used by the UI."""
    return [cls.spec() for cls in REGISTRY.values()]
