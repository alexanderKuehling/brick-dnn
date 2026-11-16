"""The generators of the built-in datasets that are generated instead of downloaded.

Every generated dataset (``BuiltinDataset.generated``) has an entry here under its id: a
module with a function for one image (``sample``) and one for a range of images
(``arrays``). The training (segdata.py) picks the generator by the dataset's id, so a new
generated dataset only needs its module, its entry in ``BUILTIN_DATASETS`` and a line here.
The modules import NumPy only when an image is generated.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .cells import cells_arrays, cells_sample
from .shapes import shapes_arrays, shapes_sample


@dataclass(frozen=True)
class Generator:
    # Image ``index`` at ``size`` × ``size``: (picture H × W × 3 uint8, mask H × W uint8).
    sample: Callable[..., tuple[Any, Any]]
    # Images ``start`` … ``start + count - 1``: (N × 3 × H × W uint8, N × H × W uint8).
    arrays: Callable[..., tuple[Any, Any]]


GENERATORS: dict[str, Generator] = {
    "shapes": Generator(sample=shapes_sample, arrays=shapes_arrays),
    "cells": Generator(sample=cells_sample, arrays=cells_arrays),
}


def generator_for(dataset_id: str) -> Generator | None:
    """The generator of a generated built-in dataset, None for other datasets."""
    return GENERATORS.get(dataset_id)
