"""Brick DNN: build neural networks from bricks and export clean PyTorch code.

This package is framework-agnostic. It has no dependency on Django or PyTorch
and can be used on its own::

    from brickdnn import analyze, generate_code

    graph = {
        "nodes": [
            {"id": "in", "type": "Input", "params": {"shape": [784]}},
            {"id": "fc", "type": "Linear", "params": {"out_features": 10}},
        ],
        "edges": [{"source": "in", "target": "fc"}],
    }
    print(generate_code(graph))
"""

from .codegen import GeneratorOptions, generate_code
from .examples import EXAMPLES, get_example
from .graph import Analysis, GraphValidationError, analyze
from .layers import REGISTRY, Layer, LayerError, Param, layer_specs, register

__version__ = "1.0.0"

__all__ = [
    "EXAMPLES",
    "REGISTRY",
    "Analysis",
    "GeneratorOptions",
    "GraphValidationError",
    "Layer",
    "LayerError",
    "Param",
    "analyze",
    "generate_code",
    "get_example",
    "layer_specs",
    "register",
]
