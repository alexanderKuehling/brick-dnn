"""Write PyTorch weights as a safetensors file, without the ``safetensors`` package.

The format (https://github.com/huggingface/safetensors) is simple: 8 bytes with the length
of a JSON header (little-endian), the header (``{"<name>": {"dtype": "F32", "shape": [3, 4],
"data_offsets": [begin, end]}, ..., "__metadata__": {...}}``, padded with spaces to a
multiple of 8), then the raw little-endian bytes of every tensor. Unlike a ``torch.save``
file it holds no pickle, so it is safe to load from anywhere.

Used when a run is published to the Hugging Face Hub (brick_dnn_core/hub_publish.py). It
runs as a process of its own, so the web server never imports PyTorch::

    python -m brickdnn.training.safetensors_file weights.pt model.safetensors

prints ``{"tensors": 4, "size": 1234}`` (or ``{"error": "..."}``) as its last line.
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path
from typing import Any

# torch dtype name -> safetensors dtype
DTYPES = {
    "float64": "F64",
    "float32": "F32",
    "float16": "F16",
    "bfloat16": "BF16",
    "int64": "I64",
    "int32": "I32",
    "int16": "I16",
    "int8": "I8",
    "uint8": "U8",
    "bool": "BOOL",
}


def tensor_bytes(tensor: Any) -> tuple[str, list[int], bytes]:
    """The safetensors dtype, the shape and the little-endian bytes of a tensor."""
    import torch

    name = str(tensor.dtype).removeprefix("torch.")
    if name not in DTYPES:
        raise ValueError(f"Tensors of type {name} cannot be saved as safetensors.")
    value = tensor.detach().cpu().contiguous()
    if value.dtype == torch.bfloat16:  # NumPy has no bfloat16: the same 16 bits as int16
        value = value.view(torch.int16)
    array = value.numpy()
    if array.dtype.byteorder == ">":
        array = array.byteswap().view(array.dtype.newbyteorder("<"))
    return DTYPES[name], list(tensor.shape), array.tobytes()


def write(tensors: dict[str, Any], path: str | Path, metadata: dict[str, str] | None = None):
    """Save ``tensors`` (name -> torch.Tensor) as a safetensors file at ``path``."""
    header: dict[str, Any] = {}
    parts: list[bytes] = []
    offset = 0
    for name in sorted(tensors):
        dtype, shape, data = tensor_bytes(tensors[name])
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + len(data)],
        }
        parts.append(data)
        offset += len(data)
    if metadata:
        header["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}
    text = json.dumps(header, separators=(",", ":"), ensure_ascii=True).encode()
    text += b" " * (-len(text) % 8)
    target = Path(path)
    with target.open("wb") as out:
        out.write(struct.pack("<Q", len(text)))
        out.write(text)
        for data in parts:
            out.write(data)
    return target


def read_header(path: str | Path) -> dict[str, Any]:
    """The JSON header of a safetensors file (for checks and tests)."""
    with Path(path).open("rb") as source:
        (length,) = struct.unpack("<Q", source.read(8))
        return json.loads(source.read(length))


def state_dict_of(loaded: Any) -> dict[str, Any]:
    """The tensors of a weights file: a state_dict, or the ``state_dict`` of a checkpoint."""
    import torch

    if isinstance(loaded, dict) and isinstance(loaded.get("state_dict"), dict):
        loaded = loaded["state_dict"]
    if not isinstance(loaded, dict) or not loaded:
        raise ValueError("The file holds no state_dict.")
    tensors = {}
    for name, value in loaded.items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise ValueError("The state_dict holds something else than named tensors.")
        tensors[name] = value
    return tensors


def convert(source: str | Path, target: str | Path) -> dict[str, Any]:
    """Convert a ``torch.save`` state_dict (loaded with ``weights_only``) to safetensors."""
    import torch

    loaded = torch.load(source, map_location="cpu", weights_only=True)
    tensors = state_dict_of(loaded)
    write(tensors, target, {"format": "pt"})
    return {"tensors": len(tensors), "size": Path(target).stat().st_size}


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        print(json.dumps({"error": "usage: safetensors_file <weights.pt> <out.safetensors>"}))
        return 2
    try:
        result = convert(args[0], args[1])
    except Exception as exc:  # reported to the server, which then uploads the .pt file
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"[:500]}))
        return 1
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
