# brickdnn (Python package)

The code generator behind [Brick DNN](https://github.com/alexanderKuehling/brick-dnn), usable on
its own. It has **no dependencies**: it turns a JSON description of a network into a
self-contained PyTorch script, infers every tensor shape on the way and reports mistakes
before you run anything.

```bash
pip install "git+https://github.com/alexanderKuehling/brick-dnn#subdirectory=backend"

brickdnn --list-layers
brickdnn --example cnn_mnist -o model.py      # start from a bundled example
brickdnn my_graph.json --check                # validate and print shapes
brickdnn my_graph.json -o model.py            # generate code
```

```python
from brickdnn import analyze, generate_code

graph = {
    "nodes": [
        {"id": "in", "type": "Input", "params": {"shape": [1, 28, 28]}},
        {"id": "flat", "type": "Flatten"},
        {"id": "fc", "type": "Linear", "params": {"out_features": 10}},
    ],
    "edges": [{"source": "in", "target": "flat"}, {"source": "flat", "target": "fc"}],
}

print(analyze(graph).nodes["fc"].shape)   # (10,)
print(generate_code(graph))
```

See the [main README](https://github.com/alexanderKuehling/brick-dnn#readme) for the graph
format and the list of layers.
