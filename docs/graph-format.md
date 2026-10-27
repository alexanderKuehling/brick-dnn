# Graph format

Brick DNN stores networks as plain JSON. The same document is used by the editor
(*File → Save as JSON file*), the REST API and the `brickdnn` command line tool.

```json
{
  "version": 1,
  "nodes": [
    { "id": "input", "type": "Input", "params": { "shape": [1, 28, 28] }, "position": { "x": 0, "y": 0 } },
    { "id": "flat", "type": "Flatten", "params": {} },
    { "id": "head", "type": "Linear", "params": { "out_features": 10 }, "name": "classifier" }
  ],
  "edges": [
    { "source": "input", "target": "flat" },
    { "source": "flat", "target": "head" }
  ]
}
```

A saved project wraps it as `{"name": "...", "graph": {...}}`; both forms are accepted
everywhere.

## Nodes

| Key        | Required | Meaning                                                                       |
| ---------- | -------- | ----------------------------------------------------------------------------- |
| `id`       | yes      | Unique string (numbers are converted to strings)                              |
| `type`     | yes      | Layer type, e.g. `Linear`. `brickdnn --list-layers` prints all of them         |
| `params`   | no       | Layer parameters. Missing values use the default                              |
| `name`     | no       | Attribute name in the generated code (sanitised, made unique)                  |
| `position` | no       | Canvas position, ignored by the generator                                     |

Any other keys are ignored, so tools can store extra information.

### Parameters

`GET /api/layers/` (or `brickdnn.layer_specs()`) describes every parameter:

| Field     | Meaning                                                                                |
| --------- | -------------------------------------------------------------------------------------- |
| `kind`    | `int`, `float`, `bool`, `select` (one of `options`) or `ints` (list of integers)        |
| `default` | Used when the value is missing, `null` or `""`                                          |
| `auto`    | If `true`, leave the value empty to infer it from the input shape (`in_features` …)     |
| `length`  | Fixed length of an `ints` list (e.g. `2` for a 2-D kernel size). A single number is expanded |
| `min`/`max` | Allowed range                                                                         |
| `show_if` | UI hint: only relevant if another parameter has a given value                           |

If you set an `auto` parameter explicitly and it does not match the incoming tensor, the
analysis reports an error instead of silently changing it.

## Edges

`{"source": "<node id>", "target": "<node id>"}`. Duplicates are ignored.

- Layers take exactly one input, except `Input` (none) and `Add`/`Concat` (two or more).
- `Concat` joins its inputs **in the order of the edges** in the document.
- Nodes without outgoing edges are the outputs of the network. With several outputs,
  `forward()` returns a tuple in node order.
- The graph must be acyclic and every layer must be reachable from an `Input`.

## Shapes

Shapes never include the batch dimension:

| Data                     | `Input` shape | dtype   |
| ------------------------ | ------------- | ------- |
| Flat feature vector      | `[784]`       | `float` |
| Grayscale / RGB image    | `[1, 28, 28]`, `[3, 32, 32]` | `float` |
| Time series              | `[channels, length]` for `Conv1d`, `[length, features]` for RNNs | `float` |
| Token ids                | `[sequence_length]` → `Embedding` | `long` |

Recurrent layers and `TransformerEncoder` expect `(sequence, features)` and use
`batch_first=True` in the generated code. `Concat`'s `dim` also excludes the batch dimension
(`0` = first feature dimension, `-1` = last).

## Generated code

`generate_code(graph, options)` returns a script with

- the model class (`class_name`, default `NeuralNetwork`),
- `count_parameters()` and a `main()` that prints the model and runs a random batch
  (`include_main`, default `true`),
- for models with one flat `float` input and one flat output, a CSV training loop
  (`include_training`, default `true`): classification with `CrossEntropyLoss` if the output
  has more than one unit, regression with `MSELoss` otherwise.
