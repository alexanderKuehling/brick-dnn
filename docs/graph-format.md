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

Every brick also says what it is and what it connects to:

| Field         | Meaning                                                                          |
| ------------- | -------------------------------------------------------------------------------- |
| `family`      | `neural` (PyTorch layer), `classical` (kernel or classical model, scikit-learn) or `shared` (`Input`, used by both) |
| `output_kind` | What flows out of the brick along its connections: `tensor` or `kernel`          |
| `input_slots` | What it takes, one entry per kind of connection: `{"name", "label", "kind", "min", "max"}` (`max` `null`: any number). `[]` for bricks without inputs |
| `min_inputs`/`max_inputs` | All incoming connections together (unchanged, for older clients)     |

For example `Linear` takes `[{"name": "input", "kind": "tensor", "min": 1, "max": 1}]`, a
Gaussian process `[{"name": "features", "kind": "tensor", "min": 1, "max": 1}, {"name":
"kernel", "kind": "kernel", "min": 0, "max": 1}]` and `KernelSum`
`[{"name": "kernels", "kind": "kernel", "min": 2, "max": null}]`.

## Edges

`{"source": "<node id>", "target": "<node id>"}`. Duplicates are ignored.

- Layers take exactly one input, except `Input` (none) and `Add`/`Concat` (two or more).
- `Concat` joins its inputs **in the order of the edges** in the document.
- Nodes without outgoing edges are the outputs of the network. With several outputs,
  `forward()` returns a tuple in node order.
- The graph must be acyclic and every layer must be reachable from an `Input`.

An edge carries what its source puts out (the source's `output_kind`): a **tensor** or, from
a kernel brick, a **kernel**. Edges have no type field of their own; the analysis reports
every node's `output_kind`, so a client can colour or check the connections.

## Classical models

Besides neural networks, a graph can describe a *classical* model (scikit-learn) for tabular
data. Its bricks are in two categories of the catalogue:

| Category         | Bricks |
| ---------------- | ------ |
| Kernels          | `RBF`, `Matern` (ν 0.5, 1.5, 2.5), `RationalQuadratic`, `ExpSineSquared` (periodic), `DotProduct`, `WhiteKernel`, `ConstantKernel`; `KernelSum` and `KernelProduct` combine two or more kernels |
| Classical models | `GaussianProcessRegressor`, `GaussianProcessClassifier`, `SVR`, `SVC`, `HistGradientBoostingRegressor`, `HistGradientBoostingClassifier` |

```json
{
  "version": 1,
  "nodes": [
    { "id": "x", "type": "Input", "params": { "shape": [8] } },
    { "id": "rbf", "type": "RBF", "params": { "length_scale": 1.0 } },
    { "id": "noise", "type": "WhiteKernel", "params": { "noise_level": 0.1 } },
    { "id": "kernel", "type": "KernelSum", "params": {} },
    { "id": "gp", "type": "GaussianProcessRegressor", "params": { "normalize_y": true } }
  ],
  "edges": [
    { "source": "x", "target": "gp" },
    { "source": "rbf", "target": "kernel" },
    { "source": "noise", "target": "kernel" },
    { "source": "kernel", "target": "gp" }
  ]
}
```

Rules (the analysis explains every violation at the brick concerned):

- The model takes a flat `float` feature vector **straight from the `Input`** (shape
  `[number of features]`). Classical models and neural layers can't be mixed yet: no layers
  between the Input and the model, none after it, and no neural layers elsewhere in the
  network.
- One classical model per network. It ends the network; its output is the prediction: one
  value for regression (`output_shape` `[1]`), one class for classification
  (`output_shape` `null`: the classes come from the data).
- Kernel bricks have no inputs. A kernel goes into `KernelSum`, `KernelProduct` or a
  Gaussian process (at most one kernel per Gaussian process), nowhere else; every kernel
  brick is connected exactly once. `SVR` and `SVC` choose their kernel with their own
  `kernel` parameter (`linear`, `poly`, `rbf`, `sigmoid`); gradient boosting takes no kernel.
- Gradient boosting (`HistGradientBoostingRegressor`, `HistGradientBoostingClassifier`)
  takes `learning_rate` (> 0, default 0.1), `max_iter` (1 … 10,000, 200), `max_leaf_nodes`
  (2 … 4,096, 31), `max_depth` (1 … 100, empty: no limit), `min_samples_leaf` (≥ 1, 20),
  `l2_regularization` (≥ 0, 0), `early_stopping` (`auto`, `on`, `off`), `random_state`
  (≥ 0, 0) and, for the classifier, `class_weight` (`none` or `balanced`).
- A Gaussian process without a kernel uses `ConstantKernel × RBF + WhiteKernel`
  (regression) or `ConstantKernel × RBF` (classification).
- Kernel hyperparameters are **initial values**: the Gaussian process optimises them on the
  training data. `hyperparameters` chooses how: `optimize` (within scikit-learn's bounds
  1e-05 … 1e+05), `custom bounds` (the `<name>_min`/`<name>_max` parameters) or `fixed`.

`KernelSum` adds its kernels in the order of the edges, `KernelProduct` multiplies them; a
sum inside a product gets parentheses.

## Analysis

`analyze(graph).to_dict()` (`POST /api/analyze/`) returns `valid`, `errors`, `warnings`,
`inputs`, `outputs`, `total_params` and per node `type`, `output_shape`, `dtype`,
`num_params` and the visible `params`. Further fields:

| Field (per node) | Meaning |
| ---------------- | ------- |
| `output_kind`    | `tensor` or `kernel` (the kind of its outgoing edges) |
| `kernel`         | Kernel bricks: the kernel built up to this brick as scikit-learn code, e.g. `ConstantKernel(constant_value=1.0) * RBF(length_scale=1.0)`; Gaussian processes: the kernel they start from (the default one if none is connected); otherwise `null` |
| `task`           | Classical models: `regression` or `classification`; otherwise `null` |

| Field (top level) | Meaning |
| ----------------- | ------- |
| `family`          | `classical` if the graph contains kernels or a classical model, else `neural` |
| `classical`       | The classical model: `{"node", "type", "estimator", "task", "kernel"}`, or `null` |

`num_params` counts what is fitted to the data: for kernel bricks the hyperparameters the
Gaussian process optimises (none when `fixed`), for a Gaussian process those of its default
kernel (`0` with a connected kernel, which counts its own), for `SVR`/`SVC` and gradient
boosting `null` (their support vectors or trees are only known after fitting, so
`total_params` is `null` too).

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
  has more than one unit, regression with `MSELoss` otherwise. `--csv data.csv` trains on a
  random 80 % of the rows (fixed by `--seed`, default 0, which also fixes the initial weights)
  and validates on the rest; the features are standardised with the mean and standard
  deviation of the training rows. `--save model.pt` (the default) stores a checkpoint
  `{"state_dict": ..., "preprocessing": {"columns", "target", "mean", "std", "task",
  "num_classes"}}`, and `--predict new.csv` loads it, prepares the rows of another CSV file
  the same way (columns by name if both files have a header line, otherwise the first
  columns) and prints one prediction per row.

For a classical model the script uses scikit-learn instead of PyTorch:

- `make_model()` returns `Pipeline([("scale", StandardScaler()), ("model", ...)])` with the
  kernel built from the bricks, e.g.
  `ConstantKernel(constant_value=1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)`.
  `SVC` is wrapped in `CalibratedClassifierCV(..., ensemble=False)` (Platt scaling with
  5-fold cross-validation), which gives it class probabilities. Gradient boosting is
  `HistGradientBoostingRegressor(...)` or `HistGradientBoostingClassifier(...)` from
  `sklearn.ensemble` with all its settings written out.
- Without arguments, `main()` prints the model and fits it to 40 random rows.
- `--csv data.csv` (with `include_training`) fits it to a random 80 % of the rows (`--seed`)
  and reports the validation RMSE and R² or the accuracy, for a Gaussian process the fitted
  kernel and log marginal likelihood, for an SVM the number of support vectors, for
  gradient boosting the boosting iterations (of at most `max_iter`). The scaler
  learns from the training rows only. The target may hold class labels as text.
- `--save model.joblib` (the default) stores `{"format": "brickdnn-sklearn-1", "model":
  <Pipeline>, "task", "columns", "target", "class_names"}` with joblib (a pickle: only load
  files you trust); `class_names` has one name per column of `predict_proba`. Models
  trained in Brick DNN are saved in the same format.
- `--predict new.csv` loads it and prints the class with its probability, the value, or for
  a Gaussian process the value and its standard deviation. It is also there without
  `include_training`, to predict with a downloaded model.
- `class_name` does not apply. The script needs `scikit-learn` (with NumPy and joblib).
