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
| `output_kind` | What flows out of the brick along its connections: `tensor`, `kernel` or `estimator` (classical models and ensembles) |
| `input_slots` | What it takes, one entry per kind of connection: `{"name", "label", "kind", "min", "max"}` (`max` `null`: any number). `[]` for bricks without inputs |
| `min_inputs`/`max_inputs` | All incoming connections together (unchanged, for older clients)     |

For example `Linear` takes `[{"name": "input", "kind": "tensor", "min": 1, "max": 1}]`, a
Gaussian process `[{"name": "features", "kind": "tensor", "min": 1, "max": 1}, {"name":
"kernel", "kind": "kernel", "min": 0, "max": 1}]`, `KernelSum`
`[{"name": "kernels", "kind": "kernel", "min": 2, "max": null}]` and an ensemble
`[{"name": "models", "kind": "estimator", "min": 2, "max": null}]`.

## Edges

`{"source": "<node id>", "target": "<node id>"}`. Duplicates are ignored.

- Layers take exactly one input, except `Input` (none) and `Add`/`Concat` (two or more).
- `Concat` joins its inputs **in the order of the edges** in the document.
- Nodes without outgoing edges are the outputs of the network. With several outputs,
  `forward()` returns a tuple in node order.
- The graph must be acyclic and every layer must be reachable from an `Input`.

An edge carries what its source puts out (the source's `output_kind`): a **tensor**, from a
kernel brick a **kernel**, or from a classical model the model itself, an **estimator**, for
an ensemble. Edges have no type field of their own; the analysis reports every node's
`output_kind`, so a client can colour or check the connections.

## Classical models

Besides neural networks, a graph can describe a *classical* model (scikit-learn) for tabular
data. Its bricks are in three categories of the catalogue:

| Category         | Bricks |
| ---------------- | ------ |
| Kernels          | `RBF`, `Matern` (ν 0.5, 1.5, 2.5), `RationalQuadratic`, `ExpSineSquared` (periodic), `DotProduct`, `WhiteKernel`, `ConstantKernel`; `KernelSum` and `KernelProduct` combine two or more kernels |
| Classical models | `GaussianProcessRegressor`, `GaussianProcessClassifier`, `SVR`, `SVC`, `HistGradientBoostingRegressor`, `HistGradientBoostingClassifier` |
| Ensembles        | `VotingClassifier`, `VotingRegressor`, `StackingClassifier`, `StackingRegressor` combine two or more classical models |

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
  `[number of features]`), or from neural layers in between, which learn the features (a
  [hybrid network](#hybrid-networks)). Nothing follows the model.
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

### Ensembles

An ensemble combines two or more classical models of the same task. Every model takes the
features from the `Input` as usual and is connected to the ensemble; that edge carries the
model itself (an `estimator`), not its prediction. The ensemble has no other inputs and ends
the network:

```json
{
  "version": 1,
  "nodes": [
    { "id": "x", "type": "Input", "params": { "shape": [4] } },
    { "id": "boosting", "type": "HistGradientBoostingClassifier", "params": {} },
    { "id": "svc", "type": "SVC", "params": {} },
    { "id": "gp", "type": "GaussianProcessClassifier", "params": {} },
    { "id": "vote", "type": "VotingClassifier", "params": { "voting": "soft" } }
  ],
  "edges": [
    { "source": "x", "target": "boosting" },
    { "source": "x", "target": "svc" },
    { "source": "x", "target": "gp" },
    { "source": "boosting", "target": "vote" },
    { "source": "svc", "target": "vote" },
    { "source": "gp", "target": "vote" }
  ]
}
```

| Brick | Parameters | scikit-learn |
| ----- | ---------- | ------------ |
| `VotingClassifier` | `voting` (`soft`: average the class probabilities, the default; `hard`: the class most models predict), `weights` | `VotingClassifier` |
| `VotingRegressor` | `weights` | `VotingRegressor` |
| `StackingClassifier` | `cv` (2 … 20, default 5), `passthrough` (default `false`) | `StackingClassifier` with `final_estimator=LogisticRegression()` |
| `StackingRegressor` | `cv`, `passthrough` | `StackingRegressor` with `final_estimator=RidgeCV()` |

`weights` is empty (every model counts the same) or one whole number ≥ 0 per model **in the
order of the edges into the ensemble**; at least one must be above 0. Stacking fits a
logistic regression (classifier) or a ridge regression (regressor) on the predictions the
models make for rows they were not fitted on (`cv` folds); with `passthrough` it also gets
the features. Rules:

- At least two models, all classifiers or all regression models (the ensemble's task), each
  fed by the same `Input`. Kernel bricks still go into the Gaussian processes.
- Every classical model of the network is connected to the ensemble, and only to it; one
  ensemble per network; ensembles can't be nested and nothing follows them.
- Soft voting needs class probabilities: every classifier brick predicts them (`SVC` through
  Platt scaling), so this only refuses models that might not in the future. Hard voting
  has none: training and predictions use the share of the (weighted) votes as the
  probability of each class.
- The models' names in the ensemble (and in the generated code) are their brick names or
  ids as lower-case identifiers, made unique: `"Boosting 1"` becomes `boosting_1`.

A model connected anywhere else than to an ensemble ends the network as before; the
analysis says that only an ensemble can follow it.

## Hybrid networks

Neural layers can turn the input into the features of a classical model: the layers learn
them, the model (or an ensemble) is fitted on them. For example an MLP before gradient
boosting for a table, or a small CNN before an SVM for images:

```json
{
  "version": 1,
  "nodes": [
    { "id": "in", "type": "Input", "params": { "shape": [8] } },
    { "id": "fc1", "type": "Linear", "params": { "out_features": 64 } },
    { "id": "relu", "type": "Activation", "params": { "function": "relu" } },
    { "id": "fc2", "type": "Linear", "params": { "out_features": 32 } },
    { "id": "boosting", "type": "HistGradientBoostingClassifier", "params": {} }
  ],
  "edges": [
    { "source": "in", "target": "fc1" },
    { "source": "fc1", "target": "relu" },
    { "source": "relu", "target": "fc2" },
    { "source": "fc2", "target": "boosting" }
  ]
}
```

The connection from the last neural layer into the model is the **boundary**: it carries the
features (a tensor, drawn teal and labelled *features* on the canvas). Rules (the analysis
explains every violation at the brick concerned):

- **Exactly one boundary.** The model takes a flat feature vector, shape `[n]`: a layer whose
  output has more dimensions (a convolution, say) needs `Flatten` or a `Linear` layer before
  the model (*"Support vector classification needs a flat feature vector, but Conv2d outputs
  (32, 8, 8). Add Flatten (or a Linear layer) before …"*). The models of an ensemble all take
  the same features (the same last layer).
- **Every neural layer leads to the boundary.** A branch that ends anywhere else is an error
  at its layer (*"Linear does not lead to Gradient boosting classification …"*); so are
  neural layers after a classical model (*"Nothing can follow …"*). One Input layer.
- Kernel bricks and estimator connections keep their rules.

The analysis reports `family: "hybrid"` and `hybrid: {"boundary", "features",
"neural_params"}` (the last neural layer, the length of its feature vector and the trainable
parameters of the neural layers); `classical` describes the model as for a classical
network. `total_params` stays `null` when the model's size is only known after the fit.
How such a network is trained (two phases) is described in
[training.md](training.md#hybrid-networks).

## Analysis

`analyze(graph).to_dict()` (`POST /api/analyze/`) returns `valid`, `errors`, `warnings`,
`inputs`, `outputs`, `total_params` and per node `type`, `output_shape`, `dtype`,
`num_params` and the visible `params`. Further fields:

| Field (per node) | Meaning |
| ---------------- | ------- |
| `output_kind`    | `tensor`, `kernel` or `estimator` (the kind of its outgoing edges) |
| `kernel`         | Kernel bricks: the kernel built up to this brick as scikit-learn code, e.g. `ConstantKernel(constant_value=1.0) * RBF(length_scale=1.0)`; Gaussian processes: the kernel they start from (the default one if none is connected); otherwise `null` |
| `task`           | Classical models and ensembles: `regression` or `classification`; otherwise `null` |
| `members`        | Ensembles only: their models in the order of the edges (and of `weights`), `[{"node", "name", "type", "estimator", "label"}]` |

| Field (top level) | Meaning |
| ----------------- | ------- |
| `family`          | `hybrid` if neural layers learn the features of a classical model, `classical` if the graph contains kernels or a classical model (and no neural layers), else `neural` |
| `classical`       | The classical model (the ensemble, if there is one): `{"node", "type", "estimator", "task", "kernel", "members"}` (`members` as above, `null` for a single model), or `null` |
| `hybrid`          | Hybrid networks: `{"boundary", "features", "neural_params"}` (see [Hybrid networks](#hybrid-networks)), else `null` |

`num_params` counts what is fitted to the data: for kernel bricks the hyperparameters the
Gaussian process optimises (none when `fixed`), for a Gaussian process those of its default
kernel (`0` with a connected kernel, which counts its own), for `SVR`/`SVC` and gradient
boosting `null` (their support vectors or trees are only known after fitting, so
`total_params` is `null` too), as for ensembles.

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
  `sklearn.ensemble` with all its settings written out. An ensemble gets a variable per
  model, named like the model in the ensemble (a Gaussian process's kernel `<name>_kernel`,
  an SVC's SVM `<name>_svm`), then e.g. `VotingClassifier(estimators=[("boosting",
  boosting), ("svc", svc), ("gp", gp)], voting="soft")`.
- Without arguments, `main()` prints the model and fits it to 40 random rows.
- `--csv data.csv` (with `include_training`) fits it to a random 80 % of the rows (`--seed`)
  and reports the validation RMSE and R² or the accuracy, for a Gaussian process the fitted
  kernel and log marginal likelihood, for an SVM the number of support vectors, for
  gradient boosting the boosting iterations (of at most `max_iter`), for an ensemble its
  models (and the fitted final estimator of stacking). The scaler
  learns from the training rows only. The target may hold class labels as text.
- `--save model.joblib` (the default) stores `{"format": "brickdnn-sklearn-1", "model":
  <Pipeline>, "task", "columns", "target", "class_names"}` with joblib (a pickle: only load
  files you trust); `class_names` has one name per column of `predict_proba`. Models
  trained in Brick DNN are saved in the same format.
- `--predict new.csv` loads it and prints the class with its probability (hard voting: the
  class only), the value, or for a Gaussian process the value and its standard deviation. It is also there without
  `include_training`, to predict with a downloaded model.
- `class_name` does not apply. The script needs `scikit-learn` (with NumPy and joblib).

For a hybrid network the script has both parts and needs PyTorch and scikit-learn:

- `FeatureExtractor(nn.Module)` (or `class_name`): the neural layers up to the boundary, and
  `make_model()`: the classical pipeline on their features;
- `--csv data.csv` splits the rows 80/20 (`--seed`), standardises the input columns with the
  training rows, trains the neural layers with a temporary linear head on the target
  (`learn_features`, `--epochs`, `--batch-size`, `--lr`; `--frozen` keeps the random
  weights), computes the features of all rows (`extract`), fits the model on those of the
  training rows and reports the validation result;
- it saves `--weights features.pt` (`{"state_dict", "preprocessing": {"columns", "target",
  "mean", "std"}}`) and `--save model.joblib` (the classical format with `"columns":
  ["feature_1", …]` and `"hybrid": {"features", "input_columns"}`); `--predict new.csv`
  reads both. A hybrid network trained in Brick DNN saves its files in the same formats.
- Networks on images (a non-flat Input) get the classes and a demo with random inputs.
