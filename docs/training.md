# Training

Brick DNN can train the network you built **on the machine that runs the backend**, with
PyTorch. You start and watch the training in the editor; nothing is computed in the browser.
The same engine is available on the command line as `brickdnn train`. Classical models
(Gaussian processes and support vector machines) are fitted with scikit-learn instead, see
[Classical models](#classical-models).

![Finished training of a small CNN on MNIST with loss and accuracy curves](images/training.png)

## Setup

Training needs PyTorch and torchvision in the backend environment. Everything else in Brick DNN
works without them. Classical models only need scikit-learn, which is part of the backend's
requirements (and of the default Docker image), so they train even without PyTorch.

```bash
# local development setup (see the README)
make install-training            # = backend/.venv/bin/pip install torch torchvision

# Docker: PyTorch (CPU) is not in the default image because it adds about 1 GB
BRICKDNN_INSTALL_TORCH=true docker compose up -d --build     # or: make docker-training
```

Restart the backend afterwards. The **Train** button in the editor then shows which devices
are available. Training uses an NVIDIA GPU (CUDA) or the GPU of an Apple Silicon Mac (MPS)
automatically when PyTorch supports one, otherwise the CPU. On Intel Macs the CPU is the
default: PyTorch's GPU support there is no faster and has been seen computing wrong results.
Before training on an Apple GPU, Brick DNN compares one batch with the CPU and switches to the
CPU if the results differ. Docker containers only get the CPU build, so for GPU training run
the backend outside Docker.

Training requires an account (like saving networks), because runs and uploaded datasets are
stored per user.

## Training in the editor

1. Build a network, or load one of the examples. It must be free of errors.
2. Press **Train** in the toolbar.
3. Choose the data (see below). Brick DNN immediately checks whether the network fits the data
   and explains what to change, for example *"The network outputs 5 values, but MNIST has 10
   classes. The last layer should output 10 values."* The Input layer follows the data: see
   [Adapting the network to the data](#adapting-the-network-to-the-data).
4. Adjust the settings (epochs, batch size, learning rate, optimizer, device) and press
   **Start training**. Optionally hold out a [test set](#validation-and-test-data), let
   [early stopping](#early-stopping-and-the-kept-weights) end the training when the network
   stops improving, or let a [hyperparameter search](#hyperparameter-search) choose some of
   the settings.
5. Watch loss and accuracy per epoch. You can close the panel; the training continues and the
   button in the toolbar shows its progress.
6. When the run is finished (or stopped), download the **weights** (`.pt`) and the matching
   **script** (`.py`).

All runs are kept in the **History** tab together with the exact network that was trained.
Only one training runs at a time by default (`BRICKDNN_TRAINING_MAX_RUNS`).

### Data

| Source | What it is | Network requirements |
| ------ | ---------- | -------------------- |
| **Demo data** | Random inputs plus a hidden rule that the network can learn. Shows whether a network trains at all. Works with every network, including text models and several inputs/outputs. | none |
| **MNIST**, **Fashion-MNIST** | 70,000 grayscale 28×28 images of digits / clothing, 10 classes. Downloaded on first use (12 / 30 MB). | Input `1, 28, 28`, `28, 28` or `784`; 10 outputs |
| **CIFAR-10** | 60,000 colour 32×32 photos, 10 classes. Downloaded on first use (170 MB). | Input `3, 32, 32` or `3072`; 10 outputs |
| **CSV file** | Upload a `.csv`: every column but the last is a numeric feature, the last column is the target. | Input = number of feature columns; 1 output (regression) or one output per class |
| **Image folder** | Upload a `.zip` with one folder per class, e.g. `cats/001.jpg`, `dogs/001.jpg`. | Input `1, H, W` or `3, H, W` (or flat: `3·S·S` or `S·S` values for S×S images); one output per class |

Details:

- **Images** are converted to the number of channels of the Input layer (grayscale or RGB) and
  resized to its height and width, so a CNN built for 32×32 also trains on MNIST. Pixel values
  are normalised per channel. A flat Input (e.g. followed by a Linear layer) takes the images
  flattened: `784` = 28×28 grayscale for MNIST, `3072` = 3×32×32 for CIFAR-10; for an image
  folder, `12288` = 3×64×64 RGB or `784` = 28×28 grayscale (square images of at least 8×8).
- How much data is held out for validation and testing is explained
  [below](#validation-and-test-data).
- **CSV files** may use `,`, `;` or tabs as separators and decimal commas (`1,5`). A header
  line is detected automatically. Features are standardised (mean 0, standard deviation 1).
  - *Classification*: the target is a class index `0 … K-1` or a text label (`setosa`,
    `virginica` …; mapped to indices in alphabetical order). The network needs K outputs.
  - *Regression*: the target is a number and the network has a single output. The validation
    error is also reported as RMSE in the units of the target.
- Text models (Input data type `long`) and networks with several inputs or outputs can only be
  trained on demo data so far.
- Uploads are limited to `BRICKDNN_MAX_UPLOAD_MB` (default 500 MB).

### Adapting the network to the data

The Input layer follows the dataset chosen for training. When the network does not fit, the
check also says how to fix it, and Brick DNN applies the fix for you:

- **Automatically**, once each time you choose a dataset (in the training panel, after an
  upload, or with **Train with this data** in the [Data tab](data-preparation.md)), if only the
  Input layer's shape (and data type) and the number of outputs of the last layer have to
  change. A notification says what changed, e.g. *"Input set to 4 features, last layer to 3
  outputs from “iris”"*, with **Undo**. On an empty canvas, choosing a dataset starts a small
  network for it (Input → Linear → ReLU → Linear; images are flattened first).
- **With one click** otherwise: **Adapt network to “…”** below the check result lists the
  changes, for example also when the network fits but has more outputs than there are classes.
  The Inspector of an Input layer shows *Matches “…”* or offers **Set from dataset**.

Only two layers are ever changed: the Input layer (`shape`, data type `float`) and the last
Linear layer (also behind activations or dropout; `out_features` = one per class for
classification, 1 for a regression). A shape that works is kept (a CNN for 1×28×28 also trains
on CIFAR-10); otherwise the Input gets the shape of the data, flat when it is flat now (e.g.
`784` for MNIST before a Linear layer). Nothing is suggested when other layers would have to
change too, e.g. a CNN on a CSV file or a Linear layer with a fixed number of input features;
the check then explains what to do. Opening a network does not change it, even if a dataset is
remembered from before, and people who can only view a network never get any of this. In a
shared network, the changes reach the others like any edit. For a
[classical model](#classical-models) only the Input's shape is ever changed: its output
follows the data.

### Settings

| Setting | Default | Notes |
| ------- | ------- | ----- |
| Epochs | 10 | Passes over the training data |
| Batch size | 64 | Lower it if you run out of memory |
| Learning rate | 0.001 | If the loss becomes NaN, lower it |
| Optimizer | Adam | Adam, AdamW or SGD (momentum 0.9) |
| Weight decay | 0 | L2 regularisation |
| Validation split | 0.2 | Share of the data used for validation (built-in datasets: of the training images) |
| Test split | 0 (none) | Share held out as test set; not used for built-in datasets, which always test on their official test images |
| Device | Automatic | CUDA → Apple GPU (Apple Silicon) → CPU |
| Max. samples | all | Train on a subset for quick experiments; validation shrinks in proportion |
| Seed | random | Fix it for reproducible runs |
| Early stopping (patience) | off | Stop when the validation loss has not improved for this many epochs |
| Keep the best weights | on | Test and save the weights of the epoch with the lowest validation loss, not those of the last epoch |

The loss is chosen from the task: cross-entropy for classification (the network should end
with raw scores, not Softmax), mean squared error for regression.

### Validation and test data

The data is split randomly (with the seed) into up to three parts:

- **Training data** is what the network learns from.
- **Validation data** is measured after every epoch (the curves). It also chooses the kept
  weights (the best epoch) and, in a search, the best trial, so the validation result is
  slightly optimistic.
- **Test data** (optional, *Test split* > 0; built-in datasets: always their official test
  images) is not touched until training is over. Then the kept weights are measured on it
  once. This is the number to report. Stopped runs are not tested.

| Data | Validation | Test |
| ---- | ---------- | ---- |
| CSV file, image folder, demo data | `val_split` of the data | `test_split` of the data (none if 0); the rest, at least 20 %, is for training |
| MNIST, Fashion-MNIST, CIFAR-10 | `val_split` of the training images | always the official test images (10,000); `test_split` is not used |
| Datasets from the *Data* tab | the split made there | the split made there; `val_split` and `test_split` are not used |

The official test images of the built-in datasets are never used to choose an epoch or a trial,
so their test result is not optimistic. With 20 % validation, MNIST trains on 48,000 images,
validates on 12,000 and tests on the 10,000 official test images.

Normalisation statistics (CSV features and targets, converted images) are computed from the
training part only. With *Max. samples*, validation and test data shrink in proportion.

### Early stopping and the kept weights

Training longer does not always help: once the network starts to memorise the training data
(overfitting), the validation loss goes up again. Brick DNN therefore remembers the weights of
the epoch with the **lowest validation loss** and keeps those at the end: they are measured on
the test set and saved as `weights.pt`, and the run view says which epoch they come from
(`weights_epoch` in the run state). Switch *Keep the best weights* off (`restore_best: false`,
`--no-restore-best`) to keep the weights of the last epoch instead.

With **early stopping** (*Patience* N, `patience`, `--patience N`), the training ends when the
validation loss has not improved for N epochs in a row, for example *"Stopped early after epoch
12: no improvement for 5 epochs; kept the weights of epoch 7."* Such a run counts as completed
and is tested like any other. Early stopping is off by default, so every epoch is trained.

A run you stop yourself also keeps the weights of the best epoch so far (or, without *Keep the
best weights*, those of the moment it stopped).

### Hyperparameter search

Instead of guessing the learning rate or batch size, let Brick DNN try several values. A search
trains the network several times (*trials*), each time with other values for the chosen
settings and otherwise the normal settings (every trial runs the full number of epochs).

- **Random search** draws the given number of trials. Learning rate and weight decay are given
  as a range (for example 0.0001 to 0.01) and drawn on a logarithmic scale; batch sizes and
  optimizers as lists of values to choose from. If the lists only allow fewer combinations
  than trials, each combination is tried once.
- **Grid search** tries every combination of the listed values, at most 100 (in the editor,
  the learning rates are then picked from a list as well).
- All trials use the same training and validation data and start from the same initial
  weights with the same order of batches, so only the settings make the difference.
- Every trial keeps the weights of its best epoch (see
  [above](#early-stopping-and-the-kept-weights)) and may stop early; the trial whose kept
  weights have the **lowest validation loss** wins. Its weights are saved, its curves are
  shown, and it is the model that is measured on the test set.
- A trial whose loss becomes NaN or that runs out of memory (for example with a large batch
  size) is marked as failed; the search continues with the next one.
- Stopping a search keeps the best trial finished so far (during the first trial: its best
  epoch so far).

In the editor, switch on **Hyperparameter search** in the settings and tick the settings to
tune; the normal fields of those settings are then filled in by the search. The run view lists
every trial with its settings and result, the best one highlighted. A search
takes as long as all its trials together: start with few epochs or *Max. samples* to find the
region of good values, then train once with them.

### Using the trained weights

The weights belong to the exported script of the same network (class `NeuralNetwork`):

```python
import torch
from small_cnn import NeuralNetwork      # the downloaded script

model = NeuralNetwork()
model.load_state_dict(torch.load("small_cnn.pt"))
model.eval()
```

Prepare new inputs the same way as during training. The values that this needs are saved with
the weights: in the run view, download **Preprocessing (.json)** next to the weights
(`brickdnn train` writes `<name>.preprocessing.json` next to `<name>.pt`). It holds the
normalisation (mean and standard deviation per image channel or CSV column, computed from the
training part only), the image size and layout or the CSV feature columns and target, and the
class names. *How to use the weights* in the run view shows the code for one input.

The server can also predict with the weights, without downloading them:
`POST /api/training/runs/<id>/predict/` (see [api.md](api.md#training-login-required)) or the
MCP tool `predict`. It prepares the inputs like the training data and runs the network in a
separate process, like a training. AI assistants get the weights with `get_weights_download`:
a download link valid for 15 minutes, the model code, the preprocessing and a snippet like the
one above (see [mcp.md](mcp.md#downloading-the-weights)).

## Classical models

Besides neural networks, Brick DNN builds and trains classical models for tabular data
(the bricks are described in [graph-format.md](graph-format.md#classical-models)):

| Model | Task | Brick |
| ----- | ---- | ----- |
| Gaussian process regression | predicts a number with its uncertainty | `GaussianProcessRegressor` + optional kernel bricks |
| Gaussian process classification | classes with probabilities | `GaussianProcessClassifier` + optional kernel bricks |
| Support vector regression | predicts a number | `SVR` |
| Support vector classification | classes with probabilities (Platt scaling) | `SVC` |

Connect the Input (shape: the number of feature columns) straight to the model; a Gaussian
process may also take a kernel built from kernel bricks (`RBF`, `Matern`, `RationalQuadratic`,
`ExpSineSquared`, `DotProduct`, `WhiteKernel`, `ConstantKernel`, combined with `KernelSum`
and `KernelProduct`). Their hyperparameters are starting values that the fit optimises. The
examples *Gaussian process regression (RBF + noise)*, *SVM classifier* and *Periodic Gaussian
process* are good starting points.

In the editor, the bricks are in the *Classical · scikit-learn* part of the palette. A Gaussian
process has two inputs, *features* (from the Input) and *kernel*; kernel connections are
dashed, and the editor refuses a kernel into a data input and data into a kernel input.
Selecting a Gaussian process or a kernel sum and clicking a kernel in the palette connects
the kernel to it. The inspector shows the kernel of the selected brick as scikit-learn code
(for a Gaussian process without kernel bricks, its default kernel), and the toolbar counts
the hyperparameters the fit optimises instead of trainable parameters.

They are trained with scikit-learn by the same runner process, with the same run view,
history and API, but **fitted once instead of epoch by epoch**:

- **Data**: CSV files, datasets from the *Data* tab (with their split) and demo data
  (random rows with a hidden rule: 500 for Gaussian processes, 1,000 for SVMs; 3 classes
  for classifiers). Images and the
  built-in datasets give a compatibility error. Regression models need a numeric target;
  classifiers take text labels, whole numbers (each value is a class) or the classes of a
  prepared dataset. The check explains mismatches, e.g. *"The target column 'species'
  contains text labels (3 classes), so this is a classification task: use
  GaussianProcessClassifier or SVC instead of Support vector regression."*
- **Split and scaling**: the same rules as for neural networks (validation split, test split,
  the split of a prepared dataset, *Max. samples*, *Seed*); the random split uses NumPy's
  generator, so the rows differ from a neural run with the same seed. The model is a
  `Pipeline` of a `StandardScaler` and the estimator, and the scaler learns from the
  training rows only.
- **Settings**: only validation split, test split, max. samples and seed apply. Epochs,
  batch size, learning rate, optimizer, weight decay, device, early stopping and the kept
  weights do not, and the hyperparameter search is not available (a Gaussian process
  optimises its kernel hyperparameters itself).
- **Size limits**: fitting a Gaussian process takes O(n³) time and O(n²) memory, an SVM
  between O(n²) and O(n³). A Gaussian process classifier is much slower than a regression
  (its Laplace approximation is iterative, and it fits one Gaussian process per class when
  there are more than two). Gaussian process regression is fitted on at most 5,000 training
  rows (`BRICKDNN_GP_MAX_ROWS`), classification on at most 2,000 (`BRICKDNN_GPC_MAX_ROWS`),
  SVMs on at most 50,000 (`BRICKDNN_SVM_MAX_ROWS`). Larger training
  sets are subsampled like with *Max. samples* (validation and test shrink in proportion),
  which the check warns about and the log says; a larger *Max. samples* is refused.
- **In the training panel** only the settings above are shown; *Max. samples* says how many rows
  the model is fitted on at most on this server and refuses more. Classical models train even
  when PyTorch is not installed. After the fit, the run shows the fit time, the rows it was
  fitted on, the fitted and the starting kernel with the log marginal likelihood (Gaussian
  processes) or the number of support vectors (SVMs), and the accuracy and loss or the RMSE
  and R² on the training, validation and test rows, instead of loss curves.
- **What is reported**: the status says what happens ("Fitting the Gaussian process on 4,000
  rows …"); there is no progress during the fit. Afterwards the run has one history entry
  (epoch 1): the training and validation loss (cross-entropy of the predicted probabilities,
  or the mean squared error of the standardised target, like for neural networks), the
  accuracy or the RMSE and R², the test result if there is a test set, the fit time, the
  fitted kernel and log marginal likelihood of a Gaussian process (one kernel per class for
  a classifier with more than two classes) or the number of support vectors of an SVM, and
  warnings of the fit in the log (e.g. a hyperparameter that ended at its bound).
- **Stopping**: scikit-learn cannot be interrupted, so stopping during the fit ends the
  training process at once and nothing is saved.
- **The model**: the fitted pipeline is saved with joblib as `model.joblib` (download:
  `<run name>.joblib`), with `model.preprocessing.json` next to it. It is a dictionary
  `{"format": "brickdnn-sklearn-1", "model": <Pipeline>, "task", "columns", "target",
  "class_names"}`, the same as the exported script's `--save`, so the script predicts with
  it: `python my_gp.py --predict new.csv --save my-gp.joblib`. In Python:

  ```python
  import joblib

  bundle = joblib.load("my-gp.joblib")  # a pickle: only load files you trust
  mean, std = bundle["model"].predict([[3.0, 120.0]], return_std=True)
  ```

- **Predictions** on the server (REST and MCP) work like for neural networks; classifiers
  return their own probabilities, a Gaussian process regressor also its standard deviation.

## Command line

The `brickdnn` package trains without the web UI:

```bash
pip install "brickdnn[train] @ git+https://github.com/alexanderKuehling/brick-dnn#subdirectory=backend"

brickdnn train my_network.json --data mnist --epochs 5          # built-in dataset
brickdnn train --example tabular_regression --data houses.csv   # CSV file
brickdnn train my_cnn.json --data ./pets/ --epochs 20           # image folder (or a .zip)
brickdnn train --example lstm_sentiment --data demo             # demo data
brickdnn train my_cnn.json --data houses.csv --test-split 0.2   # hold out 20 % as a test set
brickdnn train --example gp_regression --data houses.csv        # a classical model: houses.joblib
```

Classical models only need scikit-learn (`pip install "brickdnn[classic]"`; `[train]`
includes it) and save the fitted model as `<name>.joblib`.

Options: `--epochs`, `--batch-size`, `--lr`, `--optimizer`, `--weight-decay`, `--val-split`,
`--test-split`, `--device`, `--max-samples`, `--seed`, `--patience N` (early stopping),
`--no-restore-best` (keep the weights of the last epoch), `--save weights.pt`, `--data-root`
(download folder for built-in datasets, default `~/.cache/brickdnn/datasets`). Press
<kbd>Ctrl</kbd>+<kbd>C</kbd> to stop; the weights of the best epoch so far are saved anyway.

Hyperparameter search: any of the following options starts one. `VALUES` is a comma-separated
list or, for the learning rate and weight decay, a range `MIN:MAX`.

```bash
brickdnn train my_cnn.json --data mnist --tune                   # learning rate 1e-4:1e-2, batch size 32,64,128
brickdnn train my_cnn.json --data mnist --trials 20 --tune-lr 1e-5:1e-1 --tune-optimizer adam,sgd
brickdnn train net.json --data houses.csv --search grid --tune-lr 0.01,0.001 --tune-batch-size 32,64
```

| Option | Meaning |
| ------ | ------- |
| `--tune` | Search with the default space (learning rate 1e-4:1e-2, batch size 32,64,128) |
| `--search random\|grid` | Random search (default) or every combination of the lists |
| `--trials N` | Trials of a random search (default 10, at most 100) |
| `--tune-lr VALUES` | Learning rates to try |
| `--tune-batch-size VALUES` | Batch sizes to try (list) |
| `--tune-optimizer VALUES` | Optimizers to try, e.g. `adam,adamw,sgd` |
| `--tune-weight-decay VALUES` | Weight decays to try |

The command prints every trial, a table of all trials (best first) and the test result, and
saves the weights of the best trial.

## How it works

```
editor ──REST (polls every second)──▶ Django API ──spawns──▶ python -m brickdnn.training.runner <run dir>
                                         │                          │
                                         │  job.json  ─────────────▶│ builds the model from the exported code,
                                         │  STOP      ─────────────▶│ prepares the data, trains
                                         ◀───────────── state.json ─┤ progress, written atomically
                                         ◀───────────── weights.pt ─┘ (a search: the best trial)
```

- Every run is a **separate process**. The web server never imports PyTorch and stays
  responsive; a crash in training cannot take it down. The process has a lower CPU priority
  than the web server, and if memory runs out, the system kills the training first (Linux).
- The model is created by executing the code that **Export .py** produces, so the trained
  weights always match the exported script.
- Run directories live in `BRICKDNN_DATA_DIR/training/runs/<id>/` (`job.json`, `state.json`,
  `weights.pt`, `weights.preprocessing.json` and `output.log` with the full output, useful when
  a run fails). Uploaded datasets
  are in `training/datasets/`, downloaded built-in datasets in `training/builtin/`.
- Stopping creates a `STOP` file; the runner finishes the current batch, saves the weights
  (of the best epoch so far) and exits. A run whose process disappears (for example because the server was restarted) is
  marked as failed.
- Predictions start `python -m brickdnn.training.predict <run dir>` per request: it builds the
  network from `job.json`, loads `weights.pt`, reads the prepared inputs from stdin and prints
  the outputs. It runs on the CPU with a lower priority and is stopped after 30 seconds; the
  web server waits for it.

The REST endpoints are documented in [api.md](api.md#training-login-required).

## Configuration

| Variable | Default | Meaning |
| -------- | ------- | ------- |
| `BRICKDNN_TRAINING` | `true` | Set to `false` to disable training completely |
| `BRICKDNN_TRAINING_MAX_RUNS` | `1` | Trainings that may run at the same time (all users) |
| `BRICKDNN_MAX_UPLOAD_MB` | `500` | Maximum size of an uploaded dataset |
| `BRICKDNN_TRAINING_RATE` | `120/hour` | Rate limit for starting runs |
| `BRICKDNN_UPLOAD_RATE` | `60/hour` | Rate limit for uploads |
| `BRICKDNN_MAX_PREDICTIONS` | `1` | Predictions running at the same time (all users) |
| `BRICKDNN_PREDICT_RATE` | `120/hour` | Rate limit for predictions |
| `BRICKDNN_MCP_UPLOAD_MB` | `5` | Largest CSV file AI assistants can upload over MCP |
| `BRICKDNN_INSTALL_TORCH` | `false` | Docker build: install PyTorch (CPU) into the backend image |
| `BRICKDNN_GP_MAX_ROWS` | `5000` | Training rows a Gaussian process regression is fitted on at most (more are subsampled) |
| `BRICKDNN_GPC_MAX_ROWS` | `2000` | Training rows a Gaussian process classifier is fitted on at most (more are subsampled) |
| `BRICKDNN_SVM_MAX_ROWS` | `50000` | Training rows an SVM is fitted on at most (more are subsampled) |

On a public server, keep in mind that every registered user can use the server's CPU/GPU for
training and predictions; disable registration or training if that is not wanted.

## Troubleshooting

- **"PyTorch is not installed on the server"**: run `make install-training` (or rebuild the
  Docker image with `BRICKDNN_INSTALL_TORCH=true`) and restart the backend.
- **"Could not download MNIST"**: the server needs internet access once per dataset.
- **Loss becomes NaN**: lower the learning rate. In a search, only that trial fails.
- **"scikit-learn is not installed on the server"**: run
  `pip install -r requirements.txt` (or `pip install scikit-learn`) in the backend
  environment and restart it.
- **A Gaussian process takes very long or runs out of memory**: set *Max. samples*, or lower
  `BRICKDNN_GP_MAX_ROWS` / `BRICKDNN_GPC_MAX_ROWS`. A fit that fails with a matrix that is not positive definite needs
  a larger `alpha` or a `WhiteKernel` in its kernel.
- **Out of memory**: lower the batch size, the image size or the network size. On a server
  with limits ([deploy/server](../deploy/server/README.md#cpu-and-memory-the-site-comes-first))
  such a run often only says "The training process stopped unexpectedly".
- **Something else failed**: the last lines of `output.log` in the run directory contain the
  full error.
