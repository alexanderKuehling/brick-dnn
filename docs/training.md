# Training

Brick DNN can train the network you built **on the machine that runs the backend**, with
PyTorch. You start and watch the training in the editor; nothing is computed in the browser.
The same engine is available on the command line as `brickdnn train`.

![Finished training of a small CNN on MNIST with loss and accuracy curves](images/training.png)

## Setup

Training needs PyTorch and torchvision in the backend environment. Everything else in Brick DNN
works without them.

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
   classes. The last layer should output 10 values."*
4. Adjust the settings (epochs, batch size, learning rate, optimizer, device) and press
   **Start training**. Optionally hold out a [test set](#validation-and-test-data) or let a
   [hyperparameter search](#hyperparameter-search) choose some of the settings.
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
| **Image folder** | Upload a `.zip` with one folder per class, e.g. `cats/001.jpg`, `dogs/001.jpg`. | Input `1, H, W` or `3, H, W`; one output per class |

Details:

- **Images** are converted to the number of channels of the Input layer (grayscale or RGB) and
  resized to its height and width, so a CNN built for 32×32 also trains on MNIST. Pixel values
  are normalised per channel.
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

### Settings

| Setting | Default | Notes |
| ------- | ------- | ----- |
| Epochs | 10 | Passes over the training data |
| Batch size | 64 | Lower it if you run out of memory |
| Learning rate | 0.001 | If the loss becomes NaN, lower it |
| Optimizer | Adam | Adam, AdamW or SGD (momentum 0.9) |
| Weight decay | 0 | L2 regularisation |
| Validation split | 0.2 | Not used for built-in datasets without a test set |
| Test split | 0 (none) | Share held out as test set; built-in datasets: their official test images |
| Device | Automatic | CUDA → Apple GPU (Apple Silicon) → CPU |
| Max. samples | all | Train on a subset for quick experiments; validation shrinks in proportion |
| Seed | random | Fix it for reproducible runs |

The loss is chosen from the task: cross-entropy for classification (the network should end
with raw scores, not Softmax), mean squared error for regression.

### Validation and test data

The data is split randomly (with the seed) into up to three parts:

- **Training data** is what the network learns from.
- **Validation data** is measured after every epoch (the curves). The hyperparameter search
  also uses it to compare trials, so after a search the validation result is slightly
  optimistic.
- **Test data** (optional, *Test split* > 0) is not touched until training is over. Then the
  final model is measured on it once. This is the number to report. Stopped runs are not tested.

| Data | Without test set | With test set |
| ---- | ---------------- | ------------- |
| CSV file, image folder, demo data | `val_split` for validation, the rest for training | `val_split` for validation, `test_split` for testing, the rest (at least 20 %) for training |
| MNIST, Fashion-MNIST, CIFAR-10 | the official test images for validation | the official test images for testing; `val_split` of the training images for validation |

Normalisation statistics (CSV features and targets, converted images) are computed from the
training part only. With *Max. samples*, validation and test data shrink in proportion.

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
- The trial with the **lowest validation loss after its last epoch** wins. Its weights are
  saved, its curves are shown, and it is the model that is measured on the test set.
- A trial whose loss becomes NaN or that runs out of memory (for example with a large batch
  size) is marked as failed; the search continues with the next one.
- Stopping a search keeps the best trial finished so far.

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

Prepare new inputs the same way as during training. The run view lists the preprocessing:
normalisation (mean and standard deviation), image size and class names.

## Command line

The `brickdnn` package trains without the web UI:

```bash
pip install "brickdnn[train] @ git+https://github.com/alexanderKuehling/brick-dnn#subdirectory=backend"

brickdnn train my_network.json --data mnist --epochs 5          # built-in dataset
brickdnn train --example tabular_regression --data houses.csv   # CSV file
brickdnn train my_cnn.json --data ./pets/ --epochs 20           # image folder (or a .zip)
brickdnn train --example lstm_sentiment --data demo             # demo data
brickdnn train my_cnn.json --data mnist --test-split 0.2        # test on the official test images
```

Options: `--epochs`, `--batch-size`, `--lr`, `--optimizer`, `--weight-decay`, `--val-split`,
`--test-split`, `--device`, `--max-samples`, `--seed`, `--save weights.pt`, `--data-root`
(download folder for built-in datasets, default `~/.cache/brickdnn/datasets`). Press
<kbd>Ctrl</kbd>+<kbd>C</kbd> to stop early; the weights are saved anyway.

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
  responsive; a crash in training cannot take it down.
- The model is created by executing the code that **Export .py** produces, so the trained
  weights always match the exported script.
- Run directories live in `BRICKDNN_DATA_DIR/training/runs/<id>/` (`job.json`, `state.json`,
  `weights.pt` and `output.log` with the full output, useful when a run fails). Uploaded datasets
  are in `training/datasets/`, downloaded built-in datasets in `training/builtin/`.
- Stopping creates a `STOP` file; the runner finishes the current batch, saves the weights and
  exits. A run whose process disappears (for example because the server was restarted) is
  marked as failed.

The REST endpoints are documented in [api.md](api.md#training-login-required).

## Configuration

| Variable | Default | Meaning |
| -------- | ------- | ------- |
| `BRICKDNN_TRAINING` | `true` | Set to `false` to disable training completely |
| `BRICKDNN_TRAINING_MAX_RUNS` | `1` | Trainings that may run at the same time (all users) |
| `BRICKDNN_MAX_UPLOAD_MB` | `500` | Maximum size of an uploaded dataset |
| `BRICKDNN_TRAINING_RATE` | `120/hour` | Rate limit for starting runs |
| `BRICKDNN_UPLOAD_RATE` | `60/hour` | Rate limit for uploads |
| `BRICKDNN_INSTALL_TORCH` | `false` | Docker build: install PyTorch (CPU) into the backend image |

On a public server, keep in mind that every registered user can use the server's CPU/GPU for
training; disable registration or training if that is not wanted.

## Troubleshooting

- **"PyTorch is not installed on the server"**: run `make install-training` (or rebuild the
  Docker image with `BRICKDNN_INSTALL_TORCH=true`) and restart the backend.
- **"Could not download MNIST"**: the server needs internet access once per dataset.
- **Loss becomes NaN**: lower the learning rate. In a search, only that trial fails.
- **Out of memory**: lower the batch size, the image size or the network size.
- **Something else failed**: the last lines of `output.log` in the run directory contain the
  full error.
