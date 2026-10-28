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
   **Start training**.
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
- Built-in datasets use their official test split for validation. For CSV files, image folders
  and demo data, a share of the data (default 20 %) is held out.
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
| Validation split | 0.2 | Not used for built-in datasets |
| Device | Automatic | CUDA → Apple GPU (Apple Silicon) → CPU |
| Max. samples | all | Train on a subset for quick experiments; validation shrinks in proportion |
| Seed | random | Fix it for reproducible runs |

The loss is chosen from the task: cross-entropy for classification (the network should end
with raw scores, not Softmax), mean squared error for regression.

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
```

Options: `--epochs`, `--batch-size`, `--lr`, `--optimizer`, `--weight-decay`, `--val-split`,
`--device`, `--max-samples`, `--seed`, `--save weights.pt`, `--data-root` (download folder for
built-in datasets, default `~/.cache/brickdnn/datasets`). Press <kbd>Ctrl</kbd>+<kbd>C</kbd> to
stop early; the weights are saved anyway.

## How it works

```
editor ──REST (polls every second)──▶ Django API ──spawns──▶ python -m brickdnn.training.runner <run dir>
                                         │                          │
                                         │  job.json  ─────────────▶│ builds the model from the exported code,
                                         │  STOP      ─────────────▶│ prepares the data, trains
                                         ◀───────────── state.json ─┤ progress, written atomically
                                         ◀───────────── weights.pt ─┘
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
- **Loss becomes NaN**: lower the learning rate.
- **Out of memory**: lower the batch size, the image size or the network size.
- **Something else failed**: the last lines of `output.log` in the run directory contain the
  full error.
