# Experiments: comparing runs

Every training is a run with its own view (curves, metrics, evaluation) in the training panel.
The **runs page** (`/runs`, *Runs* in the header, or *Compare runs* in the panel's
*History*) puts all of them side by side: a table of every run with its settings and
results, the curves of up to ten runs in one chart, settings against results, and a
read-only link to share a comparison. Trainings on your own machine can log their curves
into it too ([Runs that ran elsewhere](#runs-that-ran-elsewhere)).

## The runs table

The table lists the runs you can see: the ones you started and those of the networks shared
with you (as in the history: who started a run in a network you are a member of, the
network's editors and its owner see it). Filters above it narrow it down:

| Filter | |
| ------ | --- |
| Search | in the name, the note, the tags and the dataset's name |
| Network | the runs of one saved network (the history's *Compare runs* opens the page with it) |
| Dataset | runs of one dataset: an upload, a dataset of the Data tab or a built-in one |
| Type | neural networks, classical models, hybrid networks, ensembles, or runs that ran elsewhere |
| Status, tag | as they say |
| Starred only, only my runs | |

Click a column header to sort by it (again: the other way round); runs without a value
come last either way. The server filters, sorts and pages (50 runs a page), so the table
stays fast with many runs; it looks at the newest 5,000.

**Columns.** *Columns* chooses what the table shows:

- the run: status, when it started, how long it took, dataset, model (a neural network,
  the classical model, *SVC on learned features* for a hybrid network, *Ensemble*), the
  network, who started it, tags;
- its settings: optimizer, learning rate, batch size, epochs, weight decay, loss, validation
  and test split, patience, seed, the loss settings (label smoothing, focal gamma, Huber
  delta), an ensemble's method, a search's method and trials;
- its results, per split (*val.*, *test*, *train.*): the loss, accuracy, balanced accuracy,
  F1 (macro and weighted), ROC AUC and PR AUC (two classes), log loss, RMSE, MAE, R², MAPE …,
  mean IoU, Dice and pixel accuracy (segmentation), and the kept epoch.

The choice is kept on your account (`PUT /api/training/runs/preferences/`), so it is the same
in every browser; *Back to the default columns* forgets it. The defaults are the run, its
main settings and the main results of the tasks in the table (a regression's RMSE only when
there are regressions).

The results of a finished run are read from its state once and kept with the run
(`TrainingRun.summary`), so sorting by a test F1 does not read every run's files. They are
the numbers of the run view: the epoch whose weights were kept, and the final evaluation of
those weights ([training.md](training.md#metrics-of-the-finished-model)).

**A search** (hyperparameter search) is one row with the settings of its best trial; the
arrow before its name shows all trials in the table of the run view.

**Tags, a note and a star.** The pencil after a run's name edits its tags (words to find and
filter runs by, e.g. *baseline*; at most 20 of 40 characters), a note (2,000 characters)
and its star; the star in the first column toggles it. Who started the run and the owner and
editors of its network may change them; viewers of a network see them. The history marks
starred runs too.

## Comparing runs

Tick up to ten runs (also on different pages) to compare them below the table:

- **Curves.** The loss of every run in one chart (validation or training loss), and a second
  chart with a metric (validation accuracy, RMSE or IoU by default, by the task). Each run has
  its colour (after the eighth, dashed lines), the legend names them, the x axis counts
  epochs or the time since the start, and *Smooth* draws a moving average (with the start
  corrected, like TensorBoard). Hover or use the arrow keys to read every run's value at an
  epoch; the table button shows the values as a table.
- **Losses of different kinds** are not comparable (a focal loss is smaller than the
  cross-entropy of the same model, a mean absolute error is no squared one; see
  [Losses](training.md#losses)): when the runs trained with different losses, a warning says
  so and suggests comparing them by a metric. Runs of different tasks get a warning too.
- **Settings vs. result.** A scatter plot of any numeric setting or result against another,
  over the selected runs or all runs of the page, with a trend line from three runs on. The
  learning rate and the weight decay are drawn on a logarithmic axis.
- **Parallel coordinates.** One vertical axis per setting or result (by default the
  learning rate, batch size, epochs and the result of the scatter plot), each from its
  smallest to its largest value, and a line per run in its colour. Words (the optimizer, the
  loss) are spread evenly. The arrow keys go through the runs.

Every chart has a text summary for screen readers, a table view and keyboard access
([accessibility.md](accessibility.md)).

## Sharing a comparison

*Share comparison…* creates a read-only link (`/compare/<token>`) to the selected runs.
Anyone with the link sees them without an account: the runs' names, datasets, settings,
results and curves, with the columns and chart settings chosen when the link was made. It
does **not** show who started the runs, their networks, the trained weights or the logs, and
the tags and notes only if you include them. You can share runs you started and runs of
networks you own (not runs others started in a network you only edit). The links you made
are listed in the same dialog; turning one off deletes it, and the address stops working at
once. A run that is deleted (or that you may no longer share) drops out of the link.

The links, and the tags, notes and stars of your runs, are part of *Download my data*
(`run_comparisons.json`, `training_runs.json`) and are deleted with the account
([privacy.md](privacy.md)).

## Runs that ran elsewhere

Train the exported script on your own laptop or a GPU server, and its curves appear on the
runs page next to the runs trained here, with the badge *ran elsewhere*. Comparisons, tags,
notes and shared links work for them like for any run. Nothing is computed on the server for
them: they use none of your compute (CPU time) and do not wait in the training queue.

**1. The script.** *Export .py* (or the code view) writes a script whose CSV training loop
logs to Brick DNN; *Log to Brick DNN* in the code view is on by default and turns it off.
Logging is **inert** unless the environment variable `BRICKDNN_TOKEN` is set: without it the
script sends nothing and trains offline with exactly the same results (the same numbers,
the same weights) as a script without logging. The token is never written into the script.
With it, the script sends

- at the start: the run's name, the settings (epochs, batch size, learning rate, optimizer,
  seed, loss, validation and test split), the task, the name of the CSV file and the saved
  network it was exported from;
- after every epoch: the training and validation loss and the validation accuracy (a
  classification) or RMSE (a regression), with the seconds since the start;
- at the end: the test results (with `--test-split`, e.g. `--test-split 0.2`, which holds out
  that share of the rows and measures the model on them once at the end; without it the last
  validation results), and whether it completed, failed (an error) or was stopped (Ctrl+C).

`BRICKDNN_URL` names the server (default: the one the script was exported from, e.g.
`https://brickdnn.com`), `BRICKDNN_PROJECT` the id of the network the run belongs to (default:
the saved network it was exported from), `BRICKDNN_RUN_NAME` its name.

**2. A token.** On the page *API tokens*, create a token with *Only log training runs*: it
may create runs and log their curves, and nothing else (no networks, data or MCP), so it is
safe to keep on a shared GPU server. A full-access token logs too; read-only tokens and the
tokens of connected apps do not.

**3. Train.**

```bash
export BRICKDNN_TOKEN=bdnn_...            # PowerShell: $env:BRICKDNN_TOKEN = "bdnn_..."
python my_network.py --csv data.csv --epochs 20 --test-split 0.2
```

The script prints `Brick DNN: logging this run to …/runs`, and the run appears in the runs
table, live-updating (every five seconds) while it runs.

**It never stops the training.** The client sends the points in batches (at most every two
seconds, 200 at a time), with a timeout of five seconds. On the first error it prints one
warning, keeps up to 1,000 points and tries again ten seconds later; after three failures in
a row, or when the server refuses for good (a wrong token, a limit, a run stopped in the app),
it switches logging off and the training goes on.

**In a training loop of your own,** the same client is in the `brickdnn` package (only the
standard library; `pip install` it as in the [README](../README.md#use-the-generator-without-the-ui)):

```python
from brickdnn.tracking import BrickDNNRun

with BrickDNNRun.start(name="ResNet on my GPU", config={"learning_rate": 1e-3}, task="classification") as run:
    for epoch in range(1, 21):
        ...  # train and validate
        run.log(epoch=epoch, train_loss=train_loss, val_loss=val_loss, val_accuracy=accuracy)
    run.finish(summary={"test_accuracy": test_accuracy})
```

`log` takes an `epoch` or a `step` and any numbers (also tensors and NumPy numbers; others
are left out), `finish` the final numbers. Leaving the `with` block with an error ends the run
as failed, with Ctrl+C as stopped.

**In the app.** A run from elsewhere has its curves, settings and numbers, but no weights: it
cannot predict, be explained or join an ensemble. *Stop* ends it in the app; the script's
further reports are then refused and it stops logging. A run that reports nothing for 24
hours ends as failed. It belongs to whoever owns the token, and to the network given (one
they own or edit), whose members see it.

**Limits.** An account keeps at most 500 runs logged from elsewhere
(`BRICKDNN_TRACKING_MAX_RUNS`), 10 of them logging at a time (`BRICKDNN_TRACKING_MAX_ACTIVE`),
20,000 points per run (`BRICKDNN_TRACKING_MAX_POINTS`), 1,000 points and 512 KB per request,
32 metrics per point, and 120 requests a minute (`BRICKDNN_TRACKING_RATE`). The points are a
file next to the run and count toward the storage quota of the account.

**Privacy.** The server stores what the script sends (above), with the run: it is deleted
with the run or the account and is part of *Download my data* (`training_runs.json` has every
point). The activity log of agents records when a token created or finished a run, not every
batch of points.

## API

The endpoints are in [api.md](api.md#experiments-runs-side-by-side):
`GET /api/training/runs/table/`, `GET /api/training/runs/curves/?ids=…`,
`PATCH /api/training/runs/<id>/annotations/`, `GET`/`PUT /api/training/runs/preferences/`,
`GET`/`POST /api/training/comparisons/`, `DELETE /api/training/comparisons/<id>/` and the
public `GET /api/shared/runs/<token>/`; runs from elsewhere are logged with
[`/api/tracking/runs/`](api.md#logging-runs-from-elsewhere). Over MCP,
`list_training_runs` and `get_training_run` include them (`ran_elsewhere`, and `reported`: the
numbers the script sent at the end), see [mcp.md](mcp.md).
