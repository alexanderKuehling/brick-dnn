# Explaining models and looking inside them

Brick DNN can explain what a trained model does (explainable AI, XAI) and show what happens
inside a network, for learning. Everything below works on finished (or stopped) runs with
trained weights: open a run in the training panel and press **Explain the model** under its
name. AI assistants get the same through the MCP tools
[`explain_prediction` and `get_layer_activations`](mcp.md#explanations).

The run view itself shows a preview in **Why the model decides**, after the results:

- models trained on a CSV file: the permutation importance of the 10 most important features
  (*Show all* for the rest), and the decision boundary for models with exactly 2 input
  features;
- image classifiers: Grad-CAM (or saliency) over three validation images, two the model gets
  right and its first mistake; click an image to explain it in full.

*Show what matters* (or *Show where it looks*) computes them, since they cost compute like
predictions (see [Limits and costs](#limits-and-costs)); a run whose preview was shown in this
browser before shows it at once from the server's cache. *Open the full explanation* opens the
dialog on the matching tab. When a method does not work for a run, the section says why; runs
on demo data only offer a look inside the network. With the training panel at full width,
the section sits in a second column beside the charts and metrics.

The builder also has two aids that need no training at all:
**How data flows** and the **parameter formula and receptive field** in the inspector, see
[Learning aids in the builder](#learning-aids-in-the-builder).

| Tab in the dialog      | Method                                        | For                                    |
| ---------------------- | --------------------------------------------- | -------------------------------------- |
| Explain a prediction   | Saliency (SmoothGrad), Grad-CAM, occlusion    | image classifiers (CNNs, also MLPs)    |
| Explain a prediction   | Shapley values, integrated gradients          | models trained on a CSV file           |
| Feature importance     | Permutation importance                        | models trained on a CSV file           |
| Decision boundary      | Predictions on a grid, with training points   | models with exactly 2 input features   |
| Inside the network     | Output of every brick, filters, weight histograms | neural networks                    |

Classical models (Gaussian processes, support vector machines, gradient boosting) have no
layers: they get the
model-agnostic methods (Shapley values, permutation importance, decision boundary); a Gaussian
process regressor also shows its predictive standard deviation.

## Which sample

A prediction is explained for one sample: a **validation sample** of the run (data the model
did not train on; the list shows the model's guess and marks the mistakes, which are often the
most interesting ones), or one of your own: **draw** an image (white on black, like MNIST),
**upload** an image file (it is scaled to the network's input size) or **enter the values** of
a row.

The training does not store its split, but it can be repeated: the same seed (the setting, or
the one the run chose and wrote into its log), the same settings and the same split functions
give the same rows. The server reads them once from the dataset (at most 64 images, or 1,000
validation rows and 400 training rows) and keeps them in the run's folder
(`explain/pool.npz`). If the dataset was deleted, the seed is unknown or the CSV file is larger
than 64 MB, only samples of your own can be explained.

## Images

The heatmaps are laid over the image (the slider sets how strongly). All three explain one
class: the predicted one, or another one you pick ("why not a 7?").

- **Saliency (SmoothGrad).** How much the score of the class changes when a pixel changes a
  little: the absolute input gradient, the largest over the colour channels, averaged over 16
  copies of the image with a little noise (15 % of its value range) — the noise makes the map
  less speckled (Smilkov et al., 2017). *Limits:* it shows sensitivity, not a reason; it stays
  noisy and often lights up edges whatever the model does.
- **Grad-CAM** (Selvaraju et al., 2017). The feature maps of a convolution brick, each
  weighted by the mean gradient of the class score over it, added and clipped at zero: where
  that brick sees evidence for the class. By default the last `Conv2d` brick; any brick that
  puts out feature maps (channels × height × width) can be chosen. *Limits:* coarse — one value
  per position of that brick (a 7×7 map for a 28×28 image after two poolings), scaled up; it
  says nothing for networks without convolutions (an MLP on flattened pixels).
- **Occlusion** (Zeiler & Fergus, 2014). A square patch (about 1/7 of the image side, half
  overlapping) is set to the dataset mean, position by position, and the drop of the class
  probability is recorded. Model-agnostic and easy to read. *Limits:* only as fine as the
  patch; it cannot see what two distant regions do together; it is the slowest of the three
  (about 170 forward passes) and is skipped if it would not finish in time.

## Tables (CSV files)

- **Shapley values** of one prediction: how much each feature moved the prediction away from
  the prediction for the **mean of the training rows** (the baseline). A feature that is
  "absent" takes its mean, and the values are averaged over all orders in which the features
  can be added (the *baseline Shapley* values of Sundararajan & Najmi, 2020). They add up
  exactly to *prediction − prediction for the mean row*. With up to 10 features they are exact
  (2^10 = 1,024 predictions); with more they are estimated from random orders of the features
  (at most 20,000 predictions; the sum stays exact). For classification they explain the
  probability of the predicted class (or of the one you pick), for regression the value in the
  units of the target. The predicted class here is the most probable one, also for a binary
  classification with another [decision threshold](training.md#metrics-of-the-finished-model);
  pick the positive class to explain its probability, which the threshold is compared with.
  *Limits:* the mean row may be an impossible one (e.g. a mean of 0/1
  features); correlated features share their credit in ways that can surprise; it is a
  local explanation of this one row.
- **Integrated gradients** (networks only; Sundararajan et al., 2017): the gradient of the
  output along the straight path from the mean row to this row (32 steps), times the change of
  each feature. They add up to the same difference (up to the error of the 32 steps). For a
  linear model both methods give exactly *weight × (value − mean)*.
- **Permutation importance**: how much the validation score (accuracy, or R² for regression)
  drops when the values of one feature are shuffled across the rows, which breaks its link to
  the target. Mean and standard deviation over 5 shuffles of up to 1,000 validation rows.
  *Limits:* with correlated features each looks unimportant (the model finds the information in
  the other); shuffling creates rows that never occur; a drop says what the model uses, not
  what matters in the world.
- **Decision boundary** (exactly 2 input features): the model's prediction on a 64×64 grid
  over the range of the data (plus 10 %), coloured by class (pale where the model hesitates) or
  by value, with up to 300 training points on top. Very instructive to compare an MLP, an SVM
  and a Gaussian process on the same data. A Gaussian process regressor also shows its
  predictive standard deviation, which grows away from the training points.

## Inside the network

- **Activations**: the output of every brick for the chosen sample, in the order the data
  flows: feature maps of convolutions as small grey images (the first 16 channels, each scaled
  on its own), vectors as bars (positive red, negative blue), sequences as heatmaps (rows =
  positions), with the shape, mean, standard deviation, share of zeros (e.g. after a ReLU) and
  a histogram. The brick's description from the catalogue says what it computes.
- **Learned filters** of the first `Conv2d` brick: red for positive, blue for negative weights
  (one input channel), or as colours (three input channels).
- **Weight histograms** of every brick with weights, with mean, standard deviation and norm.

Runs trained on random demo data have nothing to explain, but their activations (for a random
input) and weights can still be looked at. Text models (token inputs) can only be trained on
demo data so far, so there are no token attributions yet.

## Limits and costs

An explanation runs like a [prediction](training.md#using-the-trained-weights): in a separate,
short-lived process (`python -m brickdnn.training.predict <run dir>` with `{"explain": …}` on
stdin), on the CPU with two threads and a lower priority, at most 30 seconds, one at a time on
the server by default (`BRICKDNN_MAX_PREDICTIONS`), and its CPU time counts as a prediction of
whoever asks (everyone who can see the run may). Optional parts (occlusion, more shuffles of
the permutation importance) are skipped after 18 seconds, with a note. Answers are cached in
the run's folder (`explain/cache/`, at most 64 answers and 8 MB per run, the least recently
used go first): asking again costs nothing and does not count against the rate limit
(`BRICKDNN_PREDICT_RATE`, shared with predictions).

Measured with the process's two threads on a shared development machine that was busy with
other work (load around 20 on 4 cores), so the production server should be no slower; the
time is that of the process, the request takes 1–2 s more:

| Run                                                                 | Method                          | Time  |
| ------------------------------------------------------------------- | ------------------------------- | ----- |
| CNN example on MNIST (421k parameters; 1 epoch on 4,000 images)     | `samples` (first: reads MNIST)  | 7.9 s |
|                                                                     | `image` (saliency, Grad-CAM, occlusion) | 5.0 s |
|                                                                     | `activations`, `weights`        | 4.3 s, 4.4 s |
| MLP on a CSV file (4 features, 3 classes, 300 rows)                 | `tabular`, `permutation`        | 3.7 s, 4.9 s |
| MLP on 2-feature "moons" data                                       | `boundary`                      | 3.6 s |
| SVM (SVC) on the same data                                          | `tabular`, `permutation`, `boundary` | 3.1 s, 3.6 s, 2.3 s |
| Gaussian process classifier / regressor                             | `boundary` (regressor with std) | 3.1 s, 3.2 s |

Most of it is starting Python and importing PyTorch or scikit-learn (2–3 s); answers from the
cache take milliseconds.

## Learning aids in the builder

- **How data flows** (button at the bottom of the canvas): follows one sample through the
  network brick by brick. Each step centres its brick and shows the tensor before and after
  (without the batch dimension), what changed ("28×28 → 14×14 pixels", "1 → 32 channels"),
  what the brick computes and how its parameters add up. *Play* steps on by itself.
- **Parameter formula** in the inspector: e.g. `3×3×32×64 + 64 = 18,496` for a 3×3
  convolution from 32 to 64 channels (a kernel per input and output channel, plus a bias per
  output channel); also for Linear, normalisation, embedding and single-layer recurrent bricks.
- **Receptive field** in the inspector of bricks in a convolution stack: how many input pixels
  one output value depends on, and how far apart neighbouring outputs are. With kernel *k* and
  stride *s* it grows as *rf' = rf + (k − 1)·jump* and *jump' = jump·s*. Bricks after a
  Flatten, Linear or recurrent brick depend on the whole input.

Both only use the analysis of the network (shapes and settings), so they need no training.

## API

`GET /api/training/runs/<id>/explain/` lists the methods for a run (`available`, and a
`reason` for the others). `POST` computes one: `{"method", "sample"?: n | "input"?: …,
"input_scale"?, "target"?, "layer"?, "node"?}`, see [api.md](api.md#explanations).
