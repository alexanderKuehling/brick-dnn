/**
 * The guides in English. Everything they say must be true of the app (docs/training.md,
 * docs/data-preparation.md, docs/explain.md); the screenshots are those of images/.
 */
import type { Guide, GuideIndex } from './types.ts';

const REPOSITORY = 'https://github.com/alexanderKuehling/brick-dnn';
const DOCS = `${REPOSITORY}/blob/main/docs`;
const DATE = '2026-11-16';
const GDPR_DATE = '2026-11-17';
const BASICS_DATE = '2026-11-18';

const index: GuideIndex = {
  title: 'Guides: deep learning, tabular data, segmentation and GDPR – Brick DNN',
  description:
    'Practical deep-learning guides: a neural network without code exported to PyTorch, gradient boosting vs. MLP, U-Net segmentation, rare classes and the GDPR.',
  h1: 'Guides',
  intro:
    'Step-by-step guides to building, training and judging deep-learning models with Brick DNN, and to the decisions along the way: which model for a table, how to segment images, what to do when one class is rare, and how to train with data that falls under the GDPR.',
};

const guides: Guide[] = [
  {
    id: 'no-code',
    slug: 'build-a-neural-network-without-code',
    title: 'Build a neural network without code and export it to PyTorch',
    description:
      'No-code deep learning: build a neural network in the browser from layers, check every tensor shape and download clean PyTorch code that runs without Brick DNN.',
    h1: 'Build a neural network without code and export it to PyTorch',
    summary:
      'Connect layers on a canvas, let Brick DNN check every shape, and download a PyTorch script that runs on its own.',
    lead: 'You can design a neural network by connecting layers on a canvas, and still end up with ordinary PyTorch code. This guide builds a small convolutional network for 28×28 images, checks it and exports it, without an account.',
    published: DATE,
    modified: DATE,
    type: 'HowTo',
    totalTime: 'PT15M',
    sections: [
      {
        id: 'before',
        title: 'What you need',
        html: '<p>A current browser. Building, checking and exporting work without an account and cost nothing; only saving networks in an account, uploading data and training on the server need a free account. The exported code needs Python and PyTorch on your own machine.</p>',
      },
      {
        id: 'build',
        title: 'Build the network',
        html: '<p>Open the <a href="/app">editor</a>. On a first visit it shows an example so that there is something to look at; <strong>File → New network</strong> gives an empty canvas.</p>',
        steps: [
          {
            title: 'Start with an Input',
            html: 'Drag <code>Input</code> from the palette onto the canvas and set its shape without the batch dimension: <code>1, 28, 28</code> for one grayscale channel of 28 × 28 pixels, or <code>784</code> for the same image as a flat vector.',
          },
          {
            title: 'Add layers',
            html: 'Drag <code>Conv2d</code> onto the canvas and connect the right edge of the Input to the left edge of the new layer. With a layer selected, clicking a layer in the palette (or pressing Enter on it) appends the new one after the selected layer and connects them. Continue with <code>Activation</code> (ReLU), <code>MaxPool2d</code>, a second convolution block, <code>Flatten</code> and two <code>Linear</code> layers, the last with 10 outputs.',
          },
          {
            title: 'Set what matters, let Brick DNN fill in the rest',
            html: 'Select a layer to see its settings. You set what is a design decision, such as <code>out_channels</code> or <code>kernel_size</code>; fields that follow from the incoming tensor, such as <code>in_channels</code> and <code>in_features</code>, are filled in automatically, also after you change an earlier layer.',
          },
          {
            title: 'Read the shapes',
            html: 'Every layer shows its output shape and parameter count while you build. A mistake is marked in red before anything runs, with a message that says what to change, for example that a kernel is larger than its input. The layers after it have no shape until it is fixed.',
          },
        ],
        image: {
          id: 'checks',
          alt: 'A convolutional network in the editor with one Conv2d layer marked red and the message “Kernel size 16 is larger than the (padded) input size 14”; the layers after it show no shape.',
          caption:
            'A mistake is shown where it happens, with the reason. The layers after it wait for the fix.',
        },
        after:
          '<p>Skip connections (<code>Add</code>), merges (<code>Concat</code>) and several inputs work the same way: connect the tensors and the shape inference checks that they fit. <strong>Tidy up layout</strong> arranges a network that has grown untidy.</p>',
      },
      {
        id: 'understand',
        title: 'Check that it does what you think',
        html: '<p>The toolbar counts the trainable parameters of the whole network. The inspector of a selected layer explains how its parameters add up and what its receptive field is. <em>How data flows</em> steps one sample through the network, layer by layer, and shows how each layer changes its shape, for example how max pooling turns 32 × 28 × 28 values into 32 × 14 × 14.</p>',
        image: {
          id: 'data-flow',
          alt: 'The “How data flows” card in the editor follows one sample through a CNN and explains how max pooling halves 32 × 28 × 28 values to 32 × 14 × 14.',
          caption: 'How data flows: one sample, one layer at a time.',
        },
      },
      {
        id: 'export',
        title: 'Export the PyTorch code',
        html: '<p><strong>Code</strong> in the toolbar opens the generated script. It is a self-contained Python file: an <code>nn.Module</code> with one named attribute per layer in <code>__init__</code> and a <code>forward</code> method in the order of the graph. You can change the class name, and whether the file is a runnable script (a <code>__main__</code> part that prints the model and runs a random batch through it). For a network with one flat input and one flat output it can also contain a training loop for CSV files. <strong>Export .py</strong> (Ctrl+E) downloads it.</p>',
        image: {
          id: 'code-export',
          alt: 'The code dialog with the generated PyTorch module of a small CNN: the layers in __init__, the forward method, and the options for class name and runnable script.',
          caption: 'The code dialog: an nn.Module with named layers, ready to download.',
        },
        after: `<p>On your machine, <code>python model.py</code> prints the model, the number of trainable parameters and the output shape of a random batch, which should match what the editor showed. The code imports nothing from Brick DNN, so you can change it, put it in a repository or a paper, and train it with your own loop. The <a href="/examples/cnn-mnist/">example page of this CNN</a> shows the complete script.</p>`,
      },
      {
        id: 'train',
        title: 'Optional: train it on the server',
        html: '<p>With a free account, <strong>Train</strong> runs the network on Brick DNN’s server in Switzerland. Choose MNIST, Fashion-MNIST, CIFAR-10, your own CSV file or a ZIP file of images. Brick DNN checks beforehand whether network and data fit, adapts the Input and the last layer if only those differ, and then shows loss and accuracy per epoch. When the run is finished, download the weights (<code>.pt</code>) together with the matching script:</p><pre class="snippet"><code>import torch\nfrom model import NeuralNetwork  # the downloaded script\n\nmodel = NeuralNetwork()\nmodel.load_state_dict(torch.load("model.pt"))\nmodel.eval()</code></pre>',
        image: {
          id: 'export',
          alt: 'The end of a finished training run with buttons to download the weights, the preprocessing and the script, or to load the network into the editor.',
          caption: 'After training: the weights, the preprocessing and the script that fits them.',
        },
      },
      {
        id: 'keep',
        title: 'Keep and share the network',
        html: `<p><strong>Save as JSON file</strong> in the File menu keeps the graph without an account, and <strong>Open JSON file</strong> brings it back. With an account, networks are saved with their history and can be shared with people or by link and edited together live. The graph format is documented in <a href="${DOCS}/graph-format.md">graph-format.md</a>, and the same networks can be built by an AI assistant over the <a href="${DOCS}/mcp.md">MCP server</a>.</p>`,
      },
    ],
    examples: ['cnn_mnist', 'mlp_mnist', 'resnet_block'],
  },
  {
    id: 'tabular',
    slug: 'gradient-boosting-vs-mlp',
    title: 'Tabular data: gradient boosting vs. MLP, and when to use which',
    description:
      'Gradient boosting or a neural network for tabular data? How both see a table, what each needs, and how to compare them fairly on the same split.',
    h1: 'Tabular data: gradient boosting vs. MLP, and when to use which',
    summary:
      'How trees and neural networks see a table, what each needs, and how to compare them fairly on the same split.',
    lead: 'For a table of rows and columns, gradient-boosted trees are usually the stronger first model, and a multi-layer perceptron (MLP) is the more flexible one. Which one wins depends on your data, so the useful question is how to find out quickly and fairly.',
    published: DATE,
    modified: DATE,
    type: 'Article',
    sections: [
      {
        id: 'short',
        title: 'The short answer',
        html: '<p>Start with gradient boosting. It needs little preparation and little tuning, copes with features on very different scales and finds interactions between columns by itself. Try an MLP next to it when the table is large, when the relationship is smooth, when you need several outputs at once, or when the table is only one of the inputs, next to an image or a text. Then compare both on the same held-out rows.</p>',
      },
      {
        id: 'how',
        title: 'How each model sees a table',
        html: '<p><strong>Gradient boosting</strong> builds decision trees one after the other, each correcting the errors of the trees before it. A tree splits on one column at a time (“rainfall above 41 mm”), so the scale of a column does not matter, a few extreme values do no harm, and a step in the data is easy to represent. Outside the range of the training data, a tree predicts the value of its outermost leaf: it does not extrapolate.</p><p><strong>An MLP</strong> multiplies the standardised features by weights, adds them up and passes the result through non-linear activations, layer after layer. It represents smooth relationships well and learns from all columns at once, but it needs numbers on comparable scales, enough rows to fit its weights, and settings such as the learning rate and the number of epochs.</p>',
      },
      {
        id: 'compare',
        title: 'Side by side',
        html: `<div class="table" role="region" aria-labelledby="compare-caption" tabindex="0"><table><caption id="compare-caption">Gradient boosting and an MLP on tabular data, in Brick DNN</caption><thead><tr><th scope="col">Question</th><th scope="col">Gradient boosting</th><th scope="col">MLP</th></tr></thead><tbody>
<tr><th scope="row">Preparation</th><td>Fill in missing values and encode categories (in the Data tab); no scaling needed</td><td>The same; features are standardised by the training (mean 0, standard deviation 1)</td></tr>
<tr><th scope="row">Rows</th><td>Up to 1,000,000 training rows, time about linear in the rows</td><td>Any number, in batches; profits from many rows</td></tr>
<tr><th scope="row">Settings to tune</th><td>Learning rate, leaves per tree, rows per leaf; good defaults</td><td>Layer sizes, learning rate, batch size, epochs, dropout</td></tr>
<tr><th scope="row">Training</th><td>Fitted once, in seconds to minutes</td><td>Epoch by epoch, with curves and early stopping</td></tr>
<tr><th scope="row">Rare classes</th><td><code>class_weight</code> <code>balanced</code></td><td>Weighted cross-entropy or focal loss</td></tr>
<tr><th scope="row">Several targets, other inputs</th><td>One target</td><td>Several outputs; can be one branch of a larger network</td></tr>
<tr><th scope="row">Export</th><td>scikit-learn script and fitted pipeline (<code>.joblib</code>)</td><td>PyTorch script and weights (<code>.pt</code>)</td></tr>
</tbody></table></div>`,
      },
      {
        id: 'fair',
        title: 'Compare them on the same rows',
        html: '<p>A comparison is only fair when both models learn from the same training rows and are judged on the same rows they never saw. In Brick DNN that is what a prepared dataset gives you:</p>',
        steps: [
          {
            title: 'Prepare the table once',
            html: 'Import the file in the <strong>Data</strong> tab, choose the target, and add steps for missing values, outliers, categories and dates. Every row is assigned to training, validation or test once, and steps that learn values (a median, a scaling) learn them from the training rows only.',
          },
          {
            title: 'Create the training dataset',
            html: '<strong>Create training dataset</strong> saves the prepared table with its split. Both models will use this split instead of a random one of their own.',
          },
          {
            title: 'Train gradient boosting',
            html: 'Open the <a href="/examples/gradient-boosting-classifier/">gradient boosting example</a> (or its regression brick), choose the dataset and train. The Input adapts to the number of features. A hyperparameter search with cross-validation on the training rows tunes the trees without touching the validation rows.',
          },
          {
            title: 'Train an MLP',
            html: 'Open the <a href="/examples/tabular-regression/">tabular MLP example</a>, or start from an empty canvas: choosing the dataset there starts a small network for it (Input → Linear → ReLU → Linear). Turn on early stopping and train.',
          },
          {
            title: 'Compare the test numbers',
            html: 'Compare on the test rows, not on the validation rows that picked the epoch or the trial: balanced accuracy, macro F1 or PR AUC for classes, RMSE and R² for a number. The History tab keeps both runs, each with the exact network that was trained.',
          },
        ],
        image: {
          id: 'data-prep',
          alt: 'A preparation in the Data tab: on the left the steps for missing values, outliers and categories with what each did, on the right the prepared table with a small distribution chart in each column header.',
          caption: 'One preparation, one split, for every model you compare.',
        },
      },
      {
        id: 'tune',
        title: 'Tune both, briefly',
        html: '<p>Untuned defaults favour gradient boosting, because its defaults are good for most tables. Give the MLP a short random search over learning rate and batch size; give the trees one over the learning rate, the leaves per tree and the rows per leaf. Rank both searches by the metric you will report, not by the loss.</p>',
        image: {
          id: 'search-results',
          alt: 'The trials of a random search for gradient boosting: learning rate, depth, rows per leaf and class weights of each trial with its cross-validated loss, accuracy and PR AUC; the best trial is highlighted.',
          caption: 'A search over the parameters of gradient boosting, ranked by PR AUC.',
        },
      },
      {
        id: 'which',
        title: 'Which one, then?',
        html: '<ul class="checks"><li><strong>Gradient boosting</strong> when the table is small or medium, has columns of mixed scales and meaning, contains thresholds and steps, or when you need a strong result quickly.</li><li><strong>An MLP</strong> when the table is large and the relationship smooth, when it must predict several targets at once, when it has to extrapolate a little beyond the training range, or when the table is one input of a network that also sees images or text (see the <a href="/examples/multi-input/">multi-input example</a>).</li><li><strong>Both</strong>: a <a href="/examples/hybrid-mlp-gradient-boosting/">hybrid network</a> lets an MLP learn features for gradient boosting, and a <a href="/examples/voting-ensemble-classifier/">voting</a> or <a href="/examples/stacking-ensemble-regression/">stacking ensemble</a> combines several classical models.</li></ul>',
      },
      {
        id: 'explain',
        title: 'Look at what the winner learned',
        html: '<p><strong>Explain the model</strong> under a finished run works for both kinds: permutation importance shows how much the validation score drops when one column is shuffled, and Shapley values show how each feature pushed one prediction. If the two models rely on very different columns, look at the data again before trusting either.</p>',
        image: {
          id: 'explain',
          alt: 'Permutation importance of a gradient boosting model: how much the validation accuracy drops when one feature is shuffled, most for rainfall, then temperature and nitrogen.',
          caption: 'Permutation importance: which columns the model really uses.',
        },
      },
    ],
    examples: [
      'gradient_boosting_classifier',
      'tabular_regression',
      'hybrid_mlp_gradient_boosting',
    ],
  },
  {
    id: 'segmentation',
    slug: 'image-segmentation-u-net',
    title: 'Image segmentation with a U-Net: from masks to a trained model',
    description:
      'Deep learning for image segmentation: train a U-Net, prepare images and masks, check the network, choose batch size and loss, and read mean IoU and Dice.',
    h1: 'Image segmentation with a U-Net: from masks to a trained model',
    summary:
      'From a ZIP file of images and masks to a trained U-Net, with mean IoU, Dice and the predicted masks.',
    lead: 'Segmentation gives every pixel a class: background, road, cell, defect. This guide trains a U-Net in Brick DNN, first on the built-in Shapes data, then on your own images and masks, and explains how to read the results.',
    published: DATE,
    modified: DATE,
    type: 'HowTo',
    totalTime: 'PT20M',
    sections: [
      {
        id: 'idea',
        title: 'What a segmentation network needs',
        html: '<p>A classifier gives one answer per image; a segmentation network gives one per pixel. Its output therefore has the height and width of its Input and one channel per class: <code>classes, height, width</code>. The network ends with raw scores, not with Softmax. A U-Net is built for exactly this: an encoder that halves the image to see what is in it, a decoder that brings it back to full size, and skip connections that carry the details across (see the <a href="/examples/unet/">U-Net example</a> for every layer and its shape).</p>',
      },
      {
        id: 'first',
        title: 'Train it once on built-in data',
        html: '<p>Before preparing your own data, check the whole path on data that needs no upload:</p>',
        steps: [
          {
            title: 'Open the U-Net example',
            html: 'Open the <a href="/app?example=unet">U-Net in the editor</a>. Its Input is <code>3, 64, 64</code> and its last 1×1 convolution has four output channels.',
          },
          {
            title: 'Choose Shapes',
            html: 'Press <strong>Train</strong> and choose <strong>Built-in → Shapes</strong>: 500 generated images of 64 × 64 pixels with squares, circles and triangles on a noisy background, and a mask for each with four classes. It fits the network as it is.',
          },
          {
            title: 'Start the training',
            html: 'Keep the batch size of 8 and ten epochs and start. On a CPU this takes a few CPU minutes and reaches about 80 % mean IoU, well within the 15 CPU minutes per run of the free plan.',
          },
        ],
      },
      {
        id: 'data',
        title: 'Prepare your own images and masks',
        html: '<p>Your data is one ZIP file with a folder <code>images</code> and a folder <code>masks</code>. An image and its mask have the same file name; the mask may also end in <code>_mask</code>, <code>_label</code>, <code>_seg</code> or <code>_gt</code>, and the extension may differ:</p><pre class="snippet"><code>images/001.jpg        masks/001.png\nimages/002.jpg        masks/002_mask.png\nclasses.txt           (optional: one class name per line)</code></pre><p>Folders <code>train/</code> and <code>val/</code> (and optionally <code>test/</code>), each with its own <code>images</code> and <code>masks</code>, fix the split; otherwise the pairs are split at random. Images may have any size up to 4096 × 4096 and are resized to the Input; masks are resized with nearest neighbour, so no classes are mixed.</p><p>How a mask is read is decided once for the whole dataset:</p><div class="table" role="region" aria-labelledby="masks-caption" tabindex="0"><table><caption id="masks-caption">Mask formats that Brick DNN reads</caption><thead><tr><th scope="col">Masks</th><th scope="col">Read as</th></tr></thead><tbody><tr><td>Only black (0) and white (255)</td><td>Two classes: background and foreground</td></tr><tr><td>Grey values or palette indices below 32</td><td>The class of each pixel (0, 1, 2 …); 255 marks pixels to ignore</td></tr><tr><td>At most 32 other grey values, e.g. 0, 128, 255</td><td>Classes in ascending order</td></tr><tr><td>Colour masks with at most 32 colours</td><td>One class per colour (order from <code>classes.json</code>, or black first)</td></tr></tbody></table></div><p>Save masks as PNG without smoothing. JPEG compression and anti-aliased edges create in-between values, and such masks are refused at upload with the reason.</p>',
      },
      {
        id: 'network',
        title: 'Fit the network to your data',
        html: '<p>When you choose your dataset, Brick DNN sets the Input shape and the <code>out_channels</code> of the last convolution to the number of classes, and says what it changed. Two things to keep in mind:</p><ul class="checks"><li>Every pooling layer halves height and width, so with three of them both must be divisible by 8. The check explains it if the output ends up smaller than the Input.</li><li>Larger inputs cost more: CPU time grows with the number of pixels. Start at 64 × 64 or 128 × 128 and go up once the network learns; the Input can be at most 512 × 512.</li></ul>',
      },
      {
        id: 'settings',
        title: 'Batch size and loss',
        html: '<p>Every layer keeps its output for every pixel of a batch, so batches are much smaller than for classification: 4 to 16 is usual. A batch that would not fit into memory is refused before the start, with the batch size that fits.</p><p>The default loss is the cross-entropy of every pixel. When one class covers only a few pixels (a thin crack, a small cell), choose <strong>Cross-entropy + Dice</strong>, which adds the Dice score of the predicted probabilities, or the <strong>weighted cross-entropy</strong> with balanced class weights from the pixel counts. The check suggests them when one class has more than 97 % of the pixels.</p>',
      },
      {
        id: 'results',
        title: 'Read the results',
        html: '<p>After every epoch Brick DNN measures three numbers on the validation images:</p><ul class="checks"><li><strong>Mean IoU</strong>, for each class the overlap of predicted and true pixels divided by their union, averaged over the classes. It is the headline number.</li><li><strong>Dice</strong>, twice the overlap divided by the sum of both, averaged the same way; always at least the IoU.</li><li><strong>Pixel accuracy</strong>, the share of pixels with the right class. With a lot of background it is high even for a poor network, so do not judge by it alone.</li></ul><p>The run view adds the IoU of every class for the kept weights and shows predictions for eight validation images: the image, the true mask, the predicted mask and an overlay.</p>',
        image: {
          id: 'segmentation',
          alt: 'Predictions of a U-Net trained on the Shapes data: for each validation image the image, the true mask, the predicted mask and an overlay, with IoU and pixel accuracy.',
          caption: 'Predicted masks next to the true ones, with the IoU of each image.',
        },
      },
      {
        id: 'use',
        title: 'Use the trained model',
        html: `<p>Predict a new image on the server with the REST API or the MCP tool <code>predict</code>: it is converted and resized like the training images, and the answer contains the share of each class and the class map as a small PNG whose pixel values are the class indices. Or download the weights and the script and run it yourself; <code>output.argmax(dim=1)</code> gives the class of every pixel. The details are in <a href="${DOCS}/training.md#image-segmentation">training.md</a>.</p>`,
      },
    ],
    examples: ['unet', 'unet_classifier', 'resnet_block'],
  },
  {
    id: 'imbalanced',
    slug: 'imbalanced-classes',
    title: 'Imbalanced classes: class weights, focal loss, oversampling, threshold',
    description:
      'When one class is rare: why accuracy misleads, and when to use class weights, focal loss, oversampling or a different decision threshold. With examples.',
    h1: 'Imbalanced classes: class weights, focal loss, oversampling and the decision threshold',
    summary:
      'Why accuracy misleads when one class is rare, and the four levers against it: weights, focal loss, oversampling and the threshold.',
    lead: 'When 2 % of the rows are fraud, a model that never predicts fraud is 98 % accurate and useless. This guide shows how to see the problem in the metrics and which of the four levers to pull first.',
    published: DATE,
    modified: DATE,
    type: 'Article',
    sections: [
      {
        id: 'metrics',
        title: 'First, measure the right thing',
        html: '<p>Accuracy counts every row the same, so the large class dominates it. On imbalanced data, look at metrics that give the rare class its own weight:</p><ul class="checks"><li><strong>Balanced accuracy</strong>, the mean recall over the classes: 50 % for a model that always predicts the larger of two classes, however unbalanced they are.</li><li><strong>Recall and precision</strong> of the rare class: how many of its rows were found, and how many alarms were right.</li><li><strong>PR AUC</strong> (average precision, for two classes), which says more than ROC AUC when positives are rare; a model that guesses gets the share of positives.</li><li><strong>The confusion matrix</strong>, which shows exactly where the errors are.</li></ul><p>Brick DNN reports all of them for every finished classification run, on the validation and the test rows. The readiness check of the Data tab points out classes that are more imbalanced than 3 : 1.</p>',
      },
      {
        id: 'levers',
        title: 'Four levers',
        html: `<div class="table" role="region" aria-labelledby="levers-caption" tabindex="0"><table><caption id="levers-caption">What each lever changes, and where it is in Brick DNN</caption><thead><tr><th scope="col">Lever</th><th scope="col">What it changes</th><th scope="col">Where</th></tr></thead><tbody>
<tr><th scope="row">Class weights</th><td>Mistakes on the rare class cost more in training. <code>balanced</code> weighs each class by <code>n / (k · count)</code>, so every class counts as much as a whole.</td><td>Loss <em>Weighted cross-entropy</em> of a network; <code>class_weight</code> of gradient boosting and SVMs</td></tr>
<tr><th scope="row">Focal loss</th><td>Rows the network already classifies well count less (<code>focal_gamma</code>, default 2), so training concentrates on the hard ones; class weights can be added as its alpha.</td><td>Loss <em>Focal loss</em> of a network</td></tr>
<tr><th scope="row">Oversampling</th><td>Repeats rare rows of the training split (or drops common ones), which changes the class shares the model sees and makes epochs longer.</td><td>Step <em>Balance classes</em> in the Data tab, training rows only</td></tr>
<tr><th scope="row">Decision threshold</th><td>Changes nothing in training: it moves the probability at which a row counts as positive.</td><td>Under <em>Metrics</em> of a finished run with two classes</td></tr>
</tbody></table></div>`,
      },
      {
        id: 'threshold',
        title: 'Start with the threshold',
        html: '<p>A model that ranks the rare rows above the common ones but rarely gives them a probability above 0.5 only needs a lower threshold. That costs nothing: no new training, no copied rows. Under <strong>Metrics</strong> of a finished binary run, a slider over the measured thresholds shows precision, recall, F1 and the counts of true and false positives on the validation and the test rows. Two buttons jump to the threshold with the best F1 and to the highest threshold with a recall of at least 80 %. <strong>Use this threshold for predictions</strong> saves it on the run, and from then on its predictions use it.</p><p>The positive class is the second one: in alphabetical order of text labels, index 1 for class indices. <code>fraud</code> before <code>ok</code> makes <code>ok</code> positive, so name the classes so that the rare one comes second, or use 0 and 1.</p>',
        image: {
          id: 'evaluation',
          alt: 'The evaluation of a finished run: a confusion matrix with the true classes in the rows and the predicted classes in the columns, almost all rows on the diagonal.',
          caption: 'The confusion matrix of a finished run shows where the errors are.',
        },
      },
      {
        id: 'weights',
        title: 'Then weights or the focal loss',
        html: '<p>When the model does not separate the rare class well at any threshold (a low PR AUC), change what it learns. For gradient boosting and SVMs, set <code>class_weight</code> to <code>balanced</code>, as in the <a href="/examples/gradient-boosting-classifier/">gradient boosting example</a>. For a network, choose the weighted cross-entropy, or the focal loss when there are many easy and few hard rows. The run shows the weights it computed from the class counts of the training rows.</p><p>Weights and oversampling both push the predicted probabilities towards the rare class, so look at the threshold again afterwards. Combining weights and oversampling is rarely needed: they do much the same.</p>',
      },
      {
        id: 'oversampling',
        title: 'Oversampling, when the model needs to see more rare rows',
        html: '<p><em>Balance classes</em> in the Data tab oversamples the rare class or undersamples the common one, by default in the training rows only: validation and test keep the real class shares, so their numbers stay honest. It suits models whose training does not take weights, and it is easy to reason about. The price is longer epochs and a model that sees the same rare rows many times, which can make it memorise them.</p>',
      },
      {
        id: 'compare',
        title: 'Compare the options fairly',
        html: '<p>Losses of different kinds cannot be compared with each other (a focal loss is smaller than the cross-entropy of the same model), so compare runs by their metrics. A hyperparameter search can try the losses for you: give it the list <code>cross_entropy</code>, <code>weighted_cross_entropy</code> and <code>focal</code>, and rank the trials by PR AUC, balanced accuracy or macro F1. A search over the loss ranked by the loss itself is refused for this reason.</p>',
        image: {
          id: 'search-results',
          alt: 'The trials of a search for a gradient boosting model, each with its learning rate, depth, rows per leaf and class weights, and its cross-validated loss, accuracy and PR AUC; the best trial is highlighted.',
          caption: 'A search that also tries class weights, ranked by PR AUC instead of accuracy.',
        },
      },
      {
        id: 'order',
        title: 'In short',
        html: '<ol class="numbered"><li>Measure with balanced accuracy, recall of the rare class, PR AUC and the confusion matrix.</li><li>Move the decision threshold; often that is enough.</li><li>If the classes do not separate, use class weights or the focal loss, then set the threshold again.</li><li>Oversample when the model cannot take weights, or to try it against them, on the training rows only.</li></ol>',
      },
    ],
    examples: ['gradient_boosting_classifier', 'svm_classifier', 'voting_ensemble_classifier'],
  },
  {
    id: 'gdpr',
    slug: 'gdpr-compliant-deep-learning',
    title: 'GDPR-compliant deep learning: train neural networks with data in Europe',
    description:
      'Training deep-learning models under the GDPR: personal or anonymised data, where Brick DNN stores and trains in Switzerland, export, deletion and self-hosting.',
    h1: 'GDPR-compliant deep learning: training neural networks with data in Europe',
    summary:
      'Which data needs care under the GDPR, where brickdnn.com stores and trains in Switzerland, and what to do with data that has to stay personal.',
    lead: 'The GDPR does not forbid training neural networks. It asks what the data is, where it goes and who processes it. This guide answers these questions for Brick DNN: which data needs care, where brickdnn.com stores data and trains in Switzerland, and how to work with data that must not leave your institution.',
    published: GDPR_DATE,
    modified: GDPR_DATE,
    type: 'Article',
    sections: [
      {
        id: 'question',
        title: 'The first question: is it personal data?',
        html: '<p>The GDPR applies to personal data: information about a person who is identified or can be identified, directly or by combining it with other information. Much of deep learning involves none. Measurements from machines and sensors, images of materials, plants or products, simulation results, field trials and public benchmarks such as MNIST or CIFAR-10 say nothing about anyone. For such data the GDPR sets no rules on training; what remains is the question which provider you trust with it.</p><p>It is personal data as soon as rows or images can be linked to people: patient records, survey answers with a birth date and a postcode, photos with faces, texts with names, voice recordings, log files with user IDs. Then you need a legal basis and a defined purpose (Art. 5 and 6 GDPR), you keep to what that purpose needs, and every provider that processes the data on your behalf needs a contract under Art. 28. Health, genetic and biometric data are special categories with stricter rules (Art. 9).</p>',
      },
      {
        id: 'anonymise',
        title: 'Anonymised, pseudonymised or personal',
        html: `<div class="table" role="region" aria-labelledby="states-caption" tabindex="0"><table><caption id="states-caption">Three states of a dataset under the GDPR</caption><thead><tr><th scope="col">State</th><th scope="col">What it means</th><th scope="col">GDPR</th></tr></thead><tbody>
<tr><th scope="row">Personal</th><td>Names, e-mail addresses, IDs, or details that point to a person</td><td>Applies in full</td></tr>
<tr><th scope="row">Pseudonymised</th><td>Identifiers replaced by codes; the key still exists, possibly somewhere else</td><td>Still personal data, applies in full (Recital 26)</td></tr>
<tr><th scope="row">Anonymised</th><td>Nobody, you included, can link a row to a person with reasonable means</td><td>Does not apply (Recital 26)</td></tr>
</tbody></table></div><p>Anonymising a table takes more than deleting the name column:</p><ul class="checks"><li>Remove direct identifiers: names, addresses, e-mail addresses, phone numbers, ID and record numbers.</li><li>Coarsen what identifies people in combination: the year instead of the birth date, the region instead of the postcode, age groups instead of exact ages.</li><li>Drop free-text columns; they often contain names.</li><li>Look for rare combinations: one row with an unusual value can be enough to recognise someone.</li><li>In images, look for faces, name tags, text burned into the image and metadata such as GPS positions in the EXIF data.</li></ul><p>Do this on your own computer before you upload anything. If you are not sure the result is anonymous, treat it as personal data and ask your data protection officer.</p>`,
      },
      {
        id: 'brickdnn',
        title: 'Where brickdnn.com keeps what',
        html: `<p>When you train on brickdnn.com, your data is stored and processed here. The <a href="/privacy">privacy policy</a> (in German) has the full list with purposes and retention periods.</p><div class="table" role="region" aria-labelledby="where-caption" tabindex="0"><table><caption id="where-caption">Where brickdnn.com stores and processes data</caption><thead><tr><th scope="col">Data</th><th scope="col">Where</th><th scope="col">Provider</th></tr></thead><tbody>
<tr><th scope="row">Uploaded files, prepared datasets, trained weights, training results</th><td>Server in Geneva, Switzerland; the trainings run there too</td><td>Infomaniak</td></tr>
<tr><th scope="row">Accounts, networks, comments, the details of data and trainings (not the files)</th><td>Database on the same server in Geneva, Switzerland</td><td>Infomaniak</td></tr>
<tr><th scope="row">Technical error reports, without user name, e-mail address, account ID or uploaded files</th><td>EU region in Frankfurt, Germany</td><td>Sentry</td></tr>
<tr><th scope="row">E-mails about your account</th><td>EU</td><td>Brevo</td></tr>
<tr><th scope="row">Payment, only if you buy a plan</th><td>United Kingdom</td><td>Paddle</td></tr>
</tbody></table></div><p>The EU Commission recognises the data protection of Switzerland and of the United Kingdom as adequate (Art. 45 GDPR), so data may go there without further safeguards. Hosting, database, e-mail and error reports run under data processing agreements. The app sets no cookies and has no analytics or tracking, and your data is not sold.</p><p>One limit matters for this guide: the terms of use ask you not to upload personal data of third parties or confidential data to brickdnn.com, and the app reminds you when you upload. The hosted service is meant for non-personal and anonymised data. For everything else, read on.</p>`,
      },
      {
        id: 'personal',
        title: 'When the data has to stay personal',
        html: `<p>Brick DNN is still useful when the data itself must not leave your institution. Three ways, from the least effort to the most:</p><ol class="numbered"><li><strong>Design here, train at home.</strong> Build and check the network on brickdnn.com without uploading anything; that needs no account. Export the PyTorch script and train it on your institution’s computers with your own data loader; the script imports nothing from Brick DNN. For a network with one flat input and one flat output it already contains a training loop for CSV files (<code>python model.py --csv data.csv</code>).</li><li><strong>Work on a stand-in.</strong> Upload an anonymised extract or synthetic data with the same columns, prepare it in the Data tab and train on the server until model and settings are right. The preparation exports as a pandas script, which you then run on the real data where it is.</li><li><strong>Run your own Brick DNN.</strong> Brick DNN is open source under the Apache 2.0 licence. <code>docker compose up -d</code> starts it on a server of your institution, with a local SQLite database unless you configure another one, and error reports are off unless you set them up; for neural networks, build the image with PyTorch (<code>BRICKDNN_INSTALL_TORCH=true</code>). Data, trainings and accounts then stay wherever that server is. The <a href="${REPOSITORY}#quick-start">quick start</a> describes the setup.</li></ol>`,
      },
      {
        id: 'control',
        title: 'Export and delete',
        html: '<p>Whatever you upload, you can take back out and remove yourself:</p><ul class="checks"><li><strong>Download my data</strong> under <em>Account</em> gives a ZIP file with your networks and their history, data sources and preparations, datasets, training runs with their settings and metrics, and the uploaded files themselves (up to 200 MB in total; larger ones are listed). Trained weights download from each finished run, with the matching script.</li><li>Uploads, datasets and training runs can be deleted one by one at any time. Deleting the account removes everything it owns, files included.</li><li>Uploads, datasets and finished trainings nobody has used for 180 days are deleted automatically, and accounts after two years without use, with e-mail reminders beforehand.</li><li>Deleted data can remain in backups for a while, at most 5 weeks, until they are rotated out.</li></ul>',
      },
      {
        id: 'checklist',
        title: 'A checklist',
        html: '<ol class="numbered"><li>Decide whether the data is personal. If it is not, train on brickdnn.com; the files stay on the server in Switzerland.</li><li>If it is, try to anonymise it on your own computer before uploading, and check rare combinations and image metadata.</li><li>If it has to stay personal, design the model here and train the exported code, or your own Brick DNN instance, where the data is.</li><li>Record purpose and legal basis in your institution’s record of processing activities, and involve the data protection officer early.</li><li>Download what you want to keep, and delete datasets and runs you no longer need.</li></ol><p>This guide describes how Brick DNN works and what the GDPR asks in general. It is not legal advice for your project.</p>',
      },
    ],
    examples: ['tabular_regression', 'cnn_mnist', 'gradient_boosting_classifier'],
  },
  {
    id: 'activations',
    slug: 'activation-functions',
    title: 'Which activation function when: ReLU, GELU, Sigmoid, Softmax',
    description:
      'Activation functions in deep learning: what ReLU, LeakyReLU, GELU, SiLU, Tanh, Sigmoid and Softmax do, and which to use in hidden layers and at the output.',
    h1: 'Which activation function when: ReLU, GELU, Sigmoid, Softmax and the others',
    summary:
      'ReLU for most hidden layers, and at the output whatever the loss expects, which is usually nothing: the reasons, the exceptions and the classic mistake.',
    lead: 'In deep learning, the activation function of the hidden layers rarely decides whether a network works: ReLU is a sound default, and GELU or SiLU are worth a try in deeper networks. The output is different. There the choice follows from the task and the loss, and a wrong one costs accuracy without any error message.',
    published: BASICS_DATE,
    modified: BASICS_DATE,
    type: 'Article',
    sections: [
      {
        id: 'short',
        title: 'The short answer',
        html: '<ul class="checks"><li><strong>Hidden layers:</strong> ReLU. In Transformers GELU, in newer convolutional networks often SiLU. If many units of a ReLU network stay at zero, LeakyReLU or ELU.</li><li><strong>Output of a regression:</strong> no activation. The last layer is a Linear layer with one output per number you predict.</li><li><strong>Output of a classification:</strong> no activation either, as long as the loss is PyTorch’s <code>CrossEntropyLoss</code> or <code>BCEWithLogitsLoss</code>. Both apply the softmax or the sigmoid themselves.</li><li><strong>Sigmoid and Tanh in hidden layers:</strong> only in shallow networks, or where a layer needs bounded values. In deep stacks they slow the first layers down.</li></ul>',
      },
      {
        id: 'why',
        title: 'What an activation function is for',
        html: '<p>A Linear layer computes weighted sums. Two of them in a row are again only a weighted sum, so without something non-linear in between, a network of any depth can represent no more than a single Linear layer. The activation function is that something. It is applied to every value on its own, has no weights to learn (PReLU is the exception) and leaves the shape of the tensor as it is.</p><p>What matters most about it is its slope, because the gradient is multiplied by that slope on its way back through the network. A function that is flat over a wide range passes little gradient to the layers before it. Most of the differences below come down to this.</p>',
      },
      {
        id: 'hidden',
        title: 'Hidden layers: the candidates',
        html: `<div class="table" role="region" aria-labelledby="hidden-caption" tabindex="0"><table><caption id="hidden-caption">Activation functions for hidden layers</caption><thead><tr><th scope="col">Function</th><th scope="col">What it computes</th><th scope="col">Typical use</th><th scope="col">Weak point</th></tr></thead><tbody>
<tr><th scope="row">ReLU</th><td><code>max(0, x)</code>: negative values become 0</td><td>The default for MLPs and convolutional networks; cheap to compute</td><td>Units can end up at 0 for every input (see below)</td></tr>
<tr><th scope="row">LeakyReLU, PReLU</th><td>Like ReLU, but negative values keep a small slope: 0.01 in LeakyReLU by default, a learned one in PReLU</td><td>When ReLU units die</td><td>One more setting or parameter; otherwise rarely better than ReLU</td></tr>
<tr><th scope="row">ELU, SELU</th><td>Negative values bend smoothly towards a negative constant; SELU is an ELU with two fixed scaling factors</td><td>ELU as a smoother ReLU; SELU for plain fully connected networks without normalisation layers</td><td>SELU only keeps activations stable with a matching weight initialisation and its own variant of dropout</td></tr>
<tr><th scope="row">GELU</th><td><code>x</code> times the share of a standard normal distribution that lies below <code>x</code>: smooth, slightly negative just below 0</td><td>The standard in Transformers (BERT, the GPT models, Vision Transformers)</td><td>A little more expensive to compute</td></tr>
<tr><th scope="row">SiLU (Swish)</th><td><code>x · sigmoid(x)</code>, a curve much like GELU</td><td>Newer convolutional networks such as EfficientNet</td><td>The same</td></tr>
<tr><th scope="row">Tanh</th><td>Squashes to −1 … 1, centred on 0</td><td>Inside recurrent layers (RNN, LSTM, GRU); small networks</td><td>Flat for large inputs: vanishing gradients in deep stacks</td></tr>
<tr><th scope="row">Sigmoid</th><td>Squashes to 0 … 1</td><td>Gates inside LSTM and GRU; outputs that are probabilities</td><td>The same, and its slope is never above 0.25</td></tr>
</tbody></table></div><p>Mish and Softplus belong to the same smooth family as GELU and SiLU and are used less often. On tables and small image problems, the difference between ReLU, GELU and SiLU is usually small compared with what the learning rate or the size of the network changes, so tune those first.</p>`,
      },
      {
        id: 'dying-relu',
        title: 'Dying ReLU',
        html: '<p>A ReLU unit whose input is negative outputs 0 and has slope 0, so no gradient flows through it. If its weights are pushed so far that its input is negative for every training sample, which a learning rate that is too high can do in a single step, it receives no gradient from then on and stays at 0 for good. A few dead units do no harm. When a large part of a layer is dead, the network has lost that capacity, and the loss stops improving early.</p><p>To see it in Brick DNN, open <strong>Explain the model</strong> under a finished run and look at <em>Inside the network</em>: for every layer it shows the share of zeros in the output for one sample. A ReLU sets a good part of its values to 0 for any single sample; that is how it works. If nearly everything is 0, and stays 0 for other samples too, most of the layer is dead.</p><p>What helps:</p><ul class="checks"><li>a lower learning rate, which is the most common cure;</li><li>LeakyReLU, ELU or GELU instead of ReLU, which keep a slope for negative inputs;</li><li>a normalisation layer (BatchNorm or LayerNorm) before the activation, which keeps its inputs centred around 0.</li></ul>',
      },
      {
        id: 'vanishing-gradients',
        title: 'Vanishing gradients with Sigmoid and Tanh',
        html: '<p>The slope of the sigmoid is at most 0.25, at x = 0, and falls towards 0 on both sides. Backpropagation multiplies the gradient by one such slope per layer. After four sigmoid layers that is a factor of at most 0.25⁴, about 0.004, before the weights are counted in, and less in practice because few units sit exactly at x = 0. The first layers then receive almost no gradient and learn very slowly, while the last ones train normally.</p><p>Tanh is better: its slope reaches 1 at x = 0, and its output is centred on 0. But it is just as flat for large inputs, so units that saturate pass almost nothing back either. This is the main reason why ReLU replaced both in deep networks: its slope is exactly 1 for every positive input, however deep the stack.</p><p>Recurrent layers are the exception that has stayed. LSTM and GRU use sigmoid and tanh inside, as gates, and are built so that the gradient has a path around them; you do not add these activations yourself. If you want Sigmoid or Tanh between your own layers, keep the stack shallow, prefer Tanh, or put normalisation layers or skip connections around them.</p>',
      },
      {
        id: 'output',
        title: 'The output layer follows the task',
        html: `<p>The last layer is not a matter of taste. It has to produce what the loss expects, and the PyTorch losses used most expect raw scores (logits), not probabilities.</p><div class="table" role="region" aria-labelledby="output-caption" tabindex="0"><table><caption id="output-caption">Last layer, activation and loss per task, in PyTorch</caption><thead><tr><th scope="col">Task</th><th scope="col">Last layer</th><th scope="col">Activation after it</th><th scope="col">Loss</th></tr></thead><tbody>
<tr><th scope="row">Regression</th><td>Linear, one output per predicted number</td><td>None</td><td><code>MSELoss</code>, <code>L1Loss</code>, <code>HuberLoss</code></td></tr>
<tr><th scope="row">Two classes</th><td>Linear, 1 output</td><td>None</td><td><code>BCEWithLogitsLoss</code>, which applies the sigmoid</td></tr>
<tr><th scope="row">Several classes, one of them is right</th><td>Linear, one output per class (this also works for two)</td><td>None</td><td><code>CrossEntropyLoss</code>, which applies the softmax</td></tr>
<tr><th scope="row">Several labels can be right at once</th><td>Linear, one output per label</td><td>None</td><td><code>BCEWithLogitsLoss</code></td></tr>
<tr><th scope="row">A class for every pixel</th><td>Convolution, one channel per class</td><td>None</td><td><code>CrossEntropyLoss</code></td></tr>
<tr><th scope="row">A number in a fixed range, e.g. 0 … 1</th><td>Linear</td><td>Sigmoid (Tanh for −1 … 1)</td><td><code>MSELoss</code></td></tr>
</tbody></table></div><p>To get probabilities from a trained classifier, apply the function to its output when you predict, outside the network and the loss: <code>torch.softmax(logits, dim=1)</code> for classes, <code>torch.sigmoid(logits)</code> for a single score. The predicted class is the same either way: the one with the largest score.</p><p>Brick DNN trains every classification with one output per class and a loss on raw scores (the cross-entropy unless you choose another), also with two classes. A regression has one output and no activation after it; the target is standardised for the training, so a ReLU or a Sigmoid at the end would cut off part of its range.</p>`,
      },
      {
        id: 'softmax-mistake',
        title: 'The classic mistake: Softmax before CrossEntropyLoss',
        html: '<p><code>nn.CrossEntropyLoss</code> is a log-softmax followed by the negative log-likelihood. If the network already ends with <code>nn.Softmax</code>, the softmax is applied twice. Nothing crashes: the shapes fit, the loss goes down and the accuracy goes up. It just works worse than it should.</p><p>The loss now sees values between 0 and 1 instead of scores of any size, and the second softmax cannot turn those into a confident prediction. With ten classes, even a perfect output of 1 for the right class and 0 for all others gives the right class a probability of only e / (e + 9), about 0.23. The loss therefore never falls below about 1.46, when plain guessing gives 2.30. The gradients are small to match, and training is slow.</p><p>The same happens with <code>nn.Sigmoid</code> before <code>BCEWithLogitsLoss</code>. These pairs belong together:</p><ul class="checks"><li>raw scores with <code>CrossEntropyLoss</code> or <code>BCEWithLogitsLoss</code>, the usual choice and numerically the most stable one;</li><li><code>nn.LogSoftmax</code> with <code>NLLLoss</code>;</li><li><code>nn.Sigmoid</code> with <code>BCELoss</code>.</li></ul><p>In Brick DNN the loss of a classification always takes raw scores, so the network should end with its last Linear layer. If it ends with Softmax or Sigmoid instead, the training panel points it out before you start.</p>',
      },
      {
        id: 'architectures',
        title: 'What CNNs and Transformers typically use',
        html: '<p><strong>Convolutional networks.</strong> The classic block is a convolution followed by ReLU, often with a BatchNorm layer between the two. VGG and ResNet are built this way, and so are the <a href="/examples/cnn-mnist/">CNN</a>, the <a href="/examples/resnet-block/">ResNet block</a> and the <a href="/examples/unet/">U-Net</a> among the examples here. Later families kept the pattern and swapped the function: EfficientNet uses SiLU, ConvNeXt uses GELU.</p><p><strong>Transformers.</strong> The feed-forward part of every block has one activation. The original Transformer used ReLU; BERT, the GPT models and Vision Transformers use GELU. PyTorch’s <code>nn.TransformerEncoderLayer</code> defaults to ReLU, and that is what the <code>TransformerEncoder</code> layer of Brick DNN generates. The softmax inside the attention is part of the layer and nothing you add.</p><p><strong>Recurrent networks.</strong> LSTM and GRU bring their tanh and sigmoid with them. What follows them is usually a Linear layer without an activation in between, as in the <a href="/examples/lstm-sentiment/">LSTM example</a>.</p><p><strong>MLPs on tables.</strong> ReLU. If the network is small and the result disappointing, the activation is seldom the reason; comparing it with <a href="/guides/gradient-boosting-vs-mlp/">gradient boosting</a> is the better next step.</p>',
      },
      {
        id: 'table',
        title: 'A small decision table',
        html: `<div class="table" role="region" aria-labelledby="decision-caption" tabindex="0"><table><caption id="decision-caption">Which activation function for which situation</caption><thead><tr><th scope="col">Situation</th><th scope="col">Use</th></tr></thead><tbody>
<tr><th scope="row">Hidden layers of an MLP or a CNN, nothing special</th><td>ReLU</td></tr>
<tr><th scope="row">A Transformer, or a deep network where the last bit of accuracy counts</th><td>GELU, or SiLU</td></tr>
<tr><th scope="row">The loss stalls early and most outputs of a layer are 0</th><td>A lower learning rate first, then LeakyReLU or ELU</td></tr>
<tr><th scope="row">A deep fully connected network without normalisation layers</th><td>BatchNorm with ReLU; SELU only if you follow its rules</td></tr>
<tr><th scope="row">Hidden values that must stay within −1 … 1</th><td>Tanh, in a shallow stack</td></tr>
<tr><th scope="row">Output: a number</th><td>None</td></tr>
<tr><th scope="row">Output: classes, with <code>CrossEntropyLoss</code> or <code>BCEWithLogitsLoss</code></th><td>None</td></tr>
<tr><th scope="row">Output: a value between 0 and 1, with <code>MSELoss</code> or <code>BCELoss</code></th><td>Sigmoid</td></tr>
<tr><th scope="row">Probabilities when predicting</th><td>Softmax or sigmoid applied to the output, outside the network</td></tr>
</tbody></table></div>`,
      },
      {
        id: 'try',
        title: 'Try them in Brick DNN',
        html: '<p>The layer is called <code>Activation</code>. Its setting <strong>Function</strong> offers ReLU, LeakyReLU (with the slope 0.01), GELU, ELU, SELU, SiLU, Mish, Sigmoid, Tanh, Softplus, Softmax and LogSoftmax. PReLU is not among them; in the exported PyTorch code it is one line to change. A comparison takes a few minutes:</p>',
        steps: [
          {
            title: 'Open an example',
            html: 'Open the <a href="/app?example=mlp_mnist">MLP for MNIST in the editor</a>: 784 inputs, two hidden layers with 256 and 128 units, each followed by ReLU, and a last Linear layer with 10 outputs and no activation after it.',
          },
          {
            title: 'Train it once',
            html: 'Press <strong>Train</strong> and choose <strong>Built-in → MNIST</strong>. Set a <em>Seed</em> under <em>Advanced settings</em>, so that the next run differs in nothing but the activation, and train a few epochs. Training needs a free account.',
          },
          {
            title: 'Change the function',
            html: 'Select the two <code>Activation</code> layers one after the other and set <em>Function</em> to GELU, or to Sigmoid for a clearer difference. The network keeps its 235,146 parameters: activations have none.',
          },
          {
            title: 'Train again and compare',
            html: 'Start a second run with the same settings. The <strong>Runs</strong> tab of the training panel keeps both, each with the exact network that was trained, and the page <em>Runs</em> shows their curves in one chart. With two hidden layers the differences are small. They tend to grow with depth: add three or four Linear layers with Sigmoid and watch the first epochs.',
          },
        ],
      },
    ],
    examples: ['mlp_mnist', 'cnn_mnist', 'transformer_classifier'],
  },
  {
    id: 'parameters',
    slug: 'parameters-features-data-points',
    title: 'Parameters, features, data points: how large should a network be?',
    description:
      'How many parameters a deep-learning model has and where they come from, how they relate to features and training rows, and how to recognise overfitting.',
    h1: 'Parameters, features and data points: how large should a neural network be?',
    summary:
      'Where the parameters of a network come from, what they have to do with the number of features and rows, and what to do when the curves show overfitting.',
    lead: 'A network has 2,817 parameters and the table 300 rows: is that too many? No formula answers this, and the rules of thumb that claim to are of limited use in deep learning. What helps is knowing where the parameters come from, reading the curves and having a few remedies at hand.',
    published: BASICS_DATE,
    modified: BASICS_DATE,
    type: 'Article',
    sections: [
      {
        id: 'terms',
        title: 'Three numbers that are easy to mix up',
        html: '<ul class="checks"><li><strong>Features</strong> are what the model gets per sample: the columns of a table without the target, or the pixels of an image times its channels. A table with 8 columns to predict from has 8 features; a grayscale image of 28 × 28 pixels has 784.</li><li><strong>Data points</strong> are the samples: rows, images, texts. For learning, the training rows count, after validation and test rows have been set aside.</li><li><strong>Parameters</strong> are the weights and biases that the training adjusts. Their number follows from the layers you chose.</li></ul><p>The features fix the size of the first layer, and the task fixes the last one. Everything in between is your decision, and with it the number of parameters. Learning rate, batch size and the like are <em>hyper</em>parameters: settings of the training, not part of the model.</p>',
      },
      {
        id: 'count',
        title: 'Where the parameters come from',
        html: `<p>Brick DNN shows the count of every layer. It is worth being able to check it by hand, because the formulas show which layers are expensive.</p><div class="table" role="region" aria-labelledby="count-caption" tabindex="0"><table><caption id="count-caption">Parameters of common layers</caption><thead><tr><th scope="col">Layer</th><th scope="col">Parameters</th><th scope="col">Example</th></tr></thead><tbody>
<tr><th scope="row">Linear</th><td>inputs × outputs, plus one bias per output</td><td>8 → 64: 8 · 64 + 64 = 576</td></tr>
<tr><th scope="row">Conv2d</th><td>filters × input channels × kernel height × kernel width, plus one bias per filter</td><td>32 filters of 3 × 3 on 1 channel: 32 · 1 · 9 + 32 = 320</td></tr>
<tr><th scope="row">Embedding</th><td>vocabulary × embedding size</td><td>10,000 tokens × 128 = 1,280,000</td></tr>
<tr><th scope="row">BatchNorm</th><td>2 per channel: a scale and a shift</td><td>64 channels: 128</td></tr>
<tr><th scope="row">Activation, pooling, Dropout, Flatten</th><td>0</td><td>–</td></tr>
</tbody></table></div><p>Three things follow:</p><ul class="checks"><li><strong>A Linear layer costs the product of its two sides.</strong> The <a href="/examples/tabular-regression/">tabular example</a> has 8 features and layers of 64, 32 and 1 units: 576 + 2,080 + 33 parameters, plus 128 in its BatchNorm layer, 2,817 in all. With hidden layers of 128 and 64 units it would have 9,729.</li><li><strong>A convolution does not care how large the image is.</strong> Its count depends on the channels and the kernel only. The expensive layer of a small CNN is the first Linear layer after <code>Flatten</code>: in the <a href="/examples/cnn-mnist/">CNN example</a>, 64 channels of 7 × 7 values are 3,136 inputs, and the Linear layer from there to 128 units has 401,536 parameters, 95 % of the 421,642 of the whole network.</li><li><strong>An Embedding grows with the vocabulary.</strong> In the <a href="/examples/transformer-classifier/">Transformer example</a>, the table of 10,000 × 128 values holds 1,280,000 of the 1,611,268 parameters, and each of its rows only learns from the samples in which its token occurs.</li></ul>`,
      },
      {
        id: 'relate',
        title: 'How the three relate',
        html: '<p><strong>Features and parameters.</strong> Every feature adds one weight per unit of the first layer. With 784 pixels and 256 units, the first layer of the <a href="/examples/mlp-mnist/">MLP example</a> alone has 200,960 of its 235,146 parameters. More features also mean more ways to find a pattern in the training rows that is not there in new data, so a column without information is not free.</p><p><strong>Rows and parameters.</strong> Every training row is one piece of evidence about what the parameters should be. A model with far more adjustable numbers than evidence can reproduce the training rows exactly, noise included, in many different ways, and nothing in those rows says which of the ways is right for new data. That is overfitting. With far fewer parameters than the task needs, the model cannot even reproduce the training rows. That is underfitting.</p><p><strong>Rows and features.</strong> The more features there are, the more rows it takes to cover their combinations. Ten yes/no features already allow 1,024 different rows.</p>',
      },
      {
        id: 'rules',
        title: 'Rules of thumb, and where they end',
        html: '<p>Statistics has rules such as “ten observations per coefficient”. They come from linear and logistic regression, where every parameter is a coefficient with a meaning of its own, and there they are sensible.</p><p>Neural networks break them routinely. The <a href="/examples/cnn-mnist/">CNN example</a> has 421,642 parameters and learns from 48,000 MNIST images, almost nine parameters per image, and still gets about 99 % of the 10,000 test images right. Networks with more parameters than training samples can generalise well when there is enough data for the difficulty of the task, when the architecture fits the data (a convolution reuses the same few weights at every position) and when the training is regularised: by dropout, by weight decay or simply by stopping in time. A ratio of parameters to rows therefore cannot tell you that a network is too large. It can tell you where to look more closely.</p><p>What can be said without false precision:</p><ul class="checks"><li><strong>A table with a few hundred rows:</strong> take a small network, one or two hidden layers of 8 to 32 units, or a classical model. Gradient boosting, a Gaussian process or a support vector machine is often the better choice here and needs less tuning; see <a href="/guides/gradient-boosting-vs-mlp/">gradient boosting vs. MLP</a>.</li><li><strong>A table with thousands of rows or more:</strong> an MLP with a few thousand to some tens of thousands of parameters is a reasonable start. Make it larger only as long as the validation result improves.</li><li><strong>Images and text:</strong> the parameter count says little. How many samples there are per class, how varied they are and how well the architecture fits matter more.</li><li><strong>In every case:</strong> hold back validation rows, and believe the curves rather than the ratio.</li></ul><p>Brick DNN uses the ratio in just this way. When a network has more than ten times as many parameters as its table has training rows, the training panel shows a short tip. It is a reason to look at the curves and to try a baseline, not an error.</p>',
      },
      {
        id: 'curves',
        title: 'What the curves show',
        html: `<p>After every epoch Brick DNN measures the loss on the training rows and on the validation rows, which the network never learns from. The two curves together say more than any count.</p><div class="table" role="region" aria-labelledby="curves-caption" tabindex="0"><table><caption id="curves-caption">Reading the training and the validation loss</caption><thead><tr><th scope="col">What you see</th><th scope="col">What it means</th><th scope="col">First thing to try</th></tr></thead><tbody>
<tr><th scope="row">Both fall, and are still falling at the end</th><td>The training is not finished</td><td>More epochs</td></tr>
<tr><th scope="row">The training loss keeps falling, the validation loss flattens or rises again</th><td>Overfitting: the network is memorising the training rows</td><td>Keep the best epoch, then less capacity or more regularisation</td></tr>
<tr><th scope="row">Both stay high and close together</th><td>Underfitting, or a learning rate that does not suit</td><td>A larger network, another learning rate, better features</td></tr>
<tr><th scope="row">The validation loss jumps from epoch to epoch</th><td>Too few validation rows, or a learning rate that is too high</td><td>More validation rows, a lower learning rate</td></tr>
</tbody></table></div><p>A gap between the two curves is normal, since the network has seen the training rows. What matters is the direction of the validation curve. With few rows, read it with care: with 40 validation rows, a single row is 2.5 percentage points of accuracy.</p>`,
        image: {
          id: 'training-curves',
          alt: 'Loss and accuracy per epoch of a small CNN on MNIST: the training loss keeps falling, while the validation loss stays about the same from the third epoch on.',
          caption:
            'From epoch 3 on, the training loss still falls and the validation loss no longer does: more epochs would not help this network.',
        },
      },
      {
        id: 'overfitting',
        title: 'What to do about overfitting',
        html: '<p>Roughly in the order of effort:</p><ol class="numbered"><li><strong>Keep the best epoch and stop early.</strong> Brick DNN keeps the weights of the epoch with the lowest validation loss unless you switch that off, so a run that overfits late does no harm. <em>Early stopping</em> in the training settings ends a run when the validation loss has not improved for the number of epochs you give.</li><li><strong>Less capacity.</strong> Remove a layer or halve the hidden layers and train again. If the validation result stays the same, the smaller network is the better one.</li><li><strong>Dropout.</strong> A <code>Dropout</code> layer after a hidden activation sets a random share of the values to 0 in every training step, which keeps units from relying on each other. Shares between 0.1 and 0.5 are common.</li><li><strong>Weight decay.</strong> It is under <em>Advanced settings</em> and pulls the weights a little towards 0 in every step. Try values between 0.0001 and 0.01, preferably with the AdamW optimizer, or let a hyperparameter search pick one.</li><li><strong>More data.</strong> More rows help more than anything above, when they can be had. Fewer and better features help too: drop columns that cannot matter.</li><li><strong>A classical baseline.</strong> Train a <a href="/examples/gradient-boosting-classifier/">gradient boosting model</a> on the same split. If it is better on a small table, use it.</li></ol><p>Underfitting is the easier case: more units or layers, more epochs, a look at the learning rate, and less dropout or weight decay if you had added any.</p>',
      },
      {
        id: 'where',
        title: 'Where Brick DNN shows the numbers',
        html: '<ul class="checks"><li><strong>Parameters:</strong> every layer on the canvas shows its own count. With no layer selected, the inspector shows the <em>Trainable parameters</em> of the whole network, and a run shows them in the tile <em>Parameters</em>.</li><li><strong>Features and rows:</strong> the list of your datasets in the training panel names both for each dataset, and the check repeats them once you choose one, for example “Classification into 2 classes from 18 features (900 rows)”. For a dataset from the Data tab it also says how many rows are in training, validation and test.</li><li><strong>Curves:</strong> <em>Training curves</em> in the view of a run, with the numbers of every epoch under <em>Values per epoch</em> and the epoch whose weights were kept.</li></ul>',
        image: {
          id: 'training-setup',
          alt: 'The training panel with a prepared dataset of 900 rows and 18 features, split into 630 training, 135 validation and 135 test rows, and the check “Ready to train”.',
          caption: 'The dataset and the check name the rows, the features and the split.',
        },
      },
    ],
    examples: ['tabular_regression', 'mlp_mnist', 'gradient_boosting_classifier'],
  },
];

export const guidesEn = { index, guides };
