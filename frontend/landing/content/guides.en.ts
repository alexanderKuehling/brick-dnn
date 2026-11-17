/**
 * The guides in English. Everything they say must be true of the app (docs/training.md,
 * docs/data-preparation.md, docs/explain.md); the screenshots are those of images/.
 */
import type { Guide, GuideIndex } from './types.ts';

const REPOSITORY = 'https://github.com/alexanderKuehling/brick-dnn';
const DOCS = `${REPOSITORY}/blob/main/docs`;
const DATE = '2026-11-16';
const GDPR_DATE = '2026-11-17';

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
];

export const guidesEn = { index, guides };
