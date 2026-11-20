/**
 * The copy of the example pages in English. The facts on the pages (layers, shapes, parameter
 * counts, code) come from examples/examples.json; what is said here must match them
 * (landing.test.ts checks the numbers that appear in the text).
 */
import type { ExampleCopy, ExampleGallery } from './types.ts';

const gallery: ExampleGallery = {
  title: 'Deep-learning examples in PyTorch and scikit-learn – Brick DNN',
  description:
    'Deep-learning examples: MLP, CNN, ResNet block, U-Net, LSTM, Transformer, plus gradient boosting, SVM, Gaussian processes and ensembles, with diagram and code.',
  h1: 'Example networks with diagram, shapes and code',
  intro:
    'Each example below is a network that ships with Brick DNN. Its page shows how the layers are connected, the output shape and parameter count of every layer, and the complete script the editor generates, in PyTorch for neural networks and in scikit-learn for classical models. Open any of them in the editor to change it, check it and train it.',
  groups: [
    {
      title: 'Images',
      intro:
        'From a multi-layer perceptron on flattened digits to a U-Net that labels every pixel.',
      ids: ['mlp_mnist', 'cnn_mnist', 'resnet_block', 'unet', 'unet_classifier'],
    },
    {
      title: 'Text',
      intro: 'Sequences of token ids, classified by a recurrent network or a Transformer encoder.',
      ids: ['lstm_sentiment', 'transformer_classifier'],
    },
    {
      title: 'Tables and mixed inputs',
      intro:
        'Neural networks for rows of numbers, a network with two inputs, and a hybrid of an MLP and gradient boosting.',
      ids: ['tabular_regression', 'multi_input', 'hybrid_mlp_gradient_boosting'],
    },
    {
      title: 'Classical models',
      intro:
        'Models for tabular data, built from bricks like the networks: gradient boosting with scikit-learn or LightGBM, a support vector machine and Gaussian processes with kernels.',
      ids: [
        'gradient_boosting_classifier',
        'lightgbm_classifier',
        'svm_classifier',
        'gp_regression',
        'periodic_gp',
      ],
    },
    {
      title: 'Ensembles',
      intro: 'Several classical models combined by voting or by stacking.',
      ids: [
        'voting_ensemble_classifier',
        'stacking_ensemble_regression',
        'boosting_stacking_ensemble',
      ],
    },
  ],
  note: 'The diagrams, shapes, parameter counts and code on these pages are produced from the examples themselves, with the same shape inference and code generator as the editor.',
};

const examples: ExampleCopy[] = [
  {
    id: 'mlp_mnist',
    name: 'MLP for MNIST',
    title: 'MLP for MNIST in PyTorch – multi-layer perceptron example',
    description:
      'A multi-layer perceptron for MNIST digits in PyTorch: 784 inputs, two hidden layers, 235,146 parameters, with diagram, layer shapes and the generated code.',
    h1: 'MLP for MNIST in PyTorch',
    task: 'classification',
    summary: 'A multi-layer perceptron: 784 pixel values, two hidden layers, ten class scores.',
    lead: 'The classic first network for handwritten digits: each 28×28 image is flattened into 784 numbers, two fully connected hidden layers with ReLU learn from them, and a last layer gives one score for each of the ten digits.',
    what: [
      'The <code>Input</code> takes a flat vector of 784 values, one per pixel. <code>Linear</code> layers connect every value with every unit: 784 inputs to 256 units need 784 × 256 weights plus 256 biases, 200,960 parameters, almost all of the network’s 235,146. A ReLU after each hidden layer makes the network non-linear, and <code>Dropout</code> with p = 0.2 switches off a fifth of the first hidden layer’s units at random during training, which reduces overfitting.',
      'The last <code>Linear</code> layer outputs ten raw scores (logits), one per digit. There is no Softmax at the end on purpose: the cross-entropy loss that training uses expects raw scores and applies the softmax itself.',
    ],
    when: [
      'To learn how a dense network, its parameter count and its training behave before moving on to convolutions.',
      'As a baseline for image data: a CNN such as the <a href="/examples/cnn-mnist/">small CNN for MNIST</a> should beat it with fewer parameters, because it uses the neighbourhood of pixels.',
      'For any input that is already a flat vector of features: change the shape of the Input and the size of the last layer.',
    ],
    train:
      'Press <strong>Train</strong> and choose <strong>MNIST</strong>: the built-in dataset (70,000 images, downloaded once on the server) fits this Input as it is, because MNIST is accepted as <code>784</code> flat values as well as <code>1, 28, 28</code>. Brick DNN keeps 20 % of the training images for validation, tests on the official 10,000 test images and shows loss and accuracy after every epoch. Because the network takes one flat vector and returns one, the exported script also contains a training loop for CSV files.',
    related: ['cnn_mnist', 'tabular_regression', 'multi_input'],
  },
  {
    id: 'cnn_mnist',
    name: 'Small CNN for MNIST',
    title: 'CNN for MNIST in PyTorch – small convolutional network example',
    description:
      'A small convolutional neural network for MNIST in PyTorch, a classic first deep-learning model: two Conv2d blocks with max pooling and a dense head, with code.',
    h1: 'CNN for MNIST in PyTorch',
    task: 'classification',
    summary: 'Two convolution blocks and a classifier head: about 99 % test accuracy on MNIST.',
    lead: 'A small convolutional network that reaches about 99 % test accuracy on handwritten digits: two blocks of convolution, ReLU and max pooling find strokes and shapes, and a dense head turns them into ten class scores.',
    what: [
      'Each <code>Conv2d</code> slides 3×3 filters over the image. The first one has 32 filters, so the 1 × 28 × 28 image becomes 32 × 28 × 28 feature maps; <code>padding=same</code> keeps height and width. <code>MaxPool2d</code> with a 2×2 window halves them, to 32 × 14 × 14. The second block repeats this with 64 filters and ends at 64 × 7 × 7.',
      '<code>Flatten</code> turns these 3,136 values into a vector for the head: a <code>Linear</code> layer with 128 units, ReLU, <code>Dropout</code> with p = 0.25 and the last <code>Linear</code> layer with ten outputs. The first dense layer holds 401,536 of the 421,642 parameters, while the convolutions need only 18,816 together, because a filter is shared across the whole image.',
    ],
    when: [
      'For small grayscale or colour images with one label each: digits, clothing (Fashion-MNIST), symbols, small product photos.',
      'As the standard first CNN in a course: it shows how convolution and pooling change the shape, and how few parameters a convolution has compared with a dense layer.',
      'As a starting point for your own images: set the Input to <code>3, 64, 64</code> for colour images and add a third block if the images are larger.',
    ],
    train:
      'Choose <strong>MNIST</strong> or <strong>Fashion-MNIST</strong> in the training panel; both fit the Input <code>1, 28, 28</code> and have ten classes. A folder of your own images (a ZIP file with one folder per class) also works: images are converted to the Input’s channels and resized to its height and width. The run view shows loss and accuracy per epoch, then the confusion matrix and per-class precision and recall on the test images.',
    related: ['mlp_mnist', 'resnet_block', 'unet_classifier'],
  },
  {
    id: 'resnet_block',
    name: 'Residual CNN for CIFAR-10',
    title: 'ResNet block for CIFAR-10 in PyTorch – residual CNN example',
    description:
      'A residual block with batch normalization and a skip connection (Add) for CIFAR-10 colour images, in PyTorch: 76,490 parameters, diagram, shapes and code.',
    h1: 'ResNet block for CIFAR-10 in PyTorch',
    task: 'classification',
    summary:
      'A ResNet-style block with a skip connection merged by Add, then global average pooling.',
    lead: 'The building block of ResNet in a small network for 32×32 colour photos: two convolutions learn a correction that an <code>Add</code> layer puts on top of their input, so the signal and its gradient can also take the short way around.',
    what: [
      'A stem of <code>Conv2d</code> (64 filters of 3×3, without bias), <code>BatchNorm2d</code> and ReLU turns the 3 × 32 × 32 image into 64 × 32 × 32 feature maps. The residual block then runs two more convolutions with batch normalization. The convolutions have no bias, because the batch normalization right after them has its own shift.',
      '<code>Add</code> sums the block’s output and the stem’s output element by element, which is why both must have the same shape, 64 × 32 × 32; Brick DNN checks that while you build. After a ReLU, <code>AdaptiveAvgPool2d</code> averages every feature map to one value (64 × 1 × 1), and a <code>Linear</code> layer of 650 parameters gives the ten class scores. Global average pooling instead of a large dense layer keeps the whole network at 76,490 parameters.',
    ],
    when: [
      'When a plain stack of convolutions stops improving as you add layers: skip connections are what lets deep networks train.',
      'To learn how a merge layer works: try connecting tensors of different shapes to <code>Add</code> and read the message.',
      'As a base for a deeper network: copy the block, connect several in a row, and add a convolution with stride 2 between them to halve the resolution.',
    ],
    train:
      'Choose <strong>CIFAR-10</strong> (60,000 colour photos in ten classes, downloaded once on the server); it fits the Input <code>3, 32, 32</code> as it is. Batch normalization behaves differently during training and evaluation, which the exported module handles through PyTorch’s <code>model.train()</code> and <code>model.eval()</code>.',
    related: ['cnn_mnist', 'unet', 'unet_classifier'],
  },
  {
    id: 'unet',
    name: 'U-Net for segmentation',
    title: 'U-Net for image segmentation in PyTorch – example code',
    description:
      'A U-Net, a common deep-learning model for semantic segmentation, in PyTorch: encoder, decoder with ConvTranspose2d, skip connections, a class score per pixel.',
    h1: 'U-Net for image segmentation in PyTorch',
    task: 'segmentation',
    summary:
      'Encoder, decoder and skip connections: a class for every pixel, trained on the built-in Shapes data.',
    lead: 'A U-Net gives every pixel of an image a class. Its encoder halves the image three times to understand what is in it, its decoder doubles it back to the full size, and skip connections hand the fine details of each level across to the decoder.',
    what: [
      'The encoder has three levels of two 3×3 convolutions with ReLU, with 32, 64 and 128 filters, each followed by <code>MaxPool2d</code>: the 3 × 64 × 64 image becomes 128 × 8 × 8. A bottleneck of two convolutions with 256 filters works on this coarse view, where every value sees a large part of the image.',
      'Each decoder level starts with <code>ConvTranspose2d</code>, a learned upsampling that doubles height and width. <code>Concat</code> then joins the result with the encoder’s output of the same size along the channels (for example 128 + 128 channels at 16 × 16), so the decoder gets back the details that pooling threw away. A last 1×1 convolution gives four channels, one score per class, for each of the 64 × 64 pixels.',
      'All convolutions use <code>padding=same</code>, so only pooling and upsampling change the size. Height and width of the Input must therefore be divisible by 8; Brick DNN checks this and says what to change.',
    ],
    when: [
      'For semantic segmentation: a class for every pixel, such as tissue in microscopy images, roads and buildings in aerial photos, defects on a surface or the background behind an object.',
      'When there are few training images: U-Net was designed for small medical datasets, and the skip connections make it learn sharp outlines from a few hundred masks.',
      'For two classes (foreground and background) set the last convolution to one channel; for more classes, one channel per class.',
    ],
    train:
      'Press <strong>Train</strong>, choose <strong>Built-in → Shapes</strong> and start: the generated dataset (400 training and 100 test images of squares, circles and triangles, with a mask for each) fits the network as it is, and ten epochs reach about 80 % mean IoU on a CPU. Your own data is a ZIP file with the folders <code>images/</code> and <code>masks/</code>. Every epoch reports mean IoU, Dice and pixel accuracy, and the run shows predicted masks next to the true ones. The <a href="/guides/image-segmentation-u-net/">segmentation guide</a> goes through it step by step.',
    related: ['unet_classifier', 'resnet_block', 'cnn_mnist'],
  },
  {
    id: 'unet_classifier',
    name: 'U-Net classifier',
    title: 'U-Net as a classifier for Fashion-MNIST in PyTorch – example',
    description:
      'A U-Net turned into an image classifier: encoder, decoder and skip connections, then global pooling and a Linear layer for Fashion-MNIST, in PyTorch.',
    h1: 'U-Net as an image classifier in PyTorch',
    task: 'classification',
    summary:
      'The U-Net architecture with a classification head: global pooling and a Linear layer give ten class scores.',
    lead: 'The same encoder, decoder and skip connections as the U-Net for segmentation, but ending in one answer per image: global average pooling and a <code>Linear</code> layer turn the full-size feature maps into ten class scores for Fashion-MNIST.',
    what: [
      'The 28×28 images of Fashion-MNIST are resized to 32×32 on the way in, so that three halvings fit (32 → 16 → 8 → 4). The filters are half those of the segmentation example, 16, 32 and 64 with a bottleneck of 128, so that the network trains on a CPU server.',
      'After the decoder, <code>AdaptiveAvgPool2d</code> averages each of the 16 feature maps of 32 × 32 to one value, <code>Flatten</code> makes a vector of 16, and the last <code>Linear</code> layer, with only 170 parameters, gives the ten class scores. The whole network has 481,898 parameters.',
    ],
    when: [
      'To see how a segmentation architecture becomes a classifier, and that the head is the only part that differs.',
      'Not as the best choice for this task: for small images with one label, the <a href="/examples/cnn-mnist/">small CNN</a> is smaller and faster.',
      'As a starting point for tasks that need both: the same encoder and decoder can feed a mask and a class.',
    ],
    train:
      'Choose <strong>Fashion-MNIST</strong>: its 28×28 grayscale images are resized to the Input <code>1, 32, 32</code> automatically. The run reports accuracy per epoch and the confusion matrix of the ten clothing classes at the end, which shows which ones the network confuses (shirts and T-shirts, typically).',
    related: ['unet', 'cnn_mnist', 'resnet_block'],
  },
  {
    id: 'lstm_sentiment',
    name: 'LSTM text classifier',
    title: 'LSTM for sentiment analysis in PyTorch – text classifier example',
    description:
      'Deep learning for text: a bidirectional LSTM classifier in PyTorch with token ids, embeddings and a Linear layer for positive or negative sentiment, with code.',
    h1: 'LSTM for sentiment analysis in PyTorch',
    task: 'sequence',
    summary: 'Token ids → embeddings → bidirectional LSTM → positive or negative.',
    lead: 'A recurrent network that reads a text as a sequence of 128 token ids and decides whether it is positive or negative: an embedding gives every token a vector, a bidirectional LSTM reads the sequence forwards and backwards, and a <code>Linear</code> layer makes the decision.',
    what: [
      'The <code>Input</code> holds 128 whole numbers (data type <code>long</code>), the ids of the tokens of a text from a vocabulary of 20,000. <code>Embedding</code> turns each id into a vector of 128 numbers; with 20,000 × 128 = 2,560,000 weights it is by far the largest part of the network’s 2,824,706 parameters.',
      'The <code>LSTM</code> with 128 hidden units runs over the 128 vectors in both directions. Only its last hidden states are passed on (<code>return_sequences</code> off), and those of the two directions are joined: a vector of 256 numbers that summarises the text. After <code>Dropout</code> with p = 0.3, a <code>Linear</code> layer gives two scores, negative and positive.',
    ],
    when: [
      'For short texts with one label: reviews, support tickets, short messages, where the word order matters.',
      'To learn how sequence models handle shapes: the Embedding adds a dimension, the LSTM removes it again.',
      'For long texts or large datasets a <a href="/examples/transformer-classifier/">Transformer encoder</a> often works better; pretrained language models better still.',
    ],
    train:
      'In Brick DNN, text models train on demo data so far: random token sequences with a hidden rule, which shows that the network learns. For real texts, export the script: it contains the <code>nn.Module</code> and a check with a random batch of token ids; tokenise your texts into ids up to 20,000 and padded to 128, and train it with your own loop.',
    related: ['transformer_classifier', 'multi_input', 'mlp_mnist'],
  },
  {
    id: 'transformer_classifier',
    name: 'Transformer text classifier',
    title: 'Transformer text classifier in PyTorch – encoder example code',
    description:
      'A small Transformer encoder for text classification in PyTorch: embeddings, learned positional embeddings, two encoder layers and mean pooling, with code.',
    h1: 'Transformer text classifier in PyTorch',
    task: 'sequence',
    summary:
      'A small Transformer encoder with learned positional embeddings and mean pooling, four classes.',
    lead: 'A compact Transformer for sorting texts into four classes: token embeddings plus learned positions go through two encoder layers with self-attention, and the mean over all positions feeds the classifier.',
    what: [
      'The <code>Input</code> takes 64 token ids. <code>Embedding</code> maps each to 128 numbers (10,000 × 128 = 1,280,000 parameters), and <code>PositionalEmbedding</code> adds a learned vector for each position, because self-attention by itself does not know the order of the tokens.',
      'The <code>TransformerEncoder</code> has two layers, each with self-attention of four heads and a feed-forward part of 256 units: 264,960 parameters together. Every position can look at every other one in each layer. <code>SequencePool</code> takes the mean over the 64 positions, <code>LayerNorm</code> normalises the resulting vector of 128, and a <code>Linear</code> layer gives the four class scores. The network has 1,611,268 parameters.',
    ],
    when: [
      'For topic or intent classification of short and medium texts, where words far apart depend on each other.',
      'To understand the parts of a Transformer encoder with real shapes, before working with large pretrained models.',
      'Raise <code>max_len</code> of the positional embedding and the Input length for longer texts; attention costs grow with the square of the length.',
    ],
    train:
      'Text models train on demo data in Brick DNN so far, which checks that the network learns. The exported PyTorch module is ready for your own training on tokenised texts: 64 token ids per text from a vocabulary of 10,000.',
    related: ['lstm_sentiment', 'multi_input', 'mlp_mnist'],
  },
  {
    id: 'tabular_regression',
    name: 'Tabular regression',
    title: 'Neural network regression on tabular data in PyTorch – MLP example',
    description:
      'Deep learning on tables: a small MLP in PyTorch predicts a number from 8 numeric features, with batch normalization and a training loop for CSV files.',
    h1: 'Neural network regression on tabular data in PyTorch',
    task: 'regression',
    summary: 'Predicts one number from 8 numeric features; the script trains on a CSV file.',
    lead: 'A small multi-layer perceptron for tables: eight numeric columns go in, one number comes out, for example a price, a yield or a temperature. The generated script trains it on a CSV file by itself.',
    what: [
      'Two hidden <code>Linear</code> layers with 64 and 32 units and ReLU learn the relationship between the eight features and the target. <code>BatchNorm1d</code> after the first hidden layer normalises its outputs over each batch, which makes training less sensitive to the learning rate. The last layer has a single output and no activation, so it can predict any number. The network has 2,817 parameters.',
      'For a regression, Brick DNN standardises the features and the target (mean 0, standard deviation 1, from the training rows) and trains with the mean squared error. The results are also reported as RMSE in the units of the target and as R².',
    ],
    when: [
      'For regression on tables with a few to a few hundred numeric columns, once the data is clean (no missing values, categories encoded).',
      'To compare with gradient boosting, which is often the stronger baseline on tables: see the <a href="/guides/gradient-boosting-vs-mlp/">guide on gradient boosting vs. MLP</a>.',
      'For more features, set the Input to their number; for several targets, give the last layer one output per target.',
    ],
    train:
      'Upload a CSV file whose last column is the target and whose other columns are numeric features, or prepare a table in the <strong>Data</strong> tab (missing values, categories, dates) and press <strong>Create training dataset</strong>; the dataset then appears in the training panel. The Input follows the number of features of the dataset. Because the network takes one flat vector and returns one, the exported script also contains a training loop: <code>python tabular_regression.py --csv data.csv</code>.',
    related: ['gp_regression', 'stacking_ensemble_regression', 'hybrid_mlp_gradient_boosting'],
  },
  {
    id: 'multi_input',
    name: 'Multi-input model',
    title: 'Multi-input neural network in PyTorch – image + metadata example',
    description:
      'A neural network with two inputs in PyTorch: an image branch and a metadata branch joined by Concat, for example to predict a price, with diagram and code.',
    h1: 'Multi-input neural network in PyTorch',
    task: 'regression',
    summary: 'An image and a vector of metadata, combined with Concat, predict one number.',
    lead: 'Two kinds of data in one network: a 28×28 image goes through one branch, ten numbers of metadata through another, and <code>Concat</code> joins what both learned before a last layer predicts a single value, for example a price from a photo and the attributes of an item.',
    what: [
      'The image branch flattens the 1 × 28 × 28 image and reduces it to 64 features with a <code>Linear</code> layer and ReLU (50,240 parameters). The metadata branch maps ten values to 16 features (176 parameters). <code>Concat</code> joins the two vectors to one of 80 values, and the head, a <code>Linear</code> layer with one output, predicts the target.',
      'The generated <code>forward</code> method takes two tensors, one per Input layer: <code>x_1</code> the image, <code>x_2</code> the metadata. Swap the image branch for convolutions (as in the <a href="/examples/cnn-mnist/">small CNN</a>) when the images matter more.',
    ],
    when: [
      'When one example has data of different kinds: a photo and a table row, a signal and the patient’s age, a text and its metadata.',
      'To learn how branches and merges work in a graph, and how the code of a network with several inputs looks.',
      'Instead of two separate models whose predictions are averaged: the network learns how the inputs interact.',
    ],
    train:
      'Networks with several inputs train on demo data in Brick DNN so far, which checks that the network learns from both inputs. The exported module takes both tensors in <code>forward(x_1, x_2)</code>, ready for your own data loader.',
    related: ['mlp_mnist', 'tabular_regression', 'cnn_mnist'],
  },
  {
    id: 'gp_regression',
    name: 'Gaussian process regression',
    title: 'Gaussian process regression in scikit-learn – RBF kernel example',
    description:
      'Gaussian process regression with a scaled RBF kernel plus white noise in scikit-learn: predictions with uncertainty from 8 features, kernel built from bricks.',
    h1: 'Gaussian process regression in scikit-learn',
    task: 'regression',
    summary: 'Predicts a number with its uncertainty; the kernel is a scaled RBF plus white noise.',
    lead: 'A Gaussian process predicts a number and says how sure it is. Its kernel, here a scaled RBF plus white noise, is built from bricks on the canvas, and the fit tunes the kernel’s hyperparameters to the data.',
    what: [
      'The kernel says how similar two rows are, and so how much one row’s target says about another’s. <code>ConstantKernel</code> × <code>RBF</code> is a smooth similarity that falls off with the distance between rows, scaled by a constant; <code>WhiteKernel</code> adds noise, because measurements are never exact. <code>KernelProduct</code> and <code>KernelSum</code> combine them into <code>ConstantKernel(constant_value=1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)</code>.',
      'These three values are starting points: fitting maximises the marginal likelihood and so finds the scale, the length scale and the noise level that fit the data. The model has no trainable weights in the sense of a network; Brick DNN counts the three kernel hyperparameters instead. The exported script is a scikit-learn <code>Pipeline</code> of a <code>StandardScaler</code> and the <code>GaussianProcessRegressor</code>.',
    ],
    when: [
      'For small to medium tables (up to a few thousand rows) where you need an uncertainty with each prediction, for example in experiments, calibration or Bayesian optimisation.',
      'When the relationship is smooth and the data is scarce: a Gaussian process makes good use of few points.',
      'Not for large tables: fitting takes time that grows with the cube of the number of rows. Brick DNN fits it on at most 5,000 training rows.',
    ],
    train:
      'Upload a CSV file with eight numeric feature columns and a numeric target (last column), or use a dataset from the <strong>Data</strong> tab. The model is fitted once, without epochs; the run shows the fitted kernel, its log marginal likelihood, RMSE and R². If a hyperparameter ends at its bound, Brick DNN says so and offers a fix.',
    related: ['periodic_gp', 'stacking_ensemble_regression', 'tabular_regression'],
  },
  {
    id: 'svm_classifier',
    name: 'SVM classifier',
    title: 'SVM classifier with RBF kernel in scikit-learn – example code',
    description:
      'A support vector machine (SVC) with an RBF kernel classifies rows of 4 numeric features, with class probabilities by Platt scaling, as a scikit-learn script.',
    h1: 'SVM classifier with RBF kernel in scikit-learn',
    task: 'classification',
    summary:
      'A support vector machine with an RBF kernel classifies rows of 4 features, with probabilities.',
    lead: 'A support vector machine with an RBF kernel for rows of four numeric features, for example the measurements of iris flowers. It separates the classes with the widest possible margin and gives class probabilities as well.',
    what: [
      'The <code>SVC</code> brick takes the features from the <code>Input</code> directly. With the RBF kernel, the boundary between the classes can be curved; <code>C = 1.0</code> sets how much a misclassified training row costs, and <code>gamma = scale</code> sets the width of the kernel from the spread of the features.',
      'An SVM gives scores, not probabilities. The generated script wraps it in <code>CalibratedClassifierCV</code>, which turns the scores into probabilities by Platt scaling with 5-fold cross-validation, and puts a <code>StandardScaler</code> in front, because an SVM depends on the scale of the features.',
    ],
    when: [
      'For small and medium tables with numeric features and a clear but non-linear boundary between the classes.',
      'Up to tens of thousands of rows: the fit grows between the square and the cube of the rows. Brick DNN fits it on at most 50,000 training rows; for more, use <a href="/examples/gradient-boosting-classifier/">gradient boosting</a>.',
      'With rare classes, set <code>class_weight</code> to <code>balanced</code>.',
    ],
    train:
      'Upload a CSV file of four numeric features and a class column (text labels such as <code>setosa</code> work), or a prepared dataset from the <strong>Data</strong> tab. The run shows accuracy, the confusion matrix and the number of support vectors; a hyperparameter search over <code>C</code> and <code>gamma</code> with cross-validation is one switch away.',
    related: [
      'gradient_boosting_classifier',
      'voting_ensemble_classifier',
      'hybrid_mlp_gradient_boosting',
    ],
  },
  {
    id: 'periodic_gp',
    name: 'Periodic Gaussian process',
    title: 'Periodic Gaussian process for seasonal data – scikit-learn example',
    description:
      'A Gaussian process for seasonal time series in scikit-learn: an RBF × ExpSineSquared periodic kernel plus a smooth trend and noise, built from kernel bricks.',
    h1: 'Periodic Gaussian process for seasonal data',
    task: 'regression',
    summary:
      'A seasonal pattern (RBF × ExpSineSquared) plus a smooth trend and noise, for a value over time.',
    lead: 'A Gaussian process for a value over time that repeats with the seasons and drifts slowly: its kernel adds a periodic part, a smooth trend and noise, each a brick on the canvas.',
    what: [
      'The Input is one feature, the time. <code>ExpSineSquared</code> is a periodic kernel: points one period apart count as similar. Multiplied by an <code>RBF</code> with a long length scale (10), the seasonal pattern may change slowly from year to year. A second <code>RBF</code> with length scale 5 models the trend, and <code>WhiteKernel</code> the noise. <code>KernelSum</code> adds the three parts.',
      'Fitting tunes all five hyperparameters, including the periodicity. Set its starting value to the season length in standardised units of the time feature, because the pipeline standardises the features first.',
    ],
    when: [
      'For seasonal measurements with few points: monthly sales, temperatures, energy use, sensor readings with a daily cycle.',
      'When you want the uncertainty of a forecast to grow the further it reaches from the data.',
      'To learn how composite kernels describe structure: remove the trend brick and see what the fit makes of the data.',
    ],
    train:
      'Use a CSV file with the time as a number (for example days since the first measurement) as the only feature column and the measured value as the target (last column). The run shows the fitted kernel with all its hyperparameters and the log marginal likelihood.',
    related: ['gp_regression', 'stacking_ensemble_regression', 'tabular_regression'],
  },
  {
    id: 'gradient_boosting_classifier',
    name: 'Gradient boosting classifier',
    title: 'Gradient boosting classifier in scikit-learn – HistGradientBoosting',
    description:
      'A histogram-based gradient boosting classifier for 30 numeric features in scikit-learn, with balanced class weights for rare classes such as fraud cases.',
    h1: 'Gradient boosting classifier in scikit-learn',
    task: 'classification',
    summary:
      'Gradient-boosted trees classify rows of 30 features; balanced class weights for rare classes.',
    lead: 'Gradient-boosted decision trees are the strong default for tabular data. This example classifies rows of 30 numeric features and weights the classes so that a rare class, a few percent of fraud cases for instance, is not ignored.',
    what: [
      'The <code>HistGradientBoostingClassifier</code> brick adds up to 200 decision trees one after the other (<code>max_iter</code>), each correcting the errors of the trees before it, scaled by the learning rate of 0.1. A tree has at most 31 leaves, each with at least 20 rows. The features are binned into histograms first, which makes it fast on hundreds of thousands of rows.',
      '<code>class_weight = balanced</code> weights each class inversely to its frequency. Without it, a model can reach a high accuracy by hardly ever predicting the rare class. Early stopping (<code>auto</code>) switches on above 10,000 training rows and stops adding trees when a held-out tenth of them no longer improves.',
    ],
    when: [
      'As the first model to try on almost any table: it handles different scales and non-linear relationships without tuning.',
      'For large tables: its time grows about linearly with the rows. Brick DNN fits it on up to 1,000,000 training rows.',
      'For imbalanced classes, with balanced weights and a look at balanced accuracy, recall and PR AUC instead of accuracy: see the <a href="/guides/imbalanced-classes/">guide on imbalanced classes</a>.',
    ],
    train:
      'Upload a CSV file or prepare a table in the <strong>Data</strong> tab: missing values have to be filled in first, as for the other classical models. The run reports accuracy, balanced accuracy, F1, the confusion matrix and, for two classes, ROC and PR curves with a decision threshold you can move. A search over learning rate, leaves and rows per leaf with cross-validation finds better settings.',
    related: ['svm_classifier', 'voting_ensemble_classifier', 'hybrid_mlp_gradient_boosting'],
  },
  {
    id: 'lightgbm_classifier',
    name: 'LightGBM classifier',
    title: 'LightGBM classifier for tabular data – Python example code',
    description:
      'A LightGBM classifier for 20 numeric features: up to 1,000 leaf-wise trees, early stopping on the validation rows and balanced class weights, with code.',
    h1: 'LightGBM classifier for tabular data',
    task: 'classification',
    summary:
      'LightGBM grows its trees leaf by leaf; early stopping on the validation rows picks how many.',
    lead: 'LightGBM is one of the gradient boosting libraries that win tabular competitions. This example classifies rows of 20 numeric features and lets early stopping decide how many trees it needs.',
    what: [
      'The <code>LGBMClassifier</code> brick adds trees one after the other, each correcting the errors of the trees before it, scaled by the learning rate of 0.05. LightGBM grows each tree leaf by leaf, always splitting the leaf that helps most, up to 31 leaves (<code>num_leaves</code>) with at least 20 rows each. Each tree sees 80 % of the training rows and of the features (<code>subsample</code>, <code>colsample_bytree</code>), which makes the trees differ and the model generalise better.',
      'At most 1,000 trees (<code>n_estimators</code>): with <code>early_stopping_rounds = 50</code> the fit watches the loss on the validation rows, stops when 50 trees in a row have not improved it and keeps the best round. The validation metrics are then a little optimistic; the test rows stay untouched. <code>class_weight = balanced</code> weights each class inversely to its frequency.',
    ],
    when: [
      'For tables with tens of thousands to millions of rows: LightGBM is among the fastest boosting libraries. Brick DNN fits it on up to 1,000,000 training rows.',
      'When the last per cent of accuracy counts, as in competitions; combine it with XGBoost and CatBoost in a <a href="/examples/boosting-stacking-ensemble/">stacking ensemble</a>.',
      'Text columns are one-hot encoded in the <strong>Data</strong> tab first, as for the other classical models.',
    ],
    train:
      'Upload a CSV file of 20 numeric features and a class column, or a dataset from the <strong>Data</strong> tab, and keep a validation split above 0 for early stopping. The run reports how many trees early stopping kept, accuracy, balanced accuracy, the confusion matrix and, for two classes, ROC and PR curves. A search over learning rate, leaves and rows per leaf finds better settings; the downloaded model works in the generated script with LightGBM installed.',
    related: ['boosting_stacking_ensemble', 'gradient_boosting_classifier', 'svm_classifier'],
  },
  {
    id: 'voting_ensemble_classifier',
    name: 'Voting ensemble',
    title: 'Voting ensemble classifier in scikit-learn – soft voting example',
    description:
      'An ensemble of gradient boosting, an SVM and a Gaussian process classifier with soft voting in scikit-learn: the class probabilities are averaged. With code.',
    h1: 'Voting ensemble classifier in scikit-learn',
    task: 'classification',
    summary:
      'Gradient boosting, an SVM and a Gaussian process; soft voting averages their probabilities.',
    lead: 'Three different classifiers look at the same four features, and the ensemble averages their class probabilities. Models that make different mistakes cancel some of them out.',
    what: [
      'The <code>Input</code> feeds all three models: <code>HistGradientBoostingClassifier</code> (trees), <code>SVC</code> (a kernel boundary) and <code>GaussianProcessClassifier</code> (with its default kernel, a scaled RBF). The dotted connections into <code>VotingClassifier</code> carry the models themselves, not their predictions; the ensemble fits each of them on the training rows.',
      'With <code>voting = soft</code>, the ensemble averages the predicted probabilities and picks the most probable class. Hard voting would count the predicted classes instead. Optional weights make one model count more; they follow the order of the connections.',
    ],
    when: [
      'When several models are about equally good and disagree on different rows: the average is often better than the best single model.',
      'For small and medium tables: the slowest model sets the limit, here the Gaussian process classifier (at most 2,000 training rows in Brick DNN).',
      'To check whether combining helps: the run lists each model’s own score next to the ensemble’s.',
    ],
    train:
      'Upload a CSV file of four numeric features and a class column, or a dataset from the <strong>Data</strong> tab. The ensemble trains like a single classical model, with the same evaluation (confusion matrix, threshold analysis for two classes), explanations and download.',
    related: ['stacking_ensemble_regression', 'gradient_boosting_classifier', 'svm_classifier'],
  },
  {
    id: 'stacking_ensemble_regression',
    name: 'Stacking ensemble regression',
    title: 'Stacking ensemble regression in scikit-learn – example code',
    description:
      'A stacking ensemble in scikit-learn: gradient boosting, an SVR and a Gaussian process predict a number, and a ridge regression learns how to weigh them.',
    h1: 'Stacking ensemble regression in scikit-learn',
    task: 'regression',
    summary: 'Gradient boosting, an SVR and a Gaussian process, combined by a ridge regression.',
    lead: 'Stacking lets a model learn how much to trust other models: gradient boosting, a support vector regression and a Gaussian process predict a number from eight features, and a ridge regression learns from their predictions how to combine them.',
    what: [
      'The three models get the features from the <code>Input</code>. The Gaussian process has a kernel of its own, an <code>RBF</code> plus a <code>WhiteKernel</code> joined by <code>KernelSum</code> (the dashed connections carry the kernel). The dotted connections into <code>StackingRegressor</code> carry the three models.',
      'With <code>cv = 5</code>, each model is fitted five times, each time without a fifth of the training rows, and predicts that fifth. The ridge regression learns from these predictions for rows the models did not see, so it does not reward a model that only memorised the data. Then each model is fitted once more on all rows. Fitting therefore takes about six times as long as fitting the three models once.',
    ],
    when: [
      'When different kinds of models capture different parts of the relationship, for example house prices with smooth and abrupt effects.',
      'When a simple average is not enough because one model is clearly better in some regions.',
      'On tables up to a few thousand rows here, because of the Gaussian process (at most 5,000 training rows in Brick DNN).',
    ],
    train:
      'Upload a CSV file of eight numeric features and a numeric target, or use a dataset from the <strong>Data</strong> tab. The run reports RMSE and R² of the ensemble and of each model on the validation rows, so you see whether stacking beats its best model.',
    related: ['voting_ensemble_classifier', 'gp_regression', 'tabular_regression'],
  },
  {
    id: 'boosting_stacking_ensemble',
    name: 'XGBoost, LightGBM and CatBoost stacked',
    title: 'Stacking XGBoost, LightGBM and CatBoost – ensemble example',
    description:
      'A stacking ensemble of XGBoost, LightGBM and CatBoost classifiers in Python: a logistic regression learns from their cross-validated probabilities how to combine them.',
    h1: 'Stacking ensemble of XGBoost, LightGBM and CatBoost',
    task: 'classification',
    summary:
      'The three boosting libraries of tabular competitions, combined by a logistic regression.',
    lead: 'Competition winners rarely submit a single model. Here XGBoost, LightGBM and CatBoost classify the same 20 features, and a logistic regression learns how much to trust each of them.',
    what: [
      'The three models get the features from the <code>Input</code>: <code>XGBClassifier</code> with 300 trees of depth 6, <code>LGBMClassifier</code> with 300 leaf-wise trees of up to 31 leaves and <code>CatBoostClassifier</code> with 500 symmetric trees of depth 6, all with a learning rate of 0.05. They build their trees differently, so they make different mistakes. The dotted connections into <code>StackingClassifier</code> carry the models themselves.',
      'With <code>cv = 5</code>, each model is fitted five times, each time without a fifth of the training rows, and predicts the class probabilities of that fifth. The logistic regression learns from these predictions for rows the models did not see; then every model is fitted once more on all rows. In an ensemble the models get no validation rows, so they fit all their trees: early stopping is off, and the numbers of trees are lower than for a single model.',
    ],
    when: [
      'When each library has reached good settings on its own and you want the extra accuracy of combining them.',
      'For tables up to hundreds of thousands of rows: fitting takes about six times as long as fitting the three models once.',
      'To compare the libraries: the run lists each model’s own score next to the ensemble’s.',
    ],
    train:
      'Upload a CSV file of 20 numeric features and a class column, or a dataset from the <strong>Data</strong> tab. The ensemble trains like a single classical model, with the same evaluation, explanations and download; the generated script needs xgboost, lightgbm and catboost installed.',
    related: ['lightgbm_classifier', 'stacking_ensemble_regression', 'voting_ensemble_classifier'],
  },
  {
    id: 'hybrid_mlp_gradient_boosting',
    name: 'MLP features for gradient boosting',
    title: 'MLP features for gradient boosting – hybrid PyTorch + scikit-learn',
    description:
      'A hybrid network: an MLP in PyTorch learns 32 features from 8 columns, then gradient-boosted trees in scikit-learn are fitted on them. Diagram, shapes and code.',
    h1: 'MLP features for gradient boosting',
    task: 'classification',
    summary:
      'An MLP learns 32 features, then gradient-boosted trees are fitted on them and predict the class.',
    lead: 'A hybrid of deep learning and trees: two <code>Linear</code> layers learn 32 new features from eight columns, and a gradient boosting classifier is fitted on those features instead of the raw columns.',
    what: [
      'Training has two phases in one run. First the neural layers learn with a temporary linear head on the target, like a small network of their own: 8 → 64 → ReLU → 32, with 2,656 trainable parameters. Then the head is dropped, the layers compute the 32 features of every row, and the <code>HistGradientBoostingClassifier</code> is fitted on the features of the training rows.',
      'The generated script contains both parts: the layers as a PyTorch module (<code>FeatureExtractor</code>), the classifier as a scikit-learn pipeline, and the code that trains both phases on a CSV file and saves both.',
    ],
    when: [
      'To test whether learned features help trees on your table: compare with the <a href="/examples/gradient-boosting-classifier/">plain gradient boosting</a> on the same split.',
      'With <em>frozen random features</em> in the training settings, as a quick baseline for what learning the features adds.',
      'As a pattern for images: a small CNN that ends in <code>Flatten</code> or <code>Linear</code> before an SVM or gradient boosting.',
    ],
    train:
      'Set the Input to the number of feature columns of your data, then train on a CSV file, a dataset from the <strong>Data</strong> tab or demo data. The training panel shows the network’s settings (epochs, learning rate …) for the first phase; the curves are those of the feature learning, and the evaluation is that of the fitted classifier. The download is a ZIP file with both parts and the script.',
    related: ['gradient_boosting_classifier', 'tabular_regression', 'mlp_mnist'],
  },
];

export const examplesEn = { gallery, examples };
