import { CONTACT_EMAIL, OVERAGE_PRICE, PLANS, type PlanFacts } from '../site.ts';
import type { LandingContent } from './types.ts';

function euro(amount: number): string {
  return `€${Number.isInteger(amount) ? amount : amount.toFixed(2)}`;
}

function cpuTime(minutes: number): string {
  if (minutes >= 60 && minutes % 60 === 0) {
    const hours = minutes / 60;
    return `${hours} CPU ${hours === 1 ? 'hour' : 'hours'}`;
  }
  return `${minutes} CPU minutes`;
}

const paid = PLANS.filter((plan) => plan.price > 0);
const paidPrices = `${paid
  .slice(0, -1)
  .map((plan) => euro(plan.price))
  .join(', ')} or ${euro(paid[paid.length - 1].price)}`;
const free = PLANS.find((plan) => plan.price === 0)!;
const mail = `<a href="mailto:${CONTACT_EMAIL}">${CONTACT_EMAIL}</a>`;

export const en: LandingContent = {
  lang: 'en',
  locale: 'en_US',
  title: 'Brick DNN – visual editor for neural networks with PyTorch export',
  description:
    'Build neural networks from layers in the browser, check every tensor shape, train on servers in Switzerland and export readable PyTorch code. Free plan available.',
  languageName: 'English',
  skipLink: 'Skip to content',
  nav: {
    label: 'Sections',
    useCases: 'Use cases',
    features: 'Features',
    pricing: 'Pricing',
    faq: 'FAQ',
    editor: 'Open the editor',
  },
  hero: {
    title: 'Build neural networks from layers. Export the PyTorch code.',
    lead: 'Brick DNN is a browser-based editor for neural networks and classical machine-learning models. You connect layers on a canvas, see the shape of every tensor while you build, train on the server and take away a readable PyTorch or scikit-learn script.',
    primary: 'Open the editor',
    secondary: 'Pricing',
    facts: [
      'Free plan',
      'No account needed to build and export',
      'Open source, Apache 2.0',
      'Servers in Switzerland',
    ],
    image: {
      id: 'editor',
      alt: 'The Brick DNN editor: a palette of layers on the left, a convolutional network on the canvas with the output shape of every layer, and the settings of the selected layer on the right.',
    },
    caption:
      'A convolutional network for MNIST in the editor. Every layer shows its output shape and number of parameters.',
  },
  glance: {
    title: 'At a glance',
    items: [
      {
        term: 'What it is',
        html: 'A web application for designing, checking, training and exporting neural networks and classical machine-learning models. A model is a graph of layers (“bricks”); Brick DNN infers every tensor shape and generates a self-contained PyTorch script, or a scikit-learn script for classical models.',
      },
      {
        term: 'Who it is for',
        html: 'Students and teachers of machine learning and deep learning, researchers who work with tabular or image data, and practitioners who need a correct starting point quickly.',
      },
      {
        term: 'What it costs',
        html: `Building, checking and exporting code are free and work without an account. The free plan includes training on the server (${free.cpuHoursPerMonth} CPU hours per month, ${free.storageGb} GB storage). Paid plans cost ${paidPrices} per month, VAT included.`,
      },
      {
        term: 'Where the data lives',
        html: 'The application, uploaded files and training results are on a server in Switzerland. Accounts and project metadata are in a database in London (United Kingdom). No cookies, no tracking.',
      },
      {
        term: 'Source code',
        html: 'Open source under the Apache 2.0 licence. It also runs on your own server with Docker Compose.',
      },
    ],
  },
  useCases: {
    title: 'Use cases',
    intro:
      'Brick DNN is used for teaching, for research with tabular and image data, and for getting from an architecture sketch to working code.',
    items: [
      {
        title: 'Teaching deep learning',
        html: 'Students connect layers and see how the shape of the data changes from layer to layer, with plain messages when something does not fit, e.g. <q>Kernel size 7 is larger than the padded input size 4</q>. <em>How data flows</em> steps one sample through the network; the inspector shows how a layer’s parameters add up and its receptive field. For group work, networks can be shared and edited together live.',
        image: {
          id: 'data-flow',
          alt: '“How data flows” in the editor: a card steps one sample through a CNN and shows how max pooling turns 32 × 28 × 28 values into 32 × 14 × 14, below the layers on the canvas.',
        },
      },
      {
        title: 'Tabular research data',
        html: 'Import CSV, Excel, Parquet or JSON files, clean them in recorded steps (missing values, outliers, categories, dates) and check whether the data is ready for training. Compare gradient boosting, support vector machines and Gaussian processes with neural networks on the same fixed train, validation and test split, and explain predictions with Shapley values or permutation importance.',
        image: {
          id: 'data-prep',
          alt: 'A preparation of field-trial data in the Data tab: on the left the steps that fill in missing values, clip outliers and encode categories, each with what it did; on the right the prepared table with a distribution in every column header.',
        },
      },
      {
        title: 'Image segmentation',
        html: 'Train a U-Net, or any network that keeps the image size, on a ZIP file of images and masks, or on the built-in <em>Shapes</em> dataset without uploading anything. Each epoch reports mean IoU, Dice and pixel accuracy, and the predicted masks are shown next to the true ones.',
        image: {
          id: 'segmentation',
          alt: 'Predictions of a U-Net trained on the built-in Shapes dataset: for each validation image the image, the true mask, the predicted mask and an overlay, with IoU and pixel accuracy.',
        },
      },
      {
        title: 'Reproducible code for papers',
        html: 'Every model exports as a self-contained Python file: an <code>nn.Module</code> with named layers for PyTorch, or a scikit-learn script for classical models and ensembles. The data preparation exports as a pandas script. The code runs without Brick DNN and can be published together with the paper.',
        image: {
          id: 'code-export',
          alt: 'The code dialog with the generated PyTorch module of a small CNN: the layers in __init__ and the forward method, ready to download as a Python file.',
        },
      },
      {
        title: 'An AI assistant that builds and trains',
        html: 'Connect Claude or another MCP client to your account with OAuth. The assistant can build and check networks, prepare data, start trainings and make predictions. You decide whether it may change things or only read, can give it a monthly budget and see each of its actions in a log.',
        image: {
          id: 'agent-activity',
          alt: 'The agent activity page: the calls an AI assistant made through the MCP server with an API token, such as analysing a network, reading a training run and generating code, each with its result and duration.',
        },
      },
    ],
  },
  steps: {
    title: 'How it works',
    items: [
      {
        title: 'Build',
        html: 'Drag layers onto the canvas and connect them, or start from an example: MLP, CNN, residual block, U-Net, LSTM, Transformer, gradient boosting or an ensemble. Fields such as <code>in_features</code> are filled in from the incoming tensor.',
        image: {
          id: 'build',
          alt: 'A voting ensemble in the builder: one input connected to gradient boosting, a support vector machine and a Gaussian process, combined by a voting classifier.',
        },
      },
      {
        title: 'Train',
        html: 'Choose demo data, MNIST, Fashion-MNIST, CIFAR-10 or your own CSV or image files. Brick DNN checks beforehand whether network and data fit, then trains on the server with live loss and metric curves, early stopping and an optional hyperparameter search.',
        image: {
          id: 'training-setup',
          alt: 'The training panel with a prepared dataset selected and the check “Ready to train: classification into 2 classes from 18 features”.',
        },
      },
      {
        title: 'Export',
        html: 'Download the PyTorch or scikit-learn script, the trained weights and the prepared data. The code is yours: read it, change it and run it wherever you like.',
        image: {
          id: 'export',
          alt: 'The end of a finished run: buttons to download the weights, the preprocessing and the script, or to load the network into the editor.',
        },
      },
    ],
  },
  features: {
    title: 'Features',
    intro: 'One tool from the first sketch to a trained, explained model.',
    items: [
      {
        title: 'Shape inference and checks',
        html: 'Every layer shows its output shape and parameter count while you build. Mistakes are marked before anything runs, with a message that says what to change.',
        image: {
          id: 'checks',
          alt: 'A CNN with a mistake: a Conv2d layer is marked red with the message “Kernel size 16 is larger than the (padded) input size 14”, and the layers after it have no shape.',
        },
      },
      {
        title: 'Training on the server',
        html: 'A queue, early stopping that keeps the best weights, and loss and metric curves for every epoch while the network learns.',
        image: {
          id: 'training-curves',
          alt: 'Loss and accuracy per epoch of a small CNN trained on MNIST, for the training and the validation data, under the test accuracy of 98.6 %.',
        },
      },
      {
        title: 'Evaluation on held-out data',
        html: 'Test data measured once after training: accuracy, balanced accuracy, F1 and a confusion matrix; for two classes also ROC and PR curves and a decision threshold you can move.',
        image: {
          id: 'evaluation',
          alt: 'The confusion matrix of the CNN on 3,000 validation images of MNIST: true digits in the rows, predicted digits in the columns, almost all on the diagonal.',
        },
      },
      {
        title: 'Hyperparameter search',
        html: 'Random or grid search over learning rate, batch size and optimizer, or over a classical model’s parameters with cross-validation, ranked by the loss or by a metric such as balanced accuracy, ROC AUC, PR AUC, MAE or R².',
        image: {
          id: 'search-results',
          alt: 'The trials of a random search for a gradient boosting model: learning rate, depth, rows per leaf and class weights of each trial with its cross-validated loss, accuracy and PR AUC; the best trial is highlighted.',
        },
      },
      {
        title: 'Explanations',
        html: 'Saliency maps, Grad-CAM and occlusion for images; Shapley values, integrated gradients and permutation importance for tables; the output of every layer for one sample.',
        image: {
          id: 'explain',
          alt: 'Permutation importance of the gradient boosting model: how much the validation accuracy drops when one feature is shuffled, most for rainfall, then temperature and nitrogen.',
        },
      },
      {
        title: 'Working together',
        html: 'Share a network with people or with a link, edit it together live, comment on it and go back to earlier versions.',
        image: {
          id: 'collaboration',
          alt: 'A network shared by two people: their avatars and a live badge in the toolbar, the layer the other person selected marked with her name, and her question about it in the comments panel.',
        },
      },
      {
        title: 'Real graphs',
        html: 'Skip connections (<code>Add</code>), merges (<code>Concat</code>), several inputs and outputs. 24 layer types for MLPs, CNNs, RNN, LSTM, GRU and Transformers.',
      },
      {
        title: 'Classical models and ensembles',
        html: 'Gradient boosting, support vector machines and Gaussian processes with kernels composed from bricks, voting and stacking ensembles, and hybrid networks whose layers learn the features of a classical model.',
      },
      {
        title: 'Data preparation',
        html: 'CSV, Excel, OpenDocument, JSON, Parquet and Feather files or a link. Join, fill in, clip outliers, scale, encode and split dates, with charts for every step and a readiness check.',
      },
      {
        title: 'MCP server and REST API',
        html: 'AI assistants connect with OAuth; scripts use personal API tokens. Both work with the same networks, data and trainings as the web app.',
      },
      {
        title: 'Your data in your hands',
        html: 'Building and exporting need no account. With one, you can download all your data or delete the account yourself, and protect it with two-factor login.',
      },
    ],
  },
  privacy: {
    title: 'Privacy and data location',
    intro: 'Brick DNN is operated from Germany and follows the GDPR. What is stored where:',
    items: [
      {
        title: 'Server in Switzerland',
        html: 'The application, uploaded files, trained weights and training results are on a server of Infomaniak in Geneva, Switzerland. Trainings run there; your data does not leave the server for them.',
      },
      {
        title: 'Database in London',
        html: 'Accounts, networks, comments and the details of data and trainings (not the files) are in a PostgreSQL database of Supabase in London, United Kingdom.',
      },
      {
        title: 'No cookies, no tracking',
        html: 'No analytics, no advertising, no cookies in the app. Fonts, scripts and images come from this server, apart from Paddle’s checkout when you buy a plan. This page loads nothing from elsewhere.',
      },
      {
        title: 'Few, named processors',
        html: 'Technical error reports of the app go to Sentry’s EU region without user name, e-mail address or account ID; e-mails about your account are sent through Brevo in the EU. Hosting, database, e-mail and error reports run under data processing agreements. Payments go through Paddle, only if you buy a plan.',
      },
    ],
    more: 'The privacy policy (in German) lists every provider, purpose and retention period.',
  },
  teaching: {
    title: 'For teaching and research',
    items: [
      'A free plan with training on the server; no account is needed to build networks and export code.',
      'Exported PyTorch and scikit-learn code runs without Brick DNN, so course material and published results stay usable.',
      'Data stays on servers in Switzerland and in the United Kingdom, without tracking.',
      'Open source: an institution can run its own instance.',
    ],
    discount: `Discounts for teaching and research are available on request: ${mail}.`,
  },
  pricing: {
    title: 'Pricing',
    intro:
      'Building, checking and exporting are free. The plans differ in compute and storage for training on the server.',
    caption: 'Plans on brickdnn.com, prices per month including VAT',
    columns: [
      'Plan',
      'Price per month',
      'Compute per month',
      'Storage',
      'Longest training',
      'Trainings at a time',
    ],
    row: (plan: PlanFacts) => [
      plan.name,
      euro(plan.price),
      `${plan.cpuHoursPerMonth} CPU ${plan.cpuHoursPerMonth === 1 ? 'hour' : 'hours'}`,
      `${plan.storageGb} GB`,
      cpuTime(plan.cpuMinutesPerRun),
      String(plan.concurrentTrainings),
    ],
    notes: [
      `Paid plans can add compute beyond the allowance at ${euro(OVERAGE_PRICE)} per CPU hour, only if you switch it on.`,
      'Monthly subscriptions, cancellable at the end of each month. Payment and invoices are handled by Paddle.',
    ],
    more: 'All details on the pricing page',
  },
  faq: {
    title: 'Frequently asked questions',
    items: [
      {
        question: 'What is Brick DNN?',
        html: 'Brick DNN is a web application for building neural networks and classical machine-learning models from layers (“bricks”). It infers the shape of every tensor, points out mistakes before anything runs, trains models on the server and exports them as readable PyTorch or scikit-learn code.',
      },
      {
        question: 'Do I need an account?',
        html: 'No account is needed to build and check networks and to export their code. Saving networks, uploading data and training need a free account.',
      },
      {
        question: 'Which frameworks does the exported code use?',
        html: 'Neural networks export as PyTorch code (an <code>nn.Module</code> with a sanity check and, for tabular data, a training loop for CSV files). Classical models and ensembles export as scikit-learn scripts, and data preparations as pandas scripts. Trained weights can be downloaded together with the matching script.',
      },
      {
        question: 'Where is my data stored and processed?',
        html: 'Uploaded files, trained weights and training results are stored and processed on a server of Infomaniak in Switzerland. Accounts and project metadata are in a database of Supabase in London. There are no cookies and no tracking; the privacy policy lists every provider.',
      },
      {
        question: 'What does it cost?',
        html: `Building, checking and exporting are free. The free plan includes ${free.cpuHoursPerMonth} CPU hours of training per month and ${free.storageGb} GB of storage. Paid plans with more compute and storage cost ${paidPrices} per month, VAT included.`,
      },
      {
        question: 'Can I use Brick DNN in a course or for research?',
        html: `Yes. Students can build and export networks without an account, and the free plan covers small trainings. Discounts for teaching and research are available on request at ${mail}.`,
      },
      {
        question: 'Can I run Brick DNN on my own server?',
        html: 'Yes. Brick DNN is open source under the Apache 2.0 licence and runs with Docker Compose. On your own machine, training uses a CUDA or Apple GPU if PyTorch finds one.',
      },
      {
        question: 'Which AI assistants work with Brick DNN?',
        html: 'Any client of the Model Context Protocol (MCP) that signs in with OAuth, for example Claude. Clients without OAuth and your own scripts use personal API tokens.',
      },
    ],
  },
  footer: {
    tagline: 'Neural networks from bricks, with readable PyTorch code.',
    legalLabel: 'Legal',
    legal: { imprint: 'Imprint', privacy: 'Privacy', terms: 'Terms', refund: 'Refund policy' },
    legalNote: 'Imprint, privacy policy and terms are in German.',
    source: 'Source code on GitHub',
    contact: 'Contact',
    language: 'Language',
  },
  appDescription:
    'Browser-based editor for neural networks and classical machine-learning models with live shape inference, training on the server and export of readable PyTorch and scikit-learn code.',
  featureList: [
    'Drag-and-drop editor for neural networks with live shape inference',
    'Export of runnable PyTorch and scikit-learn code',
    'Gradient boosting, support vector machines, Gaussian processes, ensembles and hybrid networks',
    'Data preparation for CSV, Excel, JSON and Parquet files',
    'Training on the server with hyperparameter search',
    'Image segmentation with U-Net',
    'Explanations: saliency maps, Grad-CAM, Shapley values, permutation importance',
    'Live collaboration on shared networks',
    'MCP server for AI assistants and REST API',
  ],
  ogImageAlt: 'Brick DNN: the editor with a convolutional network and the shape of every layer.',
};
