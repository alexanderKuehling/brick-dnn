/** The training panel: setup, settings, runs. */
export const training = {
  drawer: {
    title: 'Train this network',
    titleRuns: 'Trainings of this network',
    sidePanel: 'Back to the side panel',
    fullWidth: 'Full width',
    unavailable: 'Training is not available',
    loginTitle: 'Log in to train',
    loginText:
      'Training runs on the Brick DNN server and needs a free account, just like saving. Your network stays on the canvas while you log in.',
    newTraining: 'New training',
    runs: 'Runs',
    viewerNote:
      'You can view this network: you see how it was trained, but cannot start trainings in it. To train it yourself, duplicate it.',
    cannotLoad: 'Cannot load the training',
    allRuns: 'All runs',
    lostConnection: 'Lost the connection to the server ({{error}}). Retrying …',
    noSklearn: 'scikit-learn is not installed on the server.',
    hybridNoSklearn:
      'This network is hybrid: its neural layers learn the features with PyTorch, which is installed, and its classical model is fitted on them with scikit-learn, which is missing on the machine of the Brick DNN backend. In the <code>backend</code> folder, with the virtual environment active:',
    classicalNoSklearn:
      'Classical models (Gaussian processes, SVMs, gradient boosting) are fitted with scikit-learn on the machine of the Brick DNN backend; they do not need PyTorch. In the <code>backend</code> folder, with the virtual environment active:',
    dockerSklearn: 'The Docker image includes scikit-learn. Restart the backend and check again.',
    checkAgain: 'Check again',
    disabled: 'Training is disabled on this server.',
    noTorch: 'PyTorch is not installed on the server.',
    needsTorch:
      'Training runs on the machine of the Brick DNN backend and needs PyTorch there. To install it:',
    local: 'Local installation',
    inBackend: 'In the <code>backend</code> folder, with the virtual environment active:',
    hybridAlso:
      'This hybrid network also needs scikit-learn for its classical model: <code>pip install -r requirements.txt</code> in the <code>backend</code> folder (the Docker image includes it).',
    restart: 'Then restart the backend and check again.',
  },
  source: {
    label: 'Data source',
    tabs: { random: 'Demo data', builtin: 'Built-in', upload: 'Your data' },
    random:
      'Random inputs with a hidden rule to learn. Works with any network and shows whether it can learn at all. Nothing is downloaded.',
    dataset: 'Dataset',
    downloadedOrGenerated: 'Downloaded to the server on first use, or generated there.',
    downloaded: 'Downloaded to the server on first use.',
    needsTorchvision: 'Downloaded datasets need torchvision on the server.',
    sizes: '{{train}} training and {{test}} test images',
    segmentationDetails_one: '{{sizes}} with masks · {{count}} class ({{classes}}) · no download',
    segmentationDetails_other:
      '{{sizes}} with masks · {{count}} classes ({{classes}}) · no download',
    details_one: '{{sizes}} · {{count}} class · {{mb}} MB download',
    details_other: '{{sizes}} · {{count}} classes · {{mb}} MB download',
    dataTab: 'Data tab',
    messyCsv: 'Messy CSV? Clean, encode and analyse it in the',
  },
  setup: {
    budget: {
      budget_exceeded: 'Not enough compute budget',
      estimate_too_high: 'Too much compute for the budget',
      storage_quota_exceeded: 'The storage of your account is full',
      other: 'Not enough budget',
    },
    deleteTitle: 'Delete “{{name}}”?',
    deleteText:
      'The uploaded data is removed from the server. Finished trainings keep their results.',
    deleteFailed: 'Cannot delete the data',
    empty: 'The canvas is empty. Build a network (or load an example) to train it.',
    fixFirst: 'Fix the layers marked in red first, then come back here.',
    showProblems: 'Show problems',
    running: 'A training is running',
    stillTraining: '“{{name}}” is still training. You can start a new one when it has finished.',
    stillTrainingBy:
      '“{{name}}” (started by {{user}}) is still training. You can start a new one when it has finished.',
    showIt: 'Show it',
    data: 'Data',
    settings: 'Settings',
    tooMany: 'Too many trainings are waiting',
    another: 'Another training is running',
    cannotStart: 'Cannot start the training',
    openIt: 'Open it',
    visible: 'Visible to everyone in this network',
    startSearch: 'Start search',
    start: 'Start training',
    noWorker: 'No training worker is running',
    noWorkerText:
      'This server cannot train right now. A training you start waits in the queue until a worker is back.',
    afterYours: 'starts after your running training',
    busy: 'The server is busy: this training will wait in the queue (position {{position}}; an estimate).',
    busyStart:
      'The server is busy: this training will wait in the queue (position {{position}}, {{start}}; an estimate).',
    theData: 'the data',
    cannotCheck: 'Cannot check the data',
    checking: 'Checking the network against the data …',
    noFit: 'Does not fit the data',
    ready: 'Ready to train',
    fitExactly: 'To fit “{{name}}” exactly, change:',
    fit: 'To fit “{{name}}”, change:',
    adapt: 'Adapt network to “{{name}}”',
  },
  settings: {
    searched: 'Set by the search',
    notTrained: 'Not used: frozen features are not trained',
    automatic: 'Automatic',
    automaticDevice: 'Automatic: {{device}}',
    name: 'Name',
    valSplit: 'Validation split',
    noValClassical:
      'No validation data: the model is measured on the test rows only (without them, on its training rows).',
    noVal:
      'No validation data: the last epoch is kept, no early stopping or search; scores only on test (or training) data.',
    officialTest: 'Test on the official test images',
    holdOut: 'Hold out a test set',
    ownSplitImages:
      'This dataset brings its own split, the folders train, val and test: {{split}} images for training, validation and test. Training uses it as it is.',
    ownSplitRows:
      'This dataset brings its own split from the Data tab: {{split}} rows for training, validation and test. Training uses it as it is; change it in the Data tab.',
    builtinTest:
      'Always on for built-in datasets: the official test images are never trained on and not used to choose the weights; the kept weights are measured on them once, after training. The validation split is taken from the training images.',
    testHint:
      'Never trained on and not used to compare settings: the final model is measured on it once, after training.',
    testSplit: 'Test split',
    maxSamples: 'Max. samples',
    maxRowsHint:
      '{{model}} is fitted on at most {{rows}} training rows on this server; larger training sets are subsampled.',
    maxSamplesHint: 'Training rows to fit on (empty: all).',
    allUpTo: 'All, up to {{rows}}',
    all: 'All',
    seed: 'Seed',
    random: 'Random',
    classicalNote:
      '{{model}} is fitted once with scikit-learn, on the CPU: only the split, the number of rows and the seed apply.',
    classicalNoteKernel:
      '{{model}} is fitted once with scikit-learn, on the CPU: only the split, the number of rows and the seed apply; the fit optimises the kernel hyperparameters itself.',
    search: 'Hyperparameter search',
    searchClassical:
      'Fits the model several times with other values of its parameters and keeps the best one.',
    searchHybrid:
      'Fits {{model}} several times with other values of its parameters on the features, and can learn the features again with other training settings in every trial; keeps the best trial.',
    searchNeural:
      'Trains the network several times with different settings and keeps the best one (by default the one with the lowest validation loss).',
    hybridNote:
      'Two phases: the neural layers learn {{features}} with a temporary linear head on the target (the settings below), then {{model}} is fitted on the features of the training rows.',
    hybridNoteMax:
      'Two phases: the neural layers learn {{features}} with a temporary linear head on the target (the settings below), then {{model}} is fitted on the features of the training rows (at most {{rows}} on this server).',
    nFeatures_one: '{{count}} feature',
    nFeatures_other: '{{count}} features',
    theFeatures: 'the features',
    features: 'Features',
    frozenHint:
      'The neural layers keep their random initial weights: nothing is trained, so the training settings (epochs, batch size, learning rate, optimizer, weight decay, early stopping) are not used.',
    epochs: 'Epochs',
    epochsFrozen: 'Not used: the features stay random',
    perTrial: 'Per trial of the search',
    epochsHint: 'Passes over the training data',
    batchSize: 'Batch size',
    imagesPerStepMax: 'Images per optimizer step; up to {{max}} fit into memory',
    imagesPerStep: 'Images per optimizer step (4 to 16 is usual for segmentation)',
    samplesPerStep: 'Samples per optimizer step',
    learningRate: 'Learning rate',
    optimizer: 'Optimizer',
    device: 'Device',
    loss: 'Loss',
    lossHint: 'The Dice term helps classes that cover only a few pixels',
    advanced: 'Advanced settings',
    weightDecay: 'Weight decay',
    earlyStopping: 'Early stopping',
    earlyStoppingHint: 'Stop when the validation loss has not improved for this many epochs',
    restoreBest: 'Keep the weights of the best epoch',
    restoreBestHint:
      'Test and save the weights with the lowest validation loss, not those of the last epoch.',
    restoreBestNoVal: 'Not used without validation data: the weights of the last epoch are kept.',
    options: {
      featureTraining: {
        pretrain: 'Learn them (temporary head on the target)',
        frozen: 'Frozen random features (no training)',
      },
      loss: {
        cross_entropy: 'Cross-entropy',
        cross_entropy_dice: 'Cross-entropy + Dice',
      },
    },
  },
};
