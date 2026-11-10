/** What each explanation shows, in one or two plain sentences (also in docs/explain.md). */
export const METHOD_TEXT = {
  saliency:
    'Saliency (SmoothGrad): how much each pixel changes the score of the class if it changes a little, averaged over noisy copies of the image. Bright pixels matter; it is noisy and shows sensitivity, not a reason.',
  gradcam:
    'Grad-CAM: where the feature maps of a convolution brick see evidence for the class (the maps weighted by how much the class score depends on them). Coarse: one value per position of that brick, scaled up to the image.',
  occlusion:
    'Occlusion: how much the probability of the class drops when a square patch is covered with the average colour. Model-agnostic and easy to read, but it only sees what one patch at a time can hide.',
  shapley:
    'Shapley values: how much each feature moved this prediction away from the prediction for an average row (the mean of the training rows). They add up to the difference. Correlated features share their credit in ways that can surprise.',
  integrated:
    'Integrated gradients: the gradient of the output along the straight path from the average row to this row, times the change of each feature. For a linear model they equal the Shapley values.',
  permutation:
    'Permutation importance: how much the score on validation rows drops when the values of one feature are shuffled, which breaks its link to the target. Bars show the mean over repeats, whiskers one standard deviation. Correlated features can hide each other: shuffling one, the model still finds the information in the other.',
  boundary:
    'The prediction of the model for every point of the plane spanned by the two input features, with the training points on top. Pale areas are where the model hesitates.',
  boundaryStd:
    'The Gaussian process’s predictive standard deviation: how uncertain it is. It grows away from the training points.',
  activations:
    'What every brick puts out for this sample. Convolutions: the first feature maps as small grey images (bright = high). Vectors: the values as bars. Sequences: a heatmap (rows = positions).',
  filters:
    'The learned filters of the first convolution: what small patterns it responds to. Red means a positive weight, blue a negative one (colour images: the filters as colours).',
  weights:
    'How the trained weights of every brick are distributed. Most weights near 0 and a few large ones is typical; very large values can mean the training was unstable.',
} as const;
