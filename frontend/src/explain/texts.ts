import { i18n } from '@/i18n';

/** What each explanation shows, in one or two plain sentences (also in docs/explain.md). */
export const METHOD_TEXT = {
  get saliency() {
    return i18n.t('ui.explain.texts.saliencySmoothgradHowMuchEach');
  },
  get gradcam() {
    return i18n.t('ui.explain.texts.gradCamWhereTheFeature');
  },
  get occlusion() {
    return i18n.t('ui.explain.texts.occlusionHowMuchTheProbability');
  },
  get shapley() {
    return i18n.t('ui.explain.texts.shapleyValuesHowMuchEach');
  },
  get integrated() {
    return i18n.t('ui.explain.texts.integratedGradientsTheGradientOf');
  },
  get permutation() {
    return i18n.t('ui.explain.texts.permutationImportanceHowMuchThe');
  },
  get boundary() {
    return i18n.t('ui.explain.texts.thePredictionOfTheModel');
  },
  get boundaryStd() {
    return i18n.t('ui.explain.texts.theGaussianProcessSPredictive');
  },
  get activations() {
    return i18n.t('ui.explain.texts.whatEveryBrickPutsOut');
  },
  get filters() {
    return i18n.t('ui.explain.texts.theLearnedFiltersOfThe');
  },
  get weights() {
    return i18n.t('ui.explain.texts.howTheTrainedWeightsOf');
  },
} as const;
