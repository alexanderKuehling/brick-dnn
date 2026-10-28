import { ApiError } from './client';
import type { EngineOp } from './engine.worker';
import type { Api } from './index';
import type { Analysis, Example, GenerateResult, LayerCatalog, TrainingStatus } from './types';

/** Loaded on first use. The mode check keeps the demo's web worker out of normal builds. */
function callEngine<T>(op: EngineOp, body?: unknown, signal?: AbortSignal): Promise<T> {
  if (import.meta.env.MODE !== 'demo') {
    return Promise.reject(new ApiError(0, null, 'The browser engine is only part of the demo.'));
  }
  return import('./browserEngine').then((engine) => engine.callEngine<T>(op, body, signal));
}

const TRAINING_STATUS: TrainingStatus = {
  enabled: false,
  available: false,
  reason:
    'Training runs on your own computer, so it is not part of this online demo. To train ' +
    'networks, install Brick DNN locally: github.com/alexanderKuehling/brick-dnn',
  torch_version: null,
  torchvision: false,
  devices: [],
  default_device: null,
  builtin_datasets: [],
  max_upload_mb: 0,
  max_concurrent_runs: 0,
  login_required: false,
};

/**
 * The API of the online demo: the code generator runs in the browser, accounts and training
 * are not available (the UI hides them).
 */
export function createDemoApi(server: Api): Api {
  return {
    ...server,
    layers: () => callEngine<LayerCatalog>('layers'),
    examples: () => callEngine<Example[]>('examples'),
    analyze: (graph, signal) => callEngine<Analysis>('analyze', { graph }, signal),
    generate: (graph, options = {}) => callEngine<GenerateResult>('generate', { graph, options }),
    me: () => Promise.reject(new ApiError(401, null, 'There are no accounts in the online demo.')),
    training: { ...server.training, status: () => Promise.resolve(TRAINING_STATUS) },
  };
}
