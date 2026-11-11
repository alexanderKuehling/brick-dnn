import { ApiError, request } from '@/api/client';
import { hashOf } from './pixels';
import type { ExplainMethods, ExplainRequest, PickedSample } from './types';

/** Retries of an explanation while the server is busy with other predictions (503). */
const BUSY_RETRIES = 8;
const BUSY_WAIT_MS = 2500;

// The server explains one thing at a time (like predictions): the page asks one at a time.
let queue: Promise<unknown> = Promise.resolve();

function wait(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(resolve, ms);
    signal?.addEventListener('abort', () => {
      clearTimeout(timer);
      reject(new DOMException('Aborted', 'AbortError'));
    });
  });
}

async function explainOnce<T>(runId: string, body: ExplainRequest, signal?: AbortSignal) {
  for (let attempt = 0; ; attempt += 1) {
    try {
      return await request<T>(`/api/training/runs/${runId}/explain/`, {
        method: 'POST',
        body,
        auth: true,
        signal,
      });
    } catch (error) {
      if (!(error instanceof ApiError) || error.status !== 503 || attempt >= BUSY_RETRIES) {
        throw error;
      }
      if (/disabled|not installed/i.test(error.message)) {
        throw error;
      }
      await wait(BUSY_WAIT_MS, signal);
    }
  }
}

/** Explanations of a trained run (the server computes them like a prediction). */
export const explainApi = {
  methods: (runId: string, signal?: AbortSignal) =>
    request<ExplainMethods>(`/api/training/runs/${runId}/explain/`, { auth: true, signal }),
  /** One at a time, waiting while the server is busy with other predictions. */
  run: <T>(runId: string, body: ExplainRequest, signal?: AbortSignal): Promise<T> => {
    const next = queue.then(() => {
      if (signal?.aborted) {
        throw new DOMException('Aborted', 'AbortError');
      }
      return explainOnce<T>(runId, body, signal);
    });
    queue = next.catch(() => undefined);
    return next;
  },
};

/** The request fields that name the picked sample. */
export function sampleFields(sample: PickedSample | null): Partial<ExplainRequest> {
  if (!sample || sample.kind === 'random') {
    return {};
  }
  if (sample.kind === 'validation') {
    return { sample: sample.index };
  }
  return { input: sample.input, input_scale: sample.scale ?? 'raw' };
}

/** A stable key for caching answers per sample on the page. */
export function sampleKey(sample: PickedSample | null): string {
  if (!sample) {
    return 'none';
  }
  if (sample.kind === 'validation') {
    return `v${sample.index}`;
  }
  if (sample.kind === 'random') {
    return 'random';
  }
  return `i:${hashOf(sample.input)}`;
}
