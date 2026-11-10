import { request } from '@/api/client';
import { hashOf } from './pixels';
import type { ExplainMethods, ExplainRequest, PickedSample } from './types';

/** Explanations of a trained run (the server computes them like a prediction). */
export const explainApi = {
  methods: (runId: string, signal?: AbortSignal) =>
    request<ExplainMethods>(`/api/training/runs/${runId}/explain/`, { auth: true, signal }),
  run: <T>(runId: string, body: ExplainRequest, signal?: AbortSignal) =>
    request<T>(`/api/training/runs/${runId}/explain/`, {
      method: 'POST',
      body,
      auth: true,
      signal,
    }),
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
