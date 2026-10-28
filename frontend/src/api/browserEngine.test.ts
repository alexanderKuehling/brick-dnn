import { afterAll, beforeAll, describe, expect, it, vi } from 'vitest';
import { ApiError } from './client';
import type { EngineRequest, EngineResponse } from './engine.worker';

/** Stands in for the Pyodide worker: the test decides what it answers. */
class FakeWorker {
  static last: FakeWorker | null = null;
  sent: EngineRequest[] = [];
  onmessage: ((event: MessageEvent<EngineResponse>) => void) | null = null;
  onerror: (() => void) | null = null;

  constructor() {
    FakeWorker.last = this;
  }

  postMessage(message: EngineRequest) {
    this.sent.push(message);
  }

  reply(response: EngineResponse) {
    this.onmessage?.({ data: response } as MessageEvent<EngineResponse>);
  }

  terminate() {}
}

let callEngine: typeof import('./browserEngine').callEngine;

beforeAll(async () => {
  vi.stubGlobal('Worker', FakeWorker);
  ({ callEngine } = await import('./browserEngine'));
});

afterAll(() => {
  vi.unstubAllGlobals();
});

function lastRequest(): EngineRequest {
  const sent = FakeWorker.last!.sent;
  return sent[sent.length - 1];
}

describe('callEngine', () => {
  it('sends the body as JSON and resolves with the data', async () => {
    const promise = callEngine<{ ok: boolean }>('analyze', { graph: { nodes: [] } });
    const request = lastRequest();
    expect(request.op).toBe('analyze');
    expect(JSON.parse(request.body!)).toEqual({ graph: { nodes: [] } });
    FakeWorker.last!.reply({ id: request.id, status: 200, data: { ok: true } });
    await expect(promise).resolves.toEqual({ ok: true });
  });

  it('turns error answers into ApiErrors like the server client does', async () => {
    const promise = callEngine('generate', { graph: {} });
    const data = { detail: 'The network has no Input layer.', analysis: {} };
    FakeWorker.last!.reply({ id: lastRequest().id, status: 400, data });
    const error = await promise.catch((e: unknown) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect(error).toMatchObject({ status: 400, data, message: 'The network has no Input layer.' });
  });

  it('reports when Pyodide could not be loaded', async () => {
    const promise = callEngine('layers');
    FakeWorker.last!.reply({ id: lastRequest().id, error: 'NetworkError' });
    await expect(promise).rejects.toMatchObject({
      status: 0,
      message: expect.stringContaining('could not be loaded') as string,
    });
  });

  it('rejects with an AbortError and ignores a late answer', async () => {
    const controller = new AbortController();
    const promise = callEngine('analyze', { graph: {} }, controller.signal);
    const { id } = lastRequest();
    controller.abort();
    await expect(promise).rejects.toMatchObject({ name: 'AbortError' });
    expect(() => FakeWorker.last!.reply({ id, status: 200, data: {} })).not.toThrow();
  });
});
