import { ApiError, errorMessage } from './client';
import type { EngineOp, EngineRequest, EngineResponse } from './engine.worker';

const LOAD_ERROR =
  'The code generator could not be loaded. Check your internet connection and try again.';

interface Pending {
  resolve: (value: unknown) => void;
  reject: (error: Error) => void;
}

let worker: Worker | null = null;
let nextId = 1;
const pending = new Map<number, Pending>();

function abortError(): DOMException {
  return new DOMException('The request was aborted.', 'AbortError');
}

function getWorker(): Worker {
  if (worker) {
    return worker;
  }
  const created = new Worker(new URL('./engine.worker.ts', import.meta.url), { type: 'module' });
  created.onmessage = (event: MessageEvent<EngineResponse>) => {
    const response = event.data;
    const request = pending.get(response.id);
    if (!request) {
      return; // aborted in the meantime
    }
    pending.delete(response.id);
    if ('error' in response) {
      request.reject(new ApiError(0, { detail: response.error }, LOAD_ERROR));
    } else if (response.status >= 400) {
      const message = errorMessage(response.data, response.status);
      request.reject(new ApiError(response.status, response.data, message));
    } else {
      request.resolve(response.data);
    }
  };
  created.onerror = () => {
    // The worker itself failed to start: give up on it and let the next request try again.
    created.terminate();
    worker = null;
    pending.forEach((request) => request.reject(new ApiError(0, null, LOAD_ERROR)));
    pending.clear();
  };
  worker = created;
  return created;
}

/**
 * Answer a request of the public API in the browser (online demo, see `engine.worker.ts`).
 * Errors are `ApiError`s with the same status and data the server would send.
 */
export function callEngine<T>(op: EngineOp, body?: unknown, signal?: AbortSignal): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    if (signal?.aborted) {
      reject(abortError());
      return;
    }
    const id = nextId++;
    const onAbort = () => {
      pending.delete(id);
      reject(abortError());
    };
    signal?.addEventListener('abort', onAbort, { once: true });
    const settle = () => signal?.removeEventListener('abort', onAbort);
    pending.set(id, {
      resolve: (value) => {
        settle();
        resolve(value as T);
      },
      reject: (error) => {
        settle();
        reject(error);
      },
    });
    const message: EngineRequest = {
      id,
      op,
      body: body === undefined ? null : JSON.stringify(body),
    };
    getWorker().postMessage(message);
  });
}
