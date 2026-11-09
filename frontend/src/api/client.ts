import type { Tokens } from './types';

const BASE_URL = (import.meta.env.VITE_API_URL ?? '').replace(/\/+$/, '');
const TOKEN_KEY = 'brickdnn:tokens';

/**
 * The absolute URL of the API, without a trailing slash: VITE_API_URL if it is set, otherwise
 * this site (which serves the API under /api/ itself or through a proxy).
 */
export function apiBaseUrl(): string {
  return new URL(BASE_URL || '/', window.location.href).href.replace(/\/+$/, '');
}

export class ApiError extends Error {
  constructor(
    public status: number,
    public data: unknown,
    message: string
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

// --------------------------------------------------------------------------
// JWT storage
// --------------------------------------------------------------------------

function readTokens(): Tokens | null {
  try {
    const raw = localStorage.getItem(TOKEN_KEY);
    return raw ? (JSON.parse(raw) as Tokens) : null;
  } catch {
    return null;
  }
}

let tokens: Tokens | null = readTokens();
const listeners = new Set<() => void>();

export function getTokens(): Tokens | null {
  return tokens;
}

export function setTokens(next: Tokens | null): void {
  tokens = next;
  try {
    if (next) {
      localStorage.setItem(TOKEN_KEY, JSON.stringify(next));
    } else {
      localStorage.removeItem(TOKEN_KEY);
    }
  } catch {
    // Storage may be unavailable (private mode); tokens then live in memory only.
  }
  listeners.forEach((listener) => listener());
}

export function onTokensChange(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

// Keep tabs in sync: when another tab refreshes the tokens or logs out, follow along.
if (typeof window !== 'undefined') {
  window.addEventListener('storage', (event) => {
    if (event.key === TOKEN_KEY || event.key === null) {
      tokens = readTokens();
      listeners.forEach((listener) => listener());
    }
  });
}

let refreshing: Promise<boolean> | null = null;

/** Exchange the refresh token for a new access token. Concurrent calls share one request. */
async function refreshAccessToken(): Promise<boolean> {
  const current = tokens;
  if (!current?.refresh) {
    return false;
  }
  if (!refreshing) {
    refreshing = (async () => {
      try {
        const response = await fetch(`${BASE_URL}/api/auth/token/refresh/`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ refresh: current.refresh }),
        });
        if (!response.ok) {
          // Only a rejected refresh token ends the session, and only if no other tab has
          // replaced it in the meantime. Rate limits or server errors keep the user logged in.
          if (response.status === 401 && tokens?.refresh === current.refresh) {
            setTokens(null);
          }
          return tokens !== null && tokens.refresh !== current.refresh;
        }
        const data = (await response.json()) as Partial<Tokens>;
        if (tokens?.refresh !== current.refresh) {
          // Logged out (or refreshed elsewhere) while this request was running.
          return tokens !== null;
        }
        setTokens({ access: data.access!, refresh: data.refresh ?? current.refresh });
        return true;
      } catch {
        return false;
      } finally {
        refreshing = null;
      }
    })();
  }
  return refreshing;
}

// --------------------------------------------------------------------------
// Requests
// --------------------------------------------------------------------------

function capitalize(text: string): string {
  return text.charAt(0).toUpperCase() + text.slice(1).replace(/_/g, ' ');
}

/** Turn a Django REST framework error payload into a readable sentence. */
export function errorMessage(data: unknown, status: number): string {
  if (data && typeof data === 'object') {
    const record = data as Record<string, unknown>;
    if (typeof record.detail === 'string') {
      return record.detail;
    }
    const parts = Object.entries(record).flatMap(([field, value]) => {
      const messages = Array.isArray(value) ? value : typeof value === 'string' ? [value] : [];
      return messages.map((message) =>
        field === 'non_field_errors' ? String(message) : `${capitalize(field)}: ${message}`
      );
    });
    if (parts.length > 0) {
      return parts.join(' ');
    }
  }
  if (status === 429) {
    return 'Too many requests. Please wait a moment.';
  }
  if (status === 413) {
    return 'The file is too large for the server.';
  }
  return `Request failed (HTTP ${status}).`;
}

export interface RequestOptions {
  method?: 'GET' | 'POST' | 'PUT' | 'PATCH' | 'DELETE';
  /** Sent as JSON, except FormData, which is sent as multipart/form-data. */
  body?: unknown;
  /** Send the JWT access token (and refresh it once on 401). */
  auth?: boolean;
  signal?: AbortSignal;
  /**
   * Called while a FormData body is sent, with the bytes sent so far and the size of the body.
   * Such uploads go through XMLHttpRequest: fetch cannot report the progress of an upload.
   */
  onUploadProgress?: (sent: number, total: number) => void;
}

/** How a file upload is sent: `signal` cancels it, `onUploadProgress` follows it. */
export type UploadOptions = Pick<RequestOptions, 'signal' | 'onUploadProgress'>;

function isAbort(error: unknown): boolean {
  return (error as { name?: unknown } | null)?.name === 'AbortError';
}

/** Seconds until a JWT expires (from its `exp` claim), or null if it cannot be read. */
export function tokenLifetime(token: string, now = Date.now()): number | null {
  try {
    const payload = token.split('.')[1];
    if (!payload) {
      return null;
    }
    const base64 = payload.replace(/-/g, '+').replace(/_/g, '/');
    const padded = base64.padEnd(Math.ceil(base64.length / 4) * 4, '=');
    const { exp } = JSON.parse(atob(padded)) as { exp?: unknown };
    return typeof exp === 'number' ? exp - now / 1000 : null;
  } catch {
    return null;
  }
}

/** Requests refresh the access token first if it expires within this many seconds. */
const REQUEST_TOKEN_MARGIN = 10;

/** Uploads refresh it earlier: they can take a while to send. */
const UPLOAD_TOKEN_MARGIN = 60;

/** Live connections get a token that is valid for at least this many more seconds. */
const LIVE_TOKEN_MARGIN = 60;

/**
 * The access token for a connection that cannot refresh it on a 401 itself (the live editing
 * WebSocket): refreshed first if it expires soon, or always with `force`. null: logged out.
 */
export async function freshAccessToken(force = false): Promise<string | null> {
  const access = tokens?.access;
  if (!access) {
    return null;
  }
  const lifetime = tokenLifetime(access);
  if (force || (lifetime !== null && lifetime < LIVE_TOKEN_MARGIN)) {
    await refreshAccessToken();
  }
  return tokens?.access ?? null;
}

/** The response headers of an XMLHttpRequest. */
function xhrHeaders(raw: string): Headers {
  const headers = new Headers();
  for (const line of raw.split(/\r?\n/)) {
    const colon = line.indexOf(':');
    if (colon > 0) {
      try {
        headers.append(line.slice(0, colon).trim(), line.slice(colon + 1).trim());
      } catch {
        // Skip a header the Headers class does not accept.
      }
    }
  }
  return headers;
}

/** Statuses whose response cannot have a body. */
const NULL_BODY_STATUSES = [204, 205, 304];

/**
 * Send a FormData body with XMLHttpRequest, which reports the progress of the upload, and
 * return the answer as a Response. Fails like fetch: a TypeError if the server cannot be
 * reached, an AbortError when `signal` is aborted.
 */
function sendWithProgress(
  url: string,
  init: { method: string; headers: Record<string, string>; body: FormData; signal?: AbortSignal },
  onProgress: (sent: number, total: number) => void
): Promise<Response> {
  const { method, headers, body, signal } = init;
  const aborted = () => new DOMException('The operation was aborted.', 'AbortError');
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(aborted());
      return;
    }
    const xhr = new XMLHttpRequest();
    const abort = () => xhr.abort();
    const settle = (finish: () => void) => {
      signal?.removeEventListener('abort', abort);
      finish();
    };
    xhr.open(method, url);
    for (const [name, value] of Object.entries(headers)) {
      xhr.setRequestHeader(name, value);
    }
    // Upload events are only sent to listeners that exist before the request is sent.
    const progress = (event: ProgressEvent) => {
      if (event.lengthComputable) {
        onProgress(event.loaded, event.total);
      }
    };
    xhr.upload.onprogress = progress;
    xhr.upload.onload = progress;
    xhr.onload = () =>
      settle(() => {
        try {
          const empty = NULL_BODY_STATUSES.includes(xhr.status);
          resolve(
            new Response(empty ? null : xhr.responseText, {
              status: xhr.status,
              headers: xhrHeaders(xhr.getAllResponseHeaders()),
            })
          );
        } catch (error) {
          // E.g. status 0, which Response does not accept.
          reject(error);
        }
      });
    xhr.onerror = () => settle(() => reject(new TypeError('Network request failed')));
    xhr.onabort = () => settle(() => reject(aborted()));
    signal?.addEventListener('abort', abort, { once: true });
    xhr.send(body);
  });
}

/** Send a request (refreshing the access token once on 401) and return the raw response. */
async function send(path: string, options: RequestOptions, accept: string): Promise<Response> {
  const { method = 'GET', body, auth = false, signal, onUploadProgress } = options;
  const multipart = typeof FormData !== 'undefined' && body instanceof FormData;

  // Refresh an access token that has expired (or is about to) before sending, rather than
  // waiting for the 401: browsers log every 401 as an error, e.g. on the first request after
  // the page was left open. Uploads get more margin: the server rejects an expired token
  // before reading a large upload, which browsers often report as a dropped connection.
  if (auth && tokens?.access) {
    const lifetime = tokenLifetime(tokens.access);
    const margin = multipart ? UPLOAD_TOKEN_MARGIN : REQUEST_TOKEN_MARGIN;
    if (lifetime !== null && lifetime < margin) {
      await refreshAccessToken();
    }
  }

  const attempt = () => {
    const headers: Record<string, string> = { Accept: accept };
    // The browser sets the multipart Content-Type itself, including the boundary.
    if (body !== undefined && !multipart) {
      headers['Content-Type'] = 'application/json';
    }
    if (auth && tokens?.access) {
      headers.Authorization = `Bearer ${tokens.access}`;
    }
    if (multipart && onUploadProgress) {
      const init = { method, headers, body: body as FormData, signal };
      return sendWithProgress(`${BASE_URL}${path}`, init, onUploadProgress);
    }
    return fetch(`${BASE_URL}${path}`, {
      method,
      headers,
      body: body === undefined ? undefined : multipart ? (body as FormData) : JSON.stringify(body),
      signal,
    });
  };

  try {
    const response = await attempt();
    if (response.status === 401 && auth && (await refreshAccessToken())) {
      return await attempt();
    }
    return response;
  } catch (error) {
    if (isAbort(error)) {
      throw error;
    }
    throw new ApiError(0, null, 'Cannot reach the Brick DNN server. Is the backend running?');
  }
}

const INTERRUPTED = 'The connection to the server was interrupted. Please try again.';

/** Read the body; an abort is passed on, a broken connection becomes an ApiError. */
async function readBody<T>(read: () => Promise<T>): Promise<T> {
  try {
    return await read();
  } catch (error) {
    if (isAbort(error)) {
      throw error;
    }
    throw new ApiError(0, null, INTERRUPTED);
  }
}

async function failure(response: Response): Promise<ApiError> {
  let text = '';
  try {
    text = await response.text();
  } catch (error) {
    if (isAbort(error)) {
      throw error;
    }
  }
  let data: unknown = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    // Not JSON (e.g. an HTML error page of a proxy): fall back to the status.
  }
  return new ApiError(response.status, data, errorMessage(data, response.status));
}

export async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const response = await send(path, options, 'application/json');
  if (response.status === 204) {
    return undefined as T;
  }
  if (!response.ok) {
    throw await failure(response);
  }
  const text = await readBody(() => response.text());
  if (text.trim() === '') {
    return null as T;
  }
  try {
    return JSON.parse(text) as T;
  } catch {
    throw new ApiError(response.status, null, 'The server sent an unexpected response.');
  }
}

/** The file name from a Content-Disposition header, if there is one. */
export function dispositionFilename(header: string | null): string | null {
  if (!header) {
    return null;
  }
  const encoded = /filename\*\s*=\s*(?:UTF-8|utf-8)''([^;]+)/.exec(header);
  if (encoded) {
    try {
      return decodeURIComponent(encoded[1].trim());
    } catch {
      // Fall through to the plain file name.
    }
  }
  const plain = /filename\s*=\s*("([^"]*)"|[^;]+)/.exec(header);
  const name = plain ? (plain[2] ?? plain[1]).trim() : '';
  return name || null;
}

export interface FileDownload {
  blob: Blob;
  /** From Content-Disposition, or null if the server did not send one. */
  filename: string | null;
}

/** Fetch a file (e.g. trained weights) with the same authentication as API calls. */
export async function requestFile(
  path: string,
  options: Omit<RequestOptions, 'body'> = {}
): Promise<FileDownload> {
  const response = await send(path, options, '*/*');
  if (!response.ok) {
    throw await failure(response);
  }
  return {
    blob: await readBody(() => response.blob()),
    filename: dispositionFilename(response.headers.get('Content-Disposition')),
  };
}
