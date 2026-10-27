import type { Tokens } from './types';

const BASE_URL = (import.meta.env.VITE_API_URL ?? '').replace(/\/+$/, '');
const TOKEN_KEY = 'brickdnn:tokens';

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
  return `Request failed (HTTP ${status}).`;
}

export interface RequestOptions {
  method?: 'GET' | 'POST' | 'PUT' | 'PATCH' | 'DELETE';
  body?: unknown;
  /** Send the JWT access token (and refresh it once on 401). */
  auth?: boolean;
  signal?: AbortSignal;
}

export async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const { method = 'GET', body, auth = false, signal } = options;

  const send = () => {
    const headers: Record<string, string> = { Accept: 'application/json' };
    if (body !== undefined) {
      headers['Content-Type'] = 'application/json';
    }
    if (auth && tokens?.access) {
      headers.Authorization = `Bearer ${tokens.access}`;
    }
    return fetch(`${BASE_URL}${path}`, {
      method,
      headers,
      body: body === undefined ? undefined : JSON.stringify(body),
      signal,
    });
  };

  let response: Response;
  try {
    response = await send();
    if (response.status === 401 && auth && (await refreshAccessToken())) {
      response = await send();
    }
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') {
      throw error;
    }
    throw new ApiError(0, null, 'Cannot reach the Brick DNN server. Is the backend running?');
  }

  if (response.status === 204) {
    return undefined as T;
  }
  const data: unknown = await response.json().catch(() => null);
  if (!response.ok) {
    throw new ApiError(response.status, data, errorMessage(data, response.status));
  }
  return data as T;
}
