import { api } from '.';
import { afterEach, beforeEach, vi } from 'vitest';
import {
  ApiError,
  dispositionFilename,
  errorMessage,
  getTokens,
  request,
  requestFile,
  setTokens,
  tokenLifetime,
} from './client';

describe('errorMessage', () => {
  it('uses DRF detail messages', () => {
    expect(errorMessage({ detail: 'Nope.' }, 400)).toBe('Nope.');
  });

  it('joins field errors', () => {
    expect(
      errorMessage(
        { username: ['A user with that username already exists.'], non_field_errors: ['Bad.'] },
        400
      )
    ).toBe('Username: A user with that username already exists. Bad.');
  });

  it('falls back to the status', () => {
    expect(errorMessage(null, 500)).toBe('Request failed (HTTP 500).');
    expect(errorMessage(null, 429)).toContain('Too many requests');
    expect(errorMessage(null, 413)).toContain('too large');
  });
});

describe('dispositionFilename', () => {
  it('reads plain, quoted and encoded file names', () => {
    expect(dispositionFilename('attachment; filename="small_cnn.pt"')).toBe('small_cnn.pt');
    expect(dispositionFilename('attachment; filename=model.pt')).toBe('model.pt');
    expect(dispositionFilename("attachment; filename*=UTF-8''gr%C3%B6%C3%9Fe.pt")).toBe('größe.pt');
    expect(dispositionFilename('attachment')).toBeNull();
    expect(dispositionFilename(null)).toBeNull();
  });
});

function json(data: unknown, status = 200): Response {
  return new Response(JSON.stringify(data), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}

/** A fake JWT that expires at `exp` (unix seconds). */
function jwt(exp: number): string {
  const payload = btoa(JSON.stringify({ exp, user_id: 1 }))
    .replace(/\+/g, '-')
    .replace(/\//g, '_')
    .replace(/=+$/, '');
  return `header.${payload}.signature`;
}

describe('tokenLifetime', () => {
  it('reads the expiry of a JWT', () => {
    expect(tokenLifetime(jwt(1000), 400_000)).toBe(600);
    expect(tokenLifetime('not-a-token')).toBeNull();
    expect(tokenLifetime('a.%%%.c')).toBeNull();
  });
});

function headersOf(call: unknown[]): Record<string, string> {
  return (call[1] as RequestInit).headers as Record<string, string>;
}

describe('requests', () => {
  const fetchMock = vi.fn<typeof fetch>();

  beforeEach(() => {
    vi.stubGlobal('fetch', fetchMock);
    setTokens({ access: 'old-access', refresh: 'refresh-token' });
  });

  afterEach(() => {
    fetchMock.mockReset();
    vi.unstubAllGlobals();
    setTokens(null);
  });

  it('sends JSON with the access token', async () => {
    fetchMock.mockResolvedValueOnce(json({ ok: true }));
    await request('/api/thing/', { method: 'POST', body: { a: 1 }, auth: true });
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe('/api/thing/');
    expect(init?.body).toBe('{"a":1}');
    expect(headersOf(fetchMock.mock.calls[0])).toMatchObject({
      'Content-Type': 'application/json',
      Authorization: 'Bearer old-access',
    });
  });

  it('uploads files as multipart without setting a Content-Type', async () => {
    fetchMock.mockResolvedValueOnce(json({ id: 3, name: 'iris' }, 201));
    const file = new File(['a,b\n1,2\n'], 'iris.csv', { type: 'text/csv' });
    const dataset = await api.training.datasets.upload(file, 'Iris');

    expect(dataset).toEqual({ id: 3, name: 'iris' });
    const init = fetchMock.mock.calls[0][1]!;
    expect(init.method).toBe('POST');
    expect(init.body).toBeInstanceOf(FormData);
    const form = init.body as FormData;
    expect((form.get('file') as File).name).toBe('iris.csv');
    expect(form.get('name')).toBe('Iris');
    const headers = headersOf(fetchMock.mock.calls[0]);
    expect(headers['Content-Type']).toBeUndefined();
    expect(headers.Authorization).toBe('Bearer old-access');
  });

  it('refreshes the token once and repeats an upload after 401', async () => {
    fetchMock
      .mockResolvedValueOnce(json({ detail: 'Token expired' }, 401))
      .mockResolvedValueOnce(json({ access: 'new-access' }))
      .mockResolvedValueOnce(json({ id: 4 }, 201));
    const file = new File(['x'], 'pictures.zip');
    await api.training.datasets.upload(file);

    expect(fetchMock).toHaveBeenCalledTimes(3);
    expect(fetchMock.mock.calls[1][0]).toBe('/api/auth/token/refresh/');
    expect(headersOf(fetchMock.mock.calls[2]).Authorization).toBe('Bearer new-access');
    expect(fetchMock.mock.calls[2][1]!.body).toBeInstanceOf(FormData);
    expect(getTokens()?.access).toBe('new-access');
  });

  it('turns upload errors into readable messages', async () => {
    fetchMock.mockResolvedValueOnce(json({ detail: 'Only .csv and .zip files.' }, 400));
    await expect(api.training.datasets.upload(new File(['x'], 'a.txt'))).rejects.toMatchObject({
      status: 400,
      message: 'Only .csv and .zip files.',
    });
  });

  it('downloads files with authentication', async () => {
    fetchMock.mockResolvedValueOnce(
      new Response(new Uint8Array([1, 2, 3]), {
        status: 200,
        headers: {
          'Content-Type': 'application/octet-stream',
          'Content-Disposition': 'attachment; filename="small_cnn.pt"',
        },
      })
    );
    const result = await api.training.runs.downloadWeights('abc');

    expect(fetchMock.mock.calls[0][0]).toBe('/api/training/runs/abc/weights/');
    expect(headersOf(fetchMock.mock.calls[0]).Authorization).toBe('Bearer old-access');
    expect(result.filename).toBe('small_cnn.pt');
    expect(result.blob.size).toBe(3);
  });

  it('reports download errors from the JSON body', async () => {
    fetchMock.mockResolvedValueOnce(json({ detail: 'No weights available for this run.' }, 404));
    const error = await requestFile('/api/training/runs/abc/weights/', { auth: true }).catch(
      (e: unknown) => e
    );
    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).message).toBe('No weights available for this run.');
  });

  it('refreshes an expiring token before uploading a file', async () => {
    setTokens({ access: jwt(Date.now() / 1000 + 30), refresh: 'refresh-token' });
    fetchMock
      .mockResolvedValueOnce(json({ access: 'fresh-access' }))
      .mockResolvedValueOnce(json({ id: 5 }, 201));
    await api.training.datasets.upload(new File(['x'], 'big.zip'));

    expect(fetchMock.mock.calls[0][0]).toBe('/api/auth/token/refresh/');
    expect(headersOf(fetchMock.mock.calls[1]).Authorization).toBe('Bearer fresh-access');
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it('does not refresh a token that is still valid for a while', async () => {
    setTokens({ access: jwt(Date.now() / 1000 + 600), refresh: 'refresh-token' });
    fetchMock.mockResolvedValueOnce(json({ id: 5 }, 201));
    await api.training.datasets.upload(new File(['x'], 'big.zip'));
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it('treats a broken response body as an error', async () => {
    fetchMock.mockResolvedValueOnce(new Response('<html>Bad gateway</html>', { status: 200 }));
    await expect(api.training.runs.get('r1')).rejects.toBeInstanceOf(ApiError);
  });

  it('accepts an empty response body', async () => {
    fetchMock.mockResolvedValueOnce(new Response('', { status: 200 }));
    await expect(request('/api/thing/')).resolves.toBeNull();
    fetchMock.mockResolvedValueOnce(new Response(null, { status: 204 }));
    await expect(request('/api/thing/', { method: 'DELETE' })).resolves.toBeUndefined();
  });

  it('passes on an abort while the body is read', async () => {
    const aborted = new DOMException('The operation was aborted.', 'AbortError');
    const response = json({ id: 'r1' });
    vi.spyOn(response, 'text').mockRejectedValue(aborted);
    fetchMock.mockResolvedValueOnce(response);
    await expect(api.training.runs.get('r1')).rejects.toBe(aborted);
  });

  it('reports an unreachable server', async () => {
    fetchMock.mockRejectedValueOnce(new TypeError('Failed to fetch'));
    await expect(api.training.status()).rejects.toMatchObject({ status: 0 });
  });
});
