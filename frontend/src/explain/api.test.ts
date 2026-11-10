import { vi } from 'vitest';
import { ApiError } from '@/api/client';
import { explainApi, sampleFields } from './api';

const { request } = vi.hoisted(() => ({ request: vi.fn() }));
vi.mock('@/api/client', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@/api/client')>()),
  request,
}));

beforeEach(() => {
  request.mockReset();
  vi.useFakeTimers();
});
afterEach(() => vi.useRealTimers());

describe('asking for explanations', () => {
  it('asks one at a time', async () => {
    const order: string[] = [];
    request.mockImplementation(async (_path: string, options: { body: { method: string } }) => {
      order.push(`start ${options.body.method}`);
      await Promise.resolve();
      order.push(`end ${options.body.method}`);
      return options.body.method;
    });
    const both = Promise.all([
      explainApi.run('r', { method: 'samples' }),
      explainApi.run('r', { method: 'weights' }),
    ]);
    await expect(both).resolves.toEqual(['samples', 'weights']);
    expect(order).toEqual(['start samples', 'end samples', 'start weights', 'end weights']);
  });

  it('waits while the server is busy with other predictions', async () => {
    const busy = new ApiError(503, null, 'The server is busy with other predictions.');
    request.mockRejectedValueOnce(busy).mockResolvedValueOnce({ method: 'boundary' });
    const answer = explainApi.run('r', { method: 'boundary' });
    await vi.advanceTimersByTimeAsync(3000);
    await expect(answer).resolves.toEqual({ method: 'boundary' });
    expect(request).toHaveBeenCalledTimes(2);

    request.mockRejectedValueOnce(new ApiError(503, null, 'Training is disabled on this server.'));
    await expect(explainApi.run('r', { method: 'boundary' })).rejects.toThrow('disabled');
  });

  it('names the sample', () => {
    expect(sampleFields({ kind: 'validation', index: 3 })).toEqual({ sample: 3 });
    expect(sampleFields({ kind: 'input', input: [1, 2], label: 'row' })).toEqual({
      input: [1, 2],
      input_scale: 'raw',
    });
    expect(sampleFields({ kind: 'random' })).toEqual({});
  });
});
