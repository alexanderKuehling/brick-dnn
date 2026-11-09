import { FakeWebSocket } from '@test-utils';
import { afterEach, beforeEach, vi } from 'vitest';
import { LiveClient, type LiveClientOptions } from './LiveClient';
import type { LiveOp, ServerMessage } from './protocol';

const GRAPH = { version: 1 as const, nodes: [], edges: [] };
const ALEX = { id: 1, username: 'alex' };
const MARIA = { id: 2, username: 'maria' };

function hello(over: { version?: number; client_id?: string; role?: string } = {}) {
  return {
    type: 'hello',
    project: { id: 5, name: 'MLP', description: '', graph: GRAPH, version: over.version ?? 3 },
    role: over.role ?? 'editor',
    client_id: over.client_id ?? 'me',
    user: ALEX,
  };
}

const update = (id: string, params: Record<string, number>): LiveOp => ({
  op: 'update_node',
  id,
  set: { params },
});

let getToken: ReturnType<typeof vi.fn<LiveClientOptions['getToken']>>;

function setup(options: Partial<LiveClientOptions> = {}) {
  const client = new LiveClient({
    url: 'ws://test/api/ws/projects/5/',
    getToken,
    createSocket: (url) => new FakeWebSocket(url),
    backoff: (attempt) => 1000 * (attempt + 1),
    ...options,
  });
  const received: ServerMessage[] = [];
  client.subscribe((message) => received.push(message));
  client.start();
  return { client, received };
}

/** Let the token promise resolve and open the socket. */
async function connect(message = hello()) {
  await vi.advanceTimersByTimeAsync(0);
  const socket = FakeWebSocket.last;
  socket.open();
  socket.receive(message);
  return socket;
}

beforeEach(() => {
  vi.useFakeTimers();
  FakeWebSocket.reset();
  getToken = vi.fn<LiveClientOptions['getToken']>().mockResolvedValue('token-1');
});

afterEach(() => {
  vi.useRealTimers();
});

describe('LiveClient', () => {
  it('authenticates first and is live after hello', async () => {
    const { client, received } = setup();
    expect(client.status).toBe('connecting');
    await vi.advanceTimersByTimeAsync(0);
    const socket = FakeWebSocket.last;
    expect(socket.url).toBe('ws://test/api/ws/projects/5/');
    socket.open();
    expect(socket.sent).toEqual([{ type: 'auth', token: 'token-1' }]);

    socket.receive(hello());
    expect(client.status).toBe('live');
    expect(client.clientId).toBe('me');
    expect(client.role).toBe('editor');
    expect(client.version).toBe(3);
    expect(received.map((m) => m.type)).toEqual(['hello']);
  });

  it('sends ops in batches and merges changes of the same layer', async () => {
    const { client } = setup();
    const socket = await connect();
    client.sendOps([update('fc', { out_features: 1 })]);
    client.sendOps([update('fc', { out_features: 12 }), update('in', { size: 3 })]);
    client.sendOps([update('fc', { bias: 0 })]);
    expect(socket.messages('ops')).toHaveLength(0);

    await vi.advanceTimersByTimeAsync(120);
    const [message] = socket.messages('ops');
    expect(message.ops).toEqual([
      update('fc', { out_features: 12, bias: 0 }),
      update('in', { size: 3 }),
    ]);
    expect(client.pendingCount).toBe(1);
  });

  it('takes its own echo as confirmation and passes the ops of others on', async () => {
    const { client, received } = setup();
    const socket = await connect();
    client.sendOps([update('fc', { out_features: 12 })]);
    client.flush();
    const { id, ops } = socket.messages('ops')[0];

    const theirs = {
      type: 'ops',
      ops: [update('in', { size: 4 })],
      version: 4,
      by: MARIA,
      client_id: 'other',
      id: 'x',
    };
    socket.receive(theirs);
    expect(client.pendingCount).toBe(1);
    socket.receive({ type: 'ops', ops, version: 5, by: ALEX, client_id: 'me', id });
    expect(client.pendingCount).toBe(0);
    expect(client.version).toBe(5);

    const echoes = received.filter((m) => m.type === 'ops');
    expect(echoes.map((m) => client.isOwn(m as { client_id: string }))).toEqual([false, true]);
  });

  it('reconnects after a drop with a fresh token and sends unconfirmed ops again', async () => {
    const { client, received } = setup();
    const first = await connect();
    client.sendOps([update('fc', { out_features: 12 })]);
    client.flush(); // Sent, but the confirmation never arrives.

    first.drop(1006);
    expect(client.status).toBe('offline');
    client.sendOps([update('fc', { bias: 0 })]); // Changed while offline.
    await vi.advanceTimersByTimeAsync(120);
    expect(client.pendingOps()).toEqual([
      update('fc', { out_features: 12 }),
      update('fc', { bias: 0 }),
    ]);

    // Backoff (1 s here): no new connection before the delay.
    await vi.advanceTimersByTimeAsync(800);
    expect(FakeWebSocket.instances).toHaveLength(1);
    getToken.mockResolvedValue('token-2');
    await vi.advanceTimersByTimeAsync(100);
    expect(FakeWebSocket.instances).toHaveLength(2);
    expect(getToken).toHaveBeenLastCalledWith(false);

    const second = FakeWebSocket.last;
    second.open();
    expect(second.sent[0]).toEqual({ type: 'auth', token: 'token-2' });
    let pendingAtHello: LiveOp[] = [];
    client.subscribe((message) => {
      if (message.type === 'hello') {
        pendingAtHello = client.pendingOps();
      }
    });
    second.receive(hello({ client_id: 'me-2', version: 7 }));
    // Subscribers see the pending ops at hello (to put them on top of the server state) ...
    expect(pendingAtHello).toHaveLength(2);
    // ... then they go out again, merged into one message.
    const [resent] = second.messages('ops');
    expect(resent.ops).toEqual([update('fc', { out_features: 12, bias: 0 })]);
    expect(client.status).toBe('live');
    expect(received.filter((m) => m.type === 'hello')).toHaveLength(2);

    second.receive({
      type: 'ops',
      ops: resent.ops,
      version: 8,
      by: ALEX,
      client_id: 'me-2',
      id: resent.id,
    });
    expect(client.pendingCount).toBe(0);
  });

  it('keeps ops from the draft and sends them after hello', async () => {
    const { client } = setup({ pending: [update('fc', { out_features: 3 })] });
    expect(client.pendingOps()).toHaveLength(1);
    const socket = await connect();
    expect(socket.messages('ops')[0].ops).toEqual([update('fc', { out_features: 3 })]);
  });

  it('tracks peers and throttles presence', async () => {
    const { client } = setup();
    const socket = await connect();
    socket.receive({ type: 'joined', client_id: 'p1', user: MARIA });
    expect(client.peers).toEqual([{ clientId: 'p1', user: MARIA, state: {} }]);
    socket.receive({ type: 'presence', client_id: 'p1', user: MARIA, state: { selected: ['fc'] } });
    expect(client.peers[0].state).toEqual({ selected: ['fc'] });

    client.sendPresence({ selected: ['a'] });
    client.sendPresence({ selected: ['b'] });
    client.sendPresence({ selected: ['c'] });
    expect(socket.messages('presence')).toEqual([{ type: 'presence', state: { selected: ['a'] } }]);
    await vi.advanceTimersByTimeAsync(100);
    expect(socket.messages('presence')).toHaveLength(2);
    expect(socket.messages('presence')[1]).toEqual({
      type: 'presence',
      state: { selected: ['c'] },
    });

    // A newcomer learns what we have selected.
    await vi.advanceTimersByTimeAsync(200);
    socket.receive({ type: 'joined', client_id: 'p2', user: MARIA });
    expect(socket.messages('presence')).toHaveLength(3);

    socket.receive({ type: 'left', client_id: 'p1' });
    expect(client.peers.map((peer) => peer.clientId)).toEqual(['p2']);
    socket.drop();
    expect(client.peers).toEqual([]);
  });

  it('pings while connected', async () => {
    setup();
    const socket = await connect();
    await vi.advanceTimersByTimeAsync(25000);
    expect(socket.messages('ping')).toHaveLength(1);
  });

  it('gets the state again when it missed a change, and ignores old ones', async () => {
    const { client, received } = setup();
    const socket = await connect(hello({ version: 3 }));
    const ops = (version: number) => ({
      type: 'ops',
      ops: [],
      version,
      by: MARIA,
      client_id: 'other',
      id: null,
    });
    socket.receive(ops(3));
    expect(received.filter((m) => m.type === 'ops')).toHaveLength(0);
    socket.receive(ops(5));
    expect(socket.closedByClient).toBe(1000);
    await vi.advanceTimersByTimeAsync(0);
    expect(FakeWebSocket.instances).toHaveLength(2);
    expect(client.status).toBe('offline');
  });

  it('forgets rejected ops and gets the state again', async () => {
    const { client, received } = setup();
    const socket = await connect();
    client.sendOps([update('gone', { out_features: 3 })]);
    client.flush();
    const { id } = socket.messages('ops')[0];
    socket.receive({ type: 'rejected', id, version: 3 });
    expect(client.pendingCount).toBe(0);
    expect(received.at(-1)?.type).toBe('rejected');
    await vi.advanceTimersByTimeAsync(0);
    expect(FakeWebSocket.instances).toHaveLength(2);
  });

  it('refreshes the token once when it is not accepted', async () => {
    const { client } = setup();
    await vi.advanceTimersByTimeAsync(0);
    FakeWebSocket.last.open();
    FakeWebSocket.last.drop(4401);
    await vi.advanceTimersByTimeAsync(0);
    expect(getToken).toHaveBeenLastCalledWith(true);
    expect(FakeWebSocket.instances).toHaveLength(2);
    FakeWebSocket.last.open();
    FakeWebSocket.last.drop(4401);
    expect(client.status).toBe('ended');
    expect(client.endReason).toBe('unauthenticated');
  });

  it('ends without a token', async () => {
    getToken.mockResolvedValue(null);
    const { client } = setup();
    await vi.advanceTimersByTimeAsync(0);
    expect(FakeWebSocket.instances).toHaveLength(0);
    expect(client.endReason).toBe('unauthenticated');
  });

  it('ends when access is removed or the project is deleted', async () => {
    const first = setup();
    (await connect()).drop(4403);
    expect(first.client.endReason).toBe('forbidden');

    const second = setup();
    const socket = await connect();
    socket.receive({ type: 'deleted' });
    socket.drop(4404);
    expect(second.client.status).toBe('ended');
    expect(second.client.endReason).toBe('deleted');
    await vi.advanceTimersByTimeAsync(60000);
    expect(FakeWebSocket.instances).toHaveLength(2); // No reconnects.
  });

  it('sends what it collected when stopped', async () => {
    const { client } = setup();
    const socket = await connect();
    client.sendOps([update('fc', { out_features: 3 })]);
    client.stop();
    expect(socket.messages('ops')).toHaveLength(1);
    expect(socket.closedByClient).toBe(1000);
    expect(client.status).toBe('ended');
  });
});
