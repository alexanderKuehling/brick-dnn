/**
 * The connection to the live editing server for one project, without React.
 *
 * It authenticates, keeps the connection alive (pings, reconnects with backoff and a fresh
 * token after network drops), batches ops, keeps them until the server has confirmed them
 * (and sends them again after a reconnect), throttles presence and keeps the list of peers.
 * Messages from the server are passed on to subscribers.
 */
import type { ProjectRole, UserRef } from '@/api/types';
import { appendOps } from './ops';
import {
  CLOSE_FORBIDDEN,
  CLOSE_NOT_FOUND,
  CLOSE_TOO_FAST,
  CLOSE_UNAUTHORIZED,
  MAX_OPS_PER_MESSAGE,
  type ClientMessage,
  type LiveOp,
  type Peer,
  type PresenceState,
  type ServerMessage,
} from './protocol';

/**
 * connecting: before the first `hello` · live: connected · offline: the connection dropped,
 * reconnecting · ended: for good (see `endReason`).
 */
export type LiveStatus = 'connecting' | 'live' | 'offline' | 'ended';

export type EndReason =
  | 'stopped'
  | 'unauthenticated'
  /** Access was removed. */
  | 'forbidden'
  /** No such project, or no access (at connect time). */
  | 'not_found'
  | 'deleted';

/** The part of the browser's WebSocket the client uses (tests pass a fake). */
export interface SocketLike {
  readonly readyState: number;
  send: (data: string) => void;
  close: (code?: number, reason?: string) => void;
  onopen: ((event: Event) => void) | null;
  onmessage: ((event: MessageEvent) => void) | null;
  onclose: ((event: CloseEvent) => void) | null;
  onerror: ((event: Event) => void) | null;
}

export interface LiveClientOptions {
  url: string;
  /** A valid access token; `force`: refresh it even if it looks valid (after a 4401). */
  getToken: (force: boolean) => Promise<string | null>;
  createSocket?: (url: string) => SocketLike;
  /**
   * At most one ops message per this time: a change goes out at once, the ones that follow
   * within this time are collected into the next message.
   */
  batchMs?: number;
  /** At most one presence message per this time. */
  presenceMs?: number;
  pingMs?: number;
  /** Delay before reconnect attempt `attempt` (0, 1, …). */
  backoff?: (attempt: number) => number;
  /** Ops that were not confirmed before the page was closed (from the draft). */
  pending?: LiveOp[];
}

interface PendingMessage {
  id: string;
  ops: LiveOp[];
}

const OPEN = 1;

function defaultBackoff(attempt: number): number {
  const base = Math.min(15000, 500 * 2 ** attempt);
  return base * (0.75 + Math.random() * 0.5);
}

let messageCounter = 0;
function messageId(): string {
  messageCounter += 1;
  return `${Date.now().toString(36)}-${messageCounter.toString(36)}`;
}

export class LiveClient {
  status: LiveStatus = 'connecting';
  endReason: EndReason | null = null;
  /** Ours, from `hello` (changes with every connection). */
  clientId: string | null = null;
  user: UserRef | null = null;
  role: ProjectRole | null = null;
  /** The last project version we know of. */
  version: number | null = null;
  peers: Peer[] = [];

  private readonly options: Required<Omit<LiveClientOptions, 'pending'>>;
  private socket: SocketLike | null = null;
  /** Ops collected for the next message. */
  private outbox: LiveOp[];
  /** Messages the server has not confirmed (sent, or waiting for the connection). */
  private sent: PendingMessage[] = [];
  private presence: PresenceState | null = null;
  private presenceSentAt = 0;
  private flushedAt = 0;
  private attempt = 0;
  private forceToken = false;
  private forceTokenTried = false;
  private stopped = false;
  private everLive = false;
  private batchTimer: ReturnType<typeof setTimeout> | null = null;
  private presenceTimer: ReturnType<typeof setTimeout> | null = null;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private pingTimer: ReturnType<typeof setInterval> | null = null;
  private readonly messageListeners = new Set<(message: ServerMessage) => void>();
  private readonly changeListeners = new Set<() => void>();

  constructor(options: LiveClientOptions) {
    this.options = {
      createSocket: (url) => new WebSocket(url),
      batchMs: 120,
      presenceMs: 100,
      pingMs: 25000,
      backoff: defaultBackoff,
      ...options,
    };
    this.outbox = options.pending ? appendOps([], options.pending) : [];
  }

  // ------------------------------------------------------------ public API

  start(): void {
    if (typeof window !== 'undefined') {
      window.addEventListener('online', this.onOnline);
    }
    void this.connect();
  }

  /** Close the connection for good. Collected ops are sent first, if possible. */
  stop(): void {
    if (this.status === 'live') {
      this.flush();
    }
    this.end('stopped');
  }

  /** Get the current state again: reconnect, the server answers with `hello`. */
  resync(): void {
    if (this.stopped) {
      return;
    }
    this.dropSocket();
    this.setStatus(this.everLive ? 'offline' : 'connecting');
    void this.connect();
  }

  /** Changes to send. They are applied to the canvas already ("optimistic"). */
  sendOps(ops: LiveOp[]): void {
    if (this.stopped || ops.length === 0) {
      return;
    }
    this.outbox = appendOps(this.outbox, ops);
    if (this.batchTimer === null) {
      const wait = this.flushedAt + this.options.batchMs - Date.now();
      if (wait <= 0) {
        this.flush();
        return;
      }
      this.batchTimer = setTimeout(() => this.flush(), wait);
    }
    this.changed();
  }

  /** Send the ops collected so far now (e.g. when a drag ends, before its presence). */
  flush(): void {
    if (this.batchTimer !== null) {
      clearTimeout(this.batchTimer);
      this.batchTimer = null;
    }
    if (this.outbox.length === 0) {
      return;
    }
    const ops = this.outbox;
    this.outbox = [];
    this.flushedAt = Date.now();
    for (let i = 0; i < ops.length; i += MAX_OPS_PER_MESSAGE) {
      const message = { id: messageId(), ops: ops.slice(i, i + MAX_OPS_PER_MESSAGE) };
      this.sent.push(message);
      if (this.status === 'live') {
        this.send({ type: 'ops', ...message });
      }
    }
    this.changed();
  }

  /** Our selection etc. for the others; at most one message per `presenceMs`. */
  sendPresence(state: PresenceState): void {
    this.presence = state;
    this.schedulePresence();
  }

  /** Ops not confirmed by the server yet, in order (sent ones first). */
  pendingOps(): LiveOp[] {
    return [...this.sent.flatMap((message) => message.ops), ...this.outbox];
  }

  get pendingCount(): number {
    return this.sent.length + (this.outbox.length > 0 ? 1 : 0);
  }

  /** Forget unconfirmed ops (e.g. when the user may only view the project now). */
  dropPending(): void {
    this.sent = [];
    this.outbox = [];
    this.changed();
  }

  /** Ops sent by this client (the server sends them back as confirmation). */
  isOwn(message: { client_id: string }): boolean {
    return message.client_id === this.clientId;
  }

  /** All messages from the server, e.g. for comment or history panels. */
  subscribe(listener: (message: ServerMessage) => void): () => void {
    this.messageListeners.add(listener);
    return () => this.messageListeners.delete(listener);
  }

  /** Status, peers or pending ops changed. */
  onChange(listener: () => void): () => void {
    this.changeListeners.add(listener);
    return () => this.changeListeners.delete(listener);
  }

  // ------------------------------------------------------------ connection

  private readonly onOnline = () => {
    // Back online: do not wait for the backoff.
    if (this.reconnectTimer !== null) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
      void this.connect();
    }
  };

  private async connect(): Promise<void> {
    const force = this.forceToken;
    this.forceToken = false;
    let token: string | null;
    try {
      token = await this.options.getToken(force);
    } catch {
      // The server could not be reached to refresh the token: try again later.
      if (!this.stopped && !this.socket) {
        this.scheduleReconnect();
      }
      return;
    }
    if (this.stopped || this.socket) {
      return;
    }
    if (!token) {
      this.end('unauthenticated');
      return;
    }
    let socket: SocketLike;
    try {
      socket = this.options.createSocket(this.options.url);
    } catch {
      this.scheduleReconnect();
      return;
    }
    this.socket = socket;
    socket.onopen = () => this.send({ type: 'auth', token }, true);
    socket.onmessage = (event) => {
      if (this.socket !== socket) {
        return;
      }
      let message: ServerMessage;
      try {
        message = JSON.parse(String(event.data)) as ServerMessage;
      } catch {
        return;
      }
      this.handle(message);
    };
    socket.onclose = (event) => {
      if (this.socket === socket) {
        this.closed(event.code);
      }
    };
    socket.onerror = () => undefined; // A close event follows.
  }

  private closed(code: number): void {
    this.socket = null;
    this.stopPing();
    this.clearPeers();
    if (this.stopped) {
      return;
    }
    if (code === CLOSE_UNAUTHORIZED) {
      if (!this.forceTokenTried) {
        // The token may have expired just now: refresh it and try once more.
        this.forceTokenTried = true;
        this.forceToken = true;
        this.setStatus(this.everLive ? 'offline' : 'connecting');
        void this.connect();
        return;
      }
      this.end('unauthenticated');
      return;
    }
    if (code === CLOSE_FORBIDDEN) {
      this.end('forbidden');
      return;
    }
    if (code === CLOSE_NOT_FOUND) {
      this.end(this.endReason === 'deleted' ? 'deleted' : 'not_found');
      return;
    }
    this.setStatus(this.everLive ? 'offline' : 'connecting');
    this.scheduleReconnect(code === CLOSE_TOO_FAST ? 5000 : undefined);
  }

  private scheduleReconnect(delay?: number): void {
    if (this.stopped || this.reconnectTimer !== null) {
      return;
    }
    const wait = delay ?? this.options.backoff(this.attempt);
    this.attempt += 1;
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      void this.connect();
    }, wait);
  }

  private dropSocket(): void {
    const socket = this.socket;
    this.socket = null;
    this.stopPing();
    this.clearPeers();
    if (socket) {
      socket.onopen = null;
      socket.onmessage = null;
      socket.onclose = null;
      socket.onerror = null;
      try {
        socket.close(1000);
      } catch {
        // Closing a socket that is not open yet can throw in some browsers.
      }
    }
  }

  private end(reason: EndReason): void {
    if (this.stopped) {
      return;
    }
    this.stopped = true;
    this.endReason = reason;
    for (const timer of [this.batchTimer, this.presenceTimer, this.reconnectTimer]) {
      if (timer !== null) {
        clearTimeout(timer);
      }
    }
    this.batchTimer = null;
    this.presenceTimer = null;
    this.reconnectTimer = null;
    if (typeof window !== 'undefined') {
      window.removeEventListener('online', this.onOnline);
    }
    this.dropSocket();
    this.setStatus('ended');
  }

  private send(message: ClientMessage, beforeHello = false): void {
    const socket = this.socket;
    if (socket && socket.readyState === OPEN && (beforeHello || this.status === 'live')) {
      socket.send(JSON.stringify(message));
    }
  }

  private stopPing(): void {
    if (this.pingTimer !== null) {
      clearInterval(this.pingTimer);
      this.pingTimer = null;
    }
  }

  // ------------------------------------------------------------ messages

  private handle(message: ServerMessage): void {
    switch (message.type) {
      case 'hello':
        this.hello(message);
        return;
      case 'ops':
        if (this.version !== null && message.version <= this.version) {
          return; // Included in the state we have already.
        }
        if (this.version !== null && message.version > this.version + 1) {
          // We missed a change: get the whole state again.
          this.resync();
          return;
        }
        this.version = message.version;
        if (this.isOwn(message)) {
          this.sent = this.sent.filter((pending) => pending.id !== message.id);
          this.changed();
        }
        break;
      case 'rejected':
      case 'error':
        if (message.id && this.sent.some((pending) => pending.id === message.id)) {
          // None of these ops could be applied: forget them and reload the state, which
          // undoes them on the canvas.
          this.sent = this.sent.filter((pending) => pending.id !== message.id);
          this.emit(message);
          this.resync();
          return;
        }
        break;
      case 'replaced':
        if (this.version !== null && message.project.version <= this.version) {
          return;
        }
        this.version = message.project.version;
        break;
      case 'access':
        this.role = message.role;
        break;
      case 'deleted':
        this.endReason = 'deleted';
        break;
      case 'joined':
        if (!this.peers.some((peer) => peer.clientId === message.client_id)) {
          this.peers = [
            ...this.peers,
            { clientId: message.client_id, user: message.user, state: {} },
          ];
          this.changed();
        }
        // Tell the newcomer what we have selected.
        this.schedulePresence();
        break;
      case 'left':
        this.peers = this.peers.filter((peer) => peer.clientId !== message.client_id);
        this.changed();
        break;
      case 'presence': {
        const peer = { clientId: message.client_id, user: message.user, state: message.state };
        const known = this.peers.some((p) => p.clientId === message.client_id);
        this.peers = known
          ? this.peers.map((p) => (p.clientId === message.client_id ? peer : p))
          : [...this.peers, peer];
        this.changed();
        break;
      }
      default:
        break;
    }
    this.emit(message);
  }

  private hello(message: Extract<ServerMessage, { type: 'hello' }>): void {
    this.clientId = message.client_id;
    this.user = message.user;
    this.role = message.role;
    this.version = message.project.version;
    this.attempt = 0;
    this.forceTokenTried = false;
    this.everLive = true;
    this.peers = [];
    this.status = 'live';
    // The confirmations of the old connection are lost: its messages go out again (merged).
    this.outbox = appendOps([], this.pendingOps());
    this.sent = [];
    // Subscribers apply the pending ops on top of the server state first, then they are sent.
    this.emit(message);
    this.flush();
    this.stopPing();
    this.pingTimer = setInterval(() => this.send({ type: 'ping' }), this.options.pingMs);
    this.presenceSentAt = 0;
    this.schedulePresence();
    this.changed();
  }

  private schedulePresence(): void {
    if (this.presence === null || this.presenceTimer !== null || this.status !== 'live') {
      return;
    }
    const wait = this.presenceSentAt + this.options.presenceMs - Date.now();
    const sendNow = () => {
      this.presenceTimer = null;
      if (this.presence !== null && this.status === 'live') {
        this.presenceSentAt = Date.now();
        this.send({ type: 'presence', state: this.presence });
      }
    };
    if (wait <= 0) {
      sendNow();
    } else {
      this.presenceTimer = setTimeout(sendNow, wait);
    }
  }

  private clearPeers(): void {
    if (this.peers.length > 0) {
      this.peers = [];
      this.changed();
    }
  }

  private setStatus(status: LiveStatus): void {
    if (this.status !== status) {
      this.status = status;
      this.changed();
    }
  }

  private emit(message: ServerMessage): void {
    this.messageListeners.forEach((listener) => listener(message));
  }

  private changed(): void {
    this.changeListeners.forEach((listener) => listener());
  }
}
