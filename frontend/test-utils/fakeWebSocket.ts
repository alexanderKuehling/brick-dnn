/** A WebSocket for tests: records what is sent and lets the test play the server. */
export class FakeWebSocket {
  static instances: FakeWebSocket[] = [];

  static get last(): FakeWebSocket {
    const socket = FakeWebSocket.instances[FakeWebSocket.instances.length - 1];
    if (!socket) {
      throw new Error('No WebSocket was opened.');
    }
    return socket;
  }

  static reset(): void {
    FakeWebSocket.instances = [];
  }

  readyState = 0;
  /** Messages sent by the client, parsed. */
  sent: Record<string, unknown>[] = [];
  closedByClient: number | null = null;
  onopen: ((event: Event) => void) | null = null;
  onmessage: ((event: MessageEvent) => void) | null = null;
  onclose: ((event: CloseEvent) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;

  constructor(public url: string) {
    FakeWebSocket.instances.push(this);
  }

  send(data: string): void {
    if (this.readyState !== 1) {
      throw new Error('Not open.');
    }
    this.sent.push(JSON.parse(data) as Record<string, unknown>);
  }

  close(code = 1000): void {
    this.closedByClient = code;
    this.readyState = 3;
  }

  // ---------------------------------------------------------------- the server's side

  open(): void {
    this.readyState = 1;
    this.onopen?.(new Event('open'));
  }

  receive(message: object): void {
    this.onmessage?.({ data: JSON.stringify(message) } as MessageEvent);
  }

  /** The server (or the network) closes the connection. */
  drop(code = 1006): void {
    this.readyState = 3;
    this.onclose?.({ code } as CloseEvent);
  }

  /** Sent messages of one type. */
  messages(type: string): Record<string, unknown>[] {
    return this.sent.filter((message) => message.type === type);
  }
}
