/**
 * Web worker of the online demo. It runs the `brickdnn` Python package with Pyodide and
 * answers the public code generation requests (`brickdnn.service`), so no server is needed.
 */

const PYODIDE_VERSION = '314.0.7';
const PYODIDE_URL = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;
const PACKAGE_DIR = '/brickdnn-src';

// The package's source files, bundled at build time. The training engine needs PyTorch and
// the command line tool is not needed, so both are left out.
const sources = import.meta.glob<string>(
  [
    '../../../backend/brickdnn/**/*.{py,json}',
    '!../../../backend/brickdnn/training/**',
    '!../../../backend/brickdnn/__main__.py',
  ],
  { query: '?raw', import: 'default', eager: true }
);

const HANDLER = `
import json
from brickdnn import service

HANDLERS = {
    "layers": lambda data: (200, service.layers()),
    "examples": lambda data: (200, service.examples()),
    "analyze": lambda data: (200, service.analyze_graph(data)),
    "generate": service.generate,
}

def handle(op, body):
    status, payload = HANDLERS[op](json.loads(body) if body else None)
    return json.dumps({"status": status, "data": payload})
`;

export type EngineOp = 'layers' | 'examples' | 'analyze' | 'generate';

export interface EngineRequest {
  id: number;
  op: EngineOp;
  /** The request body as JSON. */
  body: string | null;
}

/** An HTTP-like answer, or `error` if Pyodide could not be loaded. */
export type EngineResponse =
  { id: number; status: number; data: unknown } | { id: number; error: string };

interface Pyodide {
  FS: {
    mkdirTree: (path: string) => void;
    writeFile: (path: string, data: string) => void;
  };
  runPython: (code: string) => unknown;
  globals: { get: (name: string) => unknown };
}

type Handle = (op: EngineOp, body: string | null) => string;

let ready: Promise<Handle> | null = null;

async function start(): Promise<Handle> {
  const { loadPyodide } = (await import(/* @vite-ignore */ `${PYODIDE_URL}pyodide.mjs`)) as {
    loadPyodide: (options: { indexURL: string }) => Promise<Pyodide>;
  };
  const pyodide = await loadPyodide({ indexURL: PYODIDE_URL });
  for (const [path, source] of Object.entries(sources)) {
    const target = `${PACKAGE_DIR}/${path.slice(path.indexOf('brickdnn/'))}`;
    pyodide.FS.mkdirTree(target.slice(0, target.lastIndexOf('/')));
    pyodide.FS.writeFile(target, source);
  }
  pyodide.runPython(`import sys\nsys.path.insert(0, "${PACKAGE_DIR}")\n${HANDLER}`);
  return pyodide.globals.get('handle') as Handle;
}

async function answer({ id, op, body }: EngineRequest): Promise<EngineResponse> {
  let handle: Handle;
  try {
    ready ??= start();
    handle = await ready;
  } catch (error) {
    ready = null; // try again with the next request
    return { id, error: String(error) };
  }
  try {
    const { status, data } = JSON.parse(handle(op, body)) as { status: number; data: unknown };
    return { id, status, data };
  } catch (error) {
    return { id, status: 500, data: { detail: `Internal error: ${String(error)}` } };
  }
}

self.onmessage = (event: MessageEvent<EngineRequest>) => {
  void answer(event.data).then((response) => self.postMessage(response));
};
