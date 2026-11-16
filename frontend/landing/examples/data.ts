/**
 * The built-in examples as the example pages show them: examples.json, made by the backend
 * (`python -m brickdnn.examples.export`, backend/brickdnn/examples/export.py) from the example
 * graphs, the shape inference and the code generator, and committed, because the frontend is
 * built without the backend. The backend's tests and landing.test.ts check that it is current.
 */
import { readFileSync } from 'node:fs';
import { join } from 'node:path';

export const EXAMPLES_FILE = join(import.meta.dirname, 'examples.json');

export type OutputKind = 'tensor' | 'kernel' | 'estimator';

export interface ExampleNode {
  id: string;
  type: string;
  /** The layer category of the palette (its colour in the editor). */
  category: string;
  /** The parameters the example sets; the others are the layer's defaults. */
  params: Record<string, unknown>;
  position: { x: number; y: number };
  /** Without the batch dimension; null for kernels and classifiers (one class per row). */
  output_shape: number[] | null;
  output_kind: OutputKind;
  /** Trainable parameters, or a Gaussian process kernel's hyperparameters; null: none. */
  num_params: number | null;
  task: 'classification' | 'regression' | null;
  /** The scikit-learn kernel built up to this brick. */
  kernel: string | null;
}

export interface ExampleEdge {
  source: string;
  target: string;
  /** What the connection carries: data, a kernel or a model (into an ensemble). */
  kind: OutputKind;
}

export interface ExampleData {
  /** The id of the example in the app (`/app?example=<id>`). */
  id: string;
  /** Name and description in the app's Examples menu (English). */
  name: string;
  description: string;
  family: 'neural' | 'classical' | 'hybrid';
  framework: 'pytorch' | 'scikit-learn' | 'pytorch+scikit-learn';
  task: 'classification' | 'regression' | null;
  estimator: string | null;
  total_params: number | null;
  neural_params: number | null;
  inputs: string[];
  outputs: string[];
  nodes: ExampleNode[];
  edges: ExampleEdge[];
  /** The script of the editor's code dialog with its default options. */
  code: string;
}

export function readExamples(file = EXAMPLES_FILE): ExampleData[] {
  return (JSON.parse(readFileSync(file, 'utf8')) as { examples: ExampleData[] }).examples;
}

/** All examples, in the order of the app's Examples menu. */
export const EXAMPLES: ExampleData[] = readExamples();

export function exampleById(id: string): ExampleData {
  const example = EXAMPLES.find((candidate) => candidate.id === id);
  if (!example) {
    throw new Error(`There is no example “${id}” in examples.json`);
  }
  return example;
}

/** The part of an example page's address: `cnn_mnist` → `cnn-mnist`. */
export function exampleSlug(id: string): string {
  return id.replace(/_/g, '-');
}

/** The trainable parameters of the neural part: all of a neural network, none of a classical model. */
export function trainableParams(example: ExampleData): number | null {
  if (example.family === 'neural') {
    return example.total_params;
  }
  return example.neural_params;
}
