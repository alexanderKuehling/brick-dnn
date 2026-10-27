export type ParamKind = 'int' | 'float' | 'bool' | 'select' | 'ints';
export type ParamValue = number | boolean | string | number[] | null;
export type Params = Record<string, ParamValue>;

export interface ParamSpec {
  name: string;
  label: string;
  kind: ParamKind;
  default: ParamValue;
  help: string;
  options: string[];
  min: number | null;
  max: number | null;
  /** If true the value may be left empty and is inferred from the input shape. */
  auto: boolean;
  /** Fixed number of values for `ints` params (e.g. 2 for a 2-D kernel size). */
  length: number | null;
  show_if: { param: string; value: ParamValue } | null;
}

export interface LayerSpec {
  type: string;
  label: string;
  category: string;
  description: string;
  params: ParamSpec[];
  min_inputs: number;
  max_inputs: number | null;
}

export interface LayerCatalog {
  categories: string[];
  layers: LayerSpec[];
}

export interface GraphNode {
  id: string;
  type: string;
  params: Params;
  name?: string;
  position?: { x: number; y: number };
}

export interface GraphEdge {
  id?: string;
  source: string;
  target: string;
}

export interface GraphDocument {
  version: 1;
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export interface Issue {
  node: string | null;
  message: string;
}

export interface NodeAnalysis {
  type: string;
  output_shape: number[] | null;
  dtype: 'float' | 'long' | null;
  num_params: number | null;
  params: Params | null;
}

export interface Analysis {
  valid: boolean;
  errors: Issue[];
  warnings: Issue[];
  nodes: Record<string, NodeAnalysis>;
  inputs: string[];
  outputs: string[];
  total_params: number | null;
}

export interface Example {
  id: string;
  name: string;
  description: string;
  order: number;
  graph: GraphDocument;
}

export interface GenerateOptions {
  class_name?: string;
  include_main?: boolean;
  include_training?: boolean;
  filename?: string;
}

export interface GenerateResult {
  code: string;
  filename: string;
}

export interface User {
  id: number;
  username: string;
  email: string;
  date_joined: string;
}

export interface Tokens {
  access: string;
  refresh: string;
}

export interface Project {
  id: number;
  name: string;
  description: string;
  graph: GraphDocument;
  num_layers: number;
  created_at: string;
  updated_at: string;
}
