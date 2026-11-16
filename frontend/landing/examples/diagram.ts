/**
 * The diagram of an example: an inline SVG made at build time, laid out like the example in the
 * editor (the positions of the graph, layers as boxes, connections from right to left). Its
 * colours are classes of landing.css, so it follows the light and dark scheme and needs no
 * inline styles (the pages' Content-Security-Policy allows none).
 */
import type { ExampleData, ExampleEdge, ExampleNode } from './data.ts';

/** Texts of a diagram in the page's language. */
export interface DiagramLabels {
  /** The output of a node that is not a tensor shape. */
  kernel: string;
  model: string;
  classes: string;
  /** "1,234 parameters" (neural layers) or "3 hyperparameters" (kernels). */
  params: (count: number) => string;
  hyperparameters: (count: number) => string;
}

const WIDTH = 220;
const LINE = 26;
const PAD = 14;
/** A layer's box: its title lines, the output and possibly the parameters. */
const TITLE_FONT = 22;
const TEXT_FONT = 19;
/** Rendered at this share of the editor's size (14 px titles); wider diagrams scroll. */
export const DIAGRAM_SCALE = 0.62;
/** Corridors of connections that go around boxes keep this distance from them. */
const GAP = 24;

interface Box {
  node: ExampleNode;
  x: number;
  y: number;
  height: number;
  title: string[];
  lines: string[];
}

/** The size of a number for people: `1234567` → `1,234,567` (`.` for German). */
export function formatCount(value: number, lang: 'en' | 'de'): string {
  return value.toLocaleString(lang === 'de' ? 'de-DE' : 'en-US');
}

/** A shape without the batch dimension, e.g. `32 × 28 × 28`. */
export function formatShape(shape: number[]): string {
  return shape.join(' × ');
}

/**
 * What flows out of a node, as the editor shows it: a shape, a kernel, a model (into an
 * ensemble: `feedsEnsemble`) or the classes of a classifier.
 */
export function outputText(
  node: ExampleNode,
  labels: DiagramLabels,
  feedsEnsemble: boolean
): string {
  if (node.output_kind === 'kernel') {
    return labels.kernel;
  }
  if (node.output_kind === 'estimator' && feedsEnsemble) {
    return labels.model;
  }
  return node.output_shape ? formatShape(node.output_shape) : labels.classes;
}

/** Splits a long type name at its capitals (`HistGradientBoostingClassifier` → 2 lines). */
export function titleLines(type: string, maxChars = 16): string[] {
  if (type.length <= maxChars) {
    return [type];
  }
  const words = type.match(/[A-Z][a-z0-9]*|[a-z0-9]+|[A-Z]+(?![a-z])/g) ?? [type];
  const lines: string[] = [];
  let current = '';
  for (const word of words) {
    if (current && (current + word).length > maxChars) {
      lines.push(current);
      current = word;
    } else {
      current += word;
    }
  }
  lines.push(current);
  return lines;
}

function escapeXml(text: string): string {
  return text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function boxes(example: ExampleData, labels: DiagramLabels): Box[] {
  const sources = new Set(example.edges.map((edge) => edge.source));
  return example.nodes.map((node) => {
    const title = titleLines(node.type);
    const lines = [outputText(node, labels, sources.has(node.id))];
    if (node.num_params) {
      lines.push(
        node.output_kind === 'tensor'
          ? labels.params(node.num_params)
          : labels.hyperparameters(node.num_params)
      );
    }
    const height = PAD * 2 + (title.length + lines.length) * LINE - 4;
    return { node, x: node.position.x, y: node.position.y, height, title, lines };
  });
}

/** The y of the connection into `box` for its `index`th of `count` incoming connections. */
function entryY(box: Box, index: number, count: number): number {
  if (count <= 1) {
    return box.y + box.height / 2;
  }
  return box.y + box.height * (0.3 + (0.45 * index) / (count - 1));
}

/** Free horizontal corridors between `top` and `bottom` in the columns `left` to `right`. */
function laneBetween(
  all: Box[],
  from: number,
  to: number,
  left: number,
  right: number,
  nearTo: boolean
): number {
  const low = Math.min(from, to);
  const high = Math.max(from, to);
  const blocked = all
    .filter((b) => b.x < right && b.x + WIDTH > left && b.y + b.height > low && b.y < high)
    .map((b) => [b.y, b.y + b.height] as const)
    .sort((a, b) => a[0] - b[0]);
  const gaps: [number, number][] = [];
  let cursor = low;
  for (const [top, bottom] of blocked) {
    if (top > cursor) {
      gaps.push([cursor, top]);
    }
    cursor = Math.max(cursor, bottom);
  }
  if (high > cursor) {
    gaps.push([cursor, high]);
  }
  const usable = gaps.filter(([a, b]) => b - a >= GAP);
  if (!usable.length) {
    return (low + high) / 2;
  }
  // The corridor next to the target, so that a connection runs along its row.
  const pick = to > from === nearTo ? usable[usable.length - 1] : usable[0];
  return (pick[0] + pick[1]) / 2;
}

function edgePath(
  source: Box,
  target: Box,
  y1: number,
  y2: number,
  all: Box[],
  lanes: Map<number, number>
): { d: string; points: [number, number][] } {
  const x1 = source.x + WIDTH;
  const x2 = target.x;
  const between = all.filter(
    (b) =>
      b !== source &&
      b !== target &&
      b.x < x2 &&
      b.x + WIDTH > x1 &&
      b.y < Math.max(y1, y2) + 4 &&
      b.y + b.height > Math.min(y1, y2) - 4
  );
  if (x2 - x1 >= GAP * 2 && !between.length) {
    // Forward and free: a curve like the editor's.
    const bend = Math.max(GAP * 2, (x2 - x1) / 2);
    return {
      d: `M${x1} ${y1}C${x1 + bend} ${y1} ${x2 - bend} ${y2} ${x2} ${y2}`,
      points: [
        [x1, y1],
        [x2, y2],
      ],
    };
  }
  // Around the boxes: out to the right, along a free corridor, in from the left.
  let lane: number;
  if (x2 - x1 >= GAP * 2) {
    // Forward along a row that has boxes in between (a skip connection): above both rows.
    lane = Math.min(source.y, target.y) - GAP / 2;
  } else {
    lane = laneBetween(all, y1, y2, Math.min(x1, x2) - GAP, Math.max(x1, x2) + GAP, true);
    if (Math.abs(y2 - y1) < 4) {
      lane = Math.max(source.y + source.height, target.y + target.height) + GAP / 2;
    }
  }
  const key = Math.round(lane);
  const used = lanes.get(key) ?? 0;
  lanes.set(key, used + 1);
  lane += used * 8;
  const out = x1 + GAP / 2 + used * 6;
  const into = x2 - GAP / 2 - used * 6;
  const points: [number, number][] = [
    [x1, y1],
    [out, y1],
    [out, lane],
    [into, lane],
    [into, y2],
    [x2, y2],
  ];
  return { d: `M${points.map(([x, y]) => `${x} ${y}`).join('L')}`, points };
}

/**
 * The SVG of an example. `title` and `description` name it for screen readers (the layer
 * table next to it has the same facts).
 */
export function diagramSvg(
  example: ExampleData,
  labels: DiagramLabels,
  title: string,
  description: string,
  idPrefix = 'diagram'
): { svg: string; width: number; height: number } {
  const all = boxes(example, labels);
  const byId = new Map(all.map((box) => [box.node.id, box]));
  const incoming = new Map<string, ExampleEdge[]>();
  for (const edge of example.edges) {
    incoming.set(edge.target, [...(incoming.get(edge.target) ?? []), edge]);
  }
  // Data first, then kernels; from top to bottom, as the sources are.
  for (const list of incoming.values()) {
    const rank = (edge: ExampleEdge) => (edge.kind === 'kernel' ? 1 : 0);
    list.sort(
      (a, b) =>
        rank(a) - rank(b) ||
        byId.get(a.source)!.y - byId.get(b.source)!.y ||
        byId.get(a.source)!.x - byId.get(b.source)!.x
    );
  }
  const lanes = new Map<number, number>();
  const paths: string[] = [];
  const xs: number[] = [];
  const ys: number[] = [];
  for (const box of all) {
    xs.push(box.x, box.x + WIDTH);
    ys.push(box.y, box.y + box.height);
  }
  for (const edge of example.edges) {
    const source = byId.get(edge.source)!;
    const target = byId.get(edge.target)!;
    const list = incoming.get(edge.target)!;
    const y1 = source.y + source.height / 2;
    const y2 = entryY(target, list.indexOf(edge), list.length);
    const { d, points } = edgePath(source, target, y1, y2, all, lanes);
    for (const [x, y] of points) {
      xs.push(x);
      ys.push(y);
    }
    paths.push(`<path class="edge ${edge.kind}" d="${d}" marker-end="url(#${idPrefix}-arrow)"/>`);
  }
  const margin = 12;
  const minX = Math.min(...xs) - margin;
  const minY = Math.min(...ys) - margin;
  const width = Math.max(...xs) + margin - minX;
  const height = Math.max(...ys) + margin - minY;

  const nodes = all.map((box) => {
    const cat = box.node.category.toLowerCase().replace(/[^a-z]+/g, '-');
    const lines: string[] = [];
    let y = box.y + PAD + TITLE_FONT - 2;
    for (const line of box.title) {
      lines.push(
        `<text class="title" x="${box.x + 18}" y="${y}" font-size="${TITLE_FONT}">${escapeXml(line)}</text>`
      );
      y += LINE;
    }
    for (const [index, line] of box.lines.entries()) {
      lines.push(
        `<text class="${index === 0 ? 'shape' : 'meta'}" x="${box.x + 18}" y="${y}" font-size="${TEXT_FONT}">${escapeXml(line)}</text>`
      );
      y += LINE;
    }
    return `<g class="node cat-${cat}"><rect class="box" x="${box.x}" y="${box.y}" width="${WIDTH}" height="${box.height}" rx="10"/><rect class="stripe" x="${box.x}" y="${box.y}" width="7" height="${box.height}" rx="3"/>${lines.join('')}</g>`;
  });

  const svgWidth = Math.round(width * DIAGRAM_SCALE);
  const svgHeight = Math.round(height * DIAGRAM_SCALE);
  const svg = `<svg class="network" xmlns="http://www.w3.org/2000/svg" viewBox="${minX} ${minY} ${width} ${height}" width="${svgWidth}" height="${svgHeight}" role="img" aria-labelledby="${idPrefix}-title ${idPrefix}-desc"><title id="${idPrefix}-title">${escapeXml(title)}</title><desc id="${idPrefix}-desc">${escapeXml(description)}</desc><defs><marker id="${idPrefix}-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0 0L10 5L0 10z"/></marker></defs>${paths.join('')}${nodes.join('')}</svg>`;
  return { svg, width: svgWidth, height: svgHeight };
}

/** A text version of the graph for the diagram's description: "Input (784) → Linear (256) → …". */
export function describeGraph(example: ExampleData, labels: DiagramLabels): string {
  return boxes(example, labels)
    .map((box) => `${box.node.type} (${box.lines[0]})`)
    .join(example.edges.length === example.nodes.length - 1 ? ' → ' : ', ');
}
