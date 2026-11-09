/**
 * What flows along a connection: a tensor (data) or, from a kernel brick, the kernel of a
 * Gaussian process. Bricks say what they put out (`output_kind`) and take (`input_slots`);
 * the canvas draws a handle per slot and refuses connections of the wrong kind. The analysis
 * on the server stays the judge of everything else (see docs/graph-format.md).
 */
import type { Edge } from '@xyflow/react';
import type { Analysis, InputSlot, LayerSpec, NodeAnalysis, OutputKind } from '@/api/types';

/** What a brick takes. Catalogues of older servers only know one tensor input. */
export function inputSlots(spec: LayerSpec | undefined): InputSlot[] {
  if (!spec) {
    return [];
  }
  if (spec.input_slots) {
    return spec.input_slots;
  }
  if (spec.max_inputs === 0) {
    return [];
  }
  return [
    { name: 'input', label: 'Input', kind: 'tensor', min: spec.min_inputs, max: spec.max_inputs },
  ];
}

export function outputKind(spec: LayerSpec | undefined): OutputKind {
  return spec?.output_kind ?? 'tensor';
}

export function isClassicalSpec(spec: LayerSpec | undefined): boolean {
  return spec?.family === 'classical';
}

/** Handle id of a slot: only bricks with several slots name their handles. */
export function slotHandle(slots: InputSlot[], slot: InputSlot): string | undefined {
  return slots.length > 1 ? slot.name : undefined;
}

/**
 * The slot a connection goes into: the one of its handle, or the first one that takes what
 * the source puts out. Null if the brick has no such slot.
 */
export function slotFor(
  slots: InputSlot[],
  kind: OutputKind,
  handle?: string | null
): InputSlot | null {
  if (handle) {
    return slots.find((slot) => slot.name === handle) ?? null;
  }
  return slots.find((slot) => slot.kind === kind) ?? null;
}

/**
 * Why `source` cannot be connected to `target` (at the handle `targetHandle`), or null if
 * it can. Only the kind of the connection is checked here; unknown types are left to the
 * analysis.
 */
export function connectionProblem(
  source: LayerSpec | undefined,
  target: LayerSpec | undefined,
  targetHandle?: string | null
): string | null {
  if (!source || !target) {
    return null;
  }
  const slots = inputSlots(target);
  if (slots.length === 0) {
    return `${target.label} takes no inputs.`;
  }
  const kind = outputKind(source);
  const slot = slotFor(slots, kind, slots.length > 1 ? targetHandle : null);
  if (!slot) {
    return kind === 'kernel'
      ? `${target.label} cannot take a kernel.`
      : `${target.label} only takes kernels.`;
  }
  if (slot.kind !== kind) {
    return kind === 'kernel'
      ? `The ${slot.label.toLowerCase()} input of ${target.label} takes data, not a kernel.`
      : `The ${slot.label.toLowerCase()} input of ${target.label} only takes a kernel.`;
  }
  return null;
}

/** Edges into `target` that use the same slot as a new connection from a `kind` output. */
export function edgesInSlot(
  edges: Edge[],
  target: string,
  slot: InputSlot,
  kindOf: (id: string) => OutputKind
): Edge[] {
  return edges.filter((edge) => edge.target === target && kindOf(edge.source) === slot.kind);
}

/** Whether the analysis says the node puts out a kernel (or its spec does, before it). */
export function isKernelNode(
  spec: LayerSpec | undefined,
  info: NodeAnalysis | undefined | null
): boolean {
  return (info?.output_kind ?? outputKind(spec)) === 'kernel';
}

/** The network's family: classical if it contains kernels or a classical model. */
export function networkFamily(analysis: Analysis | null | undefined): 'neural' | 'classical' {
  return analysis?.family === 'classical' ? 'classical' : 'neural';
}

// --------------------------------------------------------------------------
// Kernels in words
// --------------------------------------------------------------------------

/** Short names of scikit-learn's kernels for the canvas. */
const SHORT_KERNEL_NAMES: Record<string, string> = {
  ConstantKernel: 'Constant',
  WhiteKernel: 'White',
  ExpSineSquared: 'Periodic',
  RationalQuadratic: 'RQ',
  DotProduct: 'Dot',
  Matern: 'Matérn',
};

/**
 * A kernel expression without its arguments, for the canvas:
 * "ConstantKernel(constant_value=1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)"
 * → "Constant × RBF + White".
 */
export function compactKernel(code: string): string {
  // Arguments may contain one level of parentheses (bounds such as (1e-05, 100000.0)).
  return code
    .replace(
      /([A-Za-z_]\w*)\((?:[^()]|\([^()]*\))*\)/g,
      (_, name: string) => SHORT_KERNEL_NAMES[name] ?? name
    )
    .replace(/\s\*\s/g, ' × ');
}

/**
 * The top-level terms of a kernel sum for line breaks: "A * B + C" → ["A * B", "+ C"].
 * Sums inside parentheses (a sum inside a product) stay together.
 */
export function kernelTerms(code: string): string[] {
  const terms: string[] = [];
  let depth = 0;
  let start = 0;
  for (let i = 0; i < code.length; i += 1) {
    const char = code[i];
    if (char === '(') {
      depth += 1;
    } else if (char === ')') {
      depth -= 1;
    } else if (depth === 0 && code.startsWith(' + ', i)) {
      terms.push(code.slice(start, i));
      start = i + 1;
    }
  }
  terms.push(code.slice(start));
  return terms.map((term) => term.trim()).filter(Boolean);
}

/**
 * One line per term of a kernel that a classifier with more than two classes fitted per
 * class ("setosa: …; versicolor: …"), otherwise the kernel itself.
 */
export function kernelLines(code: string): string[] {
  return code.split(/;\s+(?=[^;:()]+:\s)/);
}
