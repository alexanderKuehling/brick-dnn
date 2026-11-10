/**
 * The Input layer follows the dataset chosen for training.
 *
 * The training setup checks the open network against its dataset (`useCompatibility`); the
 * server suggests `fixes` for the Input layer and the last layer. They are applied
 *
 * - automatically, once per selection of a dataset (in the training setup, after an upload,
 *   or with "Train with this data" in the Data tab), if the network does not fit and the fixes
 *   only change the Input's shape or data type and the number of outputs. A notification says
 *   what changed and offers to undo it. An empty canvas gets a small network for the data;
 * - with one click otherwise ("Adapt network to …" in the setup, "Set from …" in the
 *   Inspector of the Input layer).
 *
 * Changes go through the builder's normal node updates, so live editing sends them as ops.
 * Viewers of a network get none of this.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { IconArrowBackUp } from '@tabler/icons-react';
import { Button, Group, Text } from '@mantine/core';
import { notifications } from '@mantine/notifications';
import type { Analysis, CompatibilityFix, GraphDocument, LayerSpec, Params } from '@/api/types';
import type { InputFit } from '@/builder/BuilderContext';
import type { LayerFlowNode, LayerNodeData } from '@/builder/graph';
import { canTrain, NO_SCOPE } from './access';
import { canStartTraining, trainingFamily } from './classical';
import {
  describeFixes,
  describeShape,
  isImageSource,
  onlyInputAndOutput,
  sameParams,
  starterNetwork,
} from './fit';
import { DEFAULT_SOURCE, effectiveSource, type EffectiveSource } from './settings';
import { useCompatibility, type CompatibilityState } from './useCompatibility';
import type { TrainingController } from './useTraining';

/** What the training setup gets: the check of the open network and the one-click fix. */
export interface DatasetFitController {
  compatibility: CompatibilityState;
  /** Applies the suggested fixes; null if there are none (or the network cannot be changed). */
  adapt: (() => void) | null;
}

export interface DatasetFitOptions {
  training: TrainingController;
  loggedIn: boolean;
  /** Viewers cannot change the network. */
  readOnly: boolean;
  doc: GraphDocument;
  nodes: LayerFlowNode[];
  analysis: Analysis | null;
  analyzing: boolean;
  specs: Record<string, LayerSpec>;
  /** The canvas shows what it will show (layer types loaded, a first example placed). */
  canvasReady: boolean;
  /** The selected layer, if it is an Input layer. */
  selectedInput: string | null;
  updateNode: (id: string, patch: Partial<LayerNodeData>) => void;
  deleteNode: (id: string) => void;
  /** Put a network on the empty canvas. */
  startNetwork: (doc: GraphDocument) => void;
}

export interface DatasetFit {
  /** For the training setup (null while training cannot be used here). */
  controller: DatasetFitController | null;
  /** For the Inspector of the selected Input layer. */
  inputFit: InputFit | null;
}

let notificationCounter = 0;

/** Says what changed, with Undo. Replaces the notice of an earlier change (only the latest
 * change can be undone: earlier ones may no longer apply to the canvas). */
function showUndo(message: string, undo: () => void) {
  if (notificationCounter > 0) {
    notifications.hide(`dataset-fit-${notificationCounter}`);
  }
  notificationCounter += 1;
  const id = `dataset-fit-${notificationCounter}`;
  notifications.show({
    id,
    color: 'blue',
    autoClose: 10000,
    message: (
      <Group gap="xs" wrap="nowrap" justify="space-between" align="center">
        <Text size="sm">{message}</Text>
        <Button
          size="compact-xs"
          variant="light"
          leftSection={<IconArrowBackUp size={14} />}
          style={{ flexShrink: 0 }}
          onClick={() => {
            notifications.hide(id);
            undo();
          }}
        >
          Undo
        </Button>
      </Group>
    ),
  });
}

export function useDatasetFit(options: DatasetFitOptions): DatasetFit {
  const { training, loggedIn, readOnly, doc, nodes, analysis, analyzing, selectedInput } = options;
  const { status, data } = training;
  const latest = useRef(options);
  latest.current = options;

  const editable =
    loggedIn &&
    !readOnly &&
    canTrain(training.scope ?? NO_SCOPE) &&
    // PyTorch for neural networks, scikit-learn for classical models.
    Boolean(status && canStartTraining(status, trainingFamily(analysis)));
  const source: EffectiveSource | null =
    editable && status
      ? effectiveSource(status, data.source ?? DEFAULT_SOURCE, data.datasets)
      : null;
  const ref = source?.ref ?? null;
  const name = source?.name ?? null;
  const realData = ref !== null && ref.kind !== 'random' && name !== null;

  // ---------------------------------------------------------------- selections
  // Every selection of a dataset arms the automatic adaptation once.
  const selection = data.selection;
  const seenSelection = useRef(selection);
  const [armed, setArmed] = useState(false);
  useEffect(() => {
    if (selection !== seenSelection.current) {
      seenSelection.current = selection;
      setArmed(true);
    }
  }, [selection]);

  // ---------------------------------------------------------------- check
  const networkValid = nodes.length > 0 && analysis?.valid === true;
  const setupVisible = training.opened && training.view === 'setup';
  const wanted = setupVisible || armed || (selectedInput !== null && realData);
  const compatibility = useCompatibility(doc, ref, editable && networkValid && wanted);
  const fresh = compatibility.fresh ? compatibility.result : null;
  const fixes = useMemo(() => fresh?.fixes ?? [], [fresh]);

  // ---------------------------------------------------------------- changes
  /** Set the params of the fixes, with a notification to undo it. */
  const apply = useCallback((list: CompatibilityFix[], datasetName: string, images: boolean) => {
    const { nodes: current, readOnly: viewOnly, updateNode } = latest.current;
    if (viewOnly) {
      return;
    }
    const changes: { id: string; before: Params; after: Params }[] = [];
    for (const fix of list) {
      const node = current.find((n) => n.id === fix.node);
      if (node) {
        const after = { ...node.data.params, ...fix.params };
        changes.push({ id: node.id, before: node.data.params, after });
        updateNode(node.id, { params: after });
      }
    }
    if (changes.length === 0) {
      return;
    }
    showUndo(`${describeFixes(list, images)} from “${datasetName}”.`, () => {
      // Only layers that were not changed since (e.g. by hand) go back.
      for (const change of changes) {
        const node = latest.current.nodes.find((n) => n.id === change.id);
        if (node && sameParams(node.data.params, change.after)) {
          latest.current.updateNode(change.id, { params: change.before });
        }
      }
    });
  }, []);

  /** A first network on the empty canvas. */
  const start = useCallback((network: GraphDocument, datasetName: string, images: boolean) => {
    latest.current.startNetwork(network);
    const input = network.nodes[0]?.params?.shape;
    const last = network.nodes[network.nodes.length - 1]?.params;
    // Segmentation networks end in a convolution with one channel per class.
    const channels = last?.out_channels;
    const outputs = channels ?? last?.out_features;
    const shape = Array.isArray(input) ? describeShape(input as number[], images) : 'the data';
    showUndo(
      `Started a network for “${datasetName}”: Input ${shape}, ${String(outputs)} ` +
        `output${channels !== undefined ? ' channel' : ''}${outputs === 1 ? '' : 's'}.`,
      () => {
        const ids = new Set(latest.current.nodes.map((n) => n.id));
        if (network.nodes.every((n) => ids.has(n.id))) {
          network.nodes.forEach((n) => latest.current.deleteNode(n.id));
        }
      }
    );
  }, []);

  // ---------------------------------------------------------------- automatic, once
  const { canvasReady, specs } = options;
  const datasetsLoading = data.datasets === null;
  const empty = nodes.length === 0;
  const analysisValid = analysis?.valid ?? null;
  useEffect(() => {
    if (!armed) {
      return;
    }
    const done = () => setArmed(false);
    if (!loggedIn || readOnly || (status && !editable)) {
      done(); // Viewers, visitors, or no training on this server.
      return;
    }
    if (!source) {
      return; // Waiting for the server's capabilities.
    }
    if (!ref || ref.kind === 'random' || !name) {
      if (!(source.choice.tab === 'upload' && datasetsLoading)) {
        done(); // Demo data fits every network.
      }
      return;
    }
    if (empty) {
      if (canvasReady) {
        const network = starterNetwork(source, specs);
        if (network) {
          start(network, name, isImageSource(source));
        }
        done();
      }
      return;
    }
    if (analyzing) {
      return;
    }
    if (analysisValid !== true) {
      done(); // The network has problems: nothing to adapt.
      return;
    }
    if (!compatibility.fresh) {
      return;
    }
    if (fresh && !fresh.ok && onlyInputAndOutput(fixes)) {
      apply(fixes, name, isImageSource(source));
    }
    done();
  }, [
    armed,
    loggedIn,
    readOnly,
    status,
    editable,
    source,
    ref,
    name,
    datasetsLoading,
    empty,
    canvasReady,
    specs,
    analyzing,
    analysisValid,
    compatibility.fresh,
    fresh,
    fixes,
    apply,
    start,
  ]);

  // ---------------------------------------------------------------- one click
  const images = source ? isImageSource(source) : false;
  const adapt = useMemo(
    () =>
      editable && name && fixes.length > 0 && onlyInputAndOutput(fixes)
        ? () => apply(fixes, name, images)
        : null,
    [editable, name, fixes, images, apply]
  );
  const controller = useMemo<DatasetFitController | null>(
    () => (editable ? { compatibility, adapt } : null),
    [editable, compatibility, adapt]
  );

  const inputFit = useMemo<InputFit | null>(() => {
    if (!editable || !realData || !name || selectedInput === null || !fresh) {
      return null;
    }
    const fix = fixes.find((f) => f.node === selectedInput && f.type === 'Input');
    if (fix) {
      return {
        nodeId: selectedInput,
        datasetName: name,
        matches: false,
        summary: fix.summary,
        apply: () => apply([fix], name, images),
      };
    }
    if (fresh.ok && analysis?.inputs.includes(selectedInput)) {
      return {
        nodeId: selectedInput,
        datasetName: name,
        matches: true,
        summary: null,
        apply: null,
      };
    }
    return null;
  }, [editable, realData, name, selectedInput, fresh, fixes, analysis, images, apply]);

  return { controller, inputFit };
}
