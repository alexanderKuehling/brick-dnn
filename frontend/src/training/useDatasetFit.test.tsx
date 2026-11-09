/** The Input layer follows the dataset chosen for training (useDatasetFit). */
import { useCallback, useMemo, useState } from 'react';
import { act, render, screen, userEvent, waitFor } from '@test-utils';
import { beforeEach, vi } from 'vitest';
import { notifications } from '@mantine/notifications';
import {
  api,
  type Analysis,
  type Compatibility,
  type GraphDocument,
  type LayerSpec,
  type Params,
} from '@/api';
import { fromDocument, toDocument, type LayerFlowNode, type LayerNodeData } from '@/builder/graph';
import { NO_SCOPE } from './access';
import { dataset, STATUS } from './fixtures';
import type { SourceChoice } from './settings';
import { useDatasetFit, type DatasetFit } from './useDatasetFit';
import type { DataController } from './useDatasets';
import type { TrainingController } from './useTraining';

vi.mock('@/api', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/api')>();
  return { ...actual, api: { training: { check: vi.fn(), estimate: vi.fn() } } };
});

const MLP: GraphDocument = {
  version: 1,
  nodes: [
    { id: 'in', type: 'Input', params: { shape: [784] }, position: { x: 0, y: 0 } },
    { id: 'fc', type: 'Linear', params: { out_features: 10 }, position: { x: 300, y: 0 } },
  ],
  edges: [{ id: 'e', source: 'in', target: 'fc' }],
};

const SPECS = Object.fromEntries(
  (
    [
      ['Input', { shape: [784], dtype: 'float' }],
      ['Linear', { out_features: 64 }],
      ['Activation', { function: 'relu' }],
      ['Flatten', {}],
    ] as [string, Params][]
  ).map(([type, defaults]) => [
    type,
    {
      type,
      label: type,
      params: Object.entries(defaults).map(([name, value]) => ({ name, default: value })),
    } as unknown as LayerSpec,
  ])
);

const IRIS = dataset({ id: 7, name: 'iris' });
const UPLOAD: SourceChoice = { tab: 'upload', builtin: null, upload: 7 };

const INPUT_FIX = {
  node: 'in',
  type: 'Input',
  params: { shape: [4] },
  previous: { shape: [784] },
  summary: 'Input shape 784 → 4',
};
const OUTPUT_FIX = {
  node: 'fc',
  type: 'Linear',
  params: { out_features: 3 },
  previous: { out_features: 10 },
  summary: 'Outputs of the last layer 10 → 3',
};

/** The server: iris needs 4 features and 3 outputs. */
function serverCheck(graph: GraphDocument): Compatibility {
  const params = Object.fromEntries(graph.nodes.map((n) => [n.id, n.params ?? {}]));
  const fixes = [
    ...(JSON.stringify(params.in?.shape) === '[4]' ? [] : [INPUT_FIX]),
    ...(params.fc?.out_features === 3 ? [] : [OUTPUT_FIX]),
  ];
  const inputWrong = fixes.some((fix) => fix.node === 'in');
  return {
    ok: !inputWrong,
    errors: inputWrong ? ['The Input expects shape (784), but the CSV file has 4 columns.'] : [],
    warnings: [],
    plan: null,
    fixes,
  };
}

interface HarnessProps {
  selection?: number;
  source?: SourceChoice;
  readOnly?: boolean;
  opened?: boolean;
  selectedInput?: string | null;
  initial?: GraphDocument;
  onFit: (fit: DatasetFit) => void;
}

const startNetwork = vi.fn();
/** The builder's node update, for changes by hand. */
let editByHand: (id: string, patch: Partial<LayerNodeData>) => void = () => undefined;

function Harness({
  selection = 0,
  source = UPLOAD,
  readOnly = false,
  opened = false,
  selectedInput = null,
  initial = MLP,
  onFit,
}: HarnessProps) {
  const [nodes, setNodes] = useState<LayerFlowNode[]>(() => fromDocument(initial).nodes);
  const [edges, setEdges] = useState(() => fromDocument(initial).edges);
  const doc = useMemo(() => toDocument(nodes, edges), [nodes, edges]);
  const analysis: Analysis | null = nodes.length
    ? {
        valid: true,
        errors: [],
        warnings: [],
        nodes: {},
        inputs: [nodes[0].id],
        outputs: [nodes[nodes.length - 1].id],
        total_params: 0,
      }
    : null;
  const updateNode = useCallback(
    (id: string, patch: Partial<LayerNodeData>) =>
      setNodes((current) =>
        current.map((n) => (n.id === id ? { ...n, data: { ...n.data, ...patch } } : n))
      ),
    []
  );
  editByHand = updateNode;
  const deleteNode = useCallback((id: string) => {
    setNodes((current) => current.filter((n) => n.id !== id));
    setEdges((current) => current.filter((e) => e.source !== id && e.target !== id));
  }, []);
  const start = useCallback((graph: GraphDocument) => {
    startNetwork(graph);
    setNodes(fromDocument(graph).nodes);
    setEdges(fromDocument(graph).edges);
  }, []);
  const data = {
    source,
    setSource: vi.fn(),
    selection,
    datasets: [IRIS],
    datasetsError: null,
    reloadDatasets: vi.fn(),
    uploading: null,
    uploadProgress: null,
    uploadError: null,
    dismissUploadError: vi.fn(),
    upload: vi.fn(),
    cancelUpload: vi.fn(),
    remove: vi.fn(),
  } as DataController;
  const training = {
    status: STATUS,
    scope: NO_SCOPE,
    opened,
    view: 'setup',
    data,
  } as unknown as TrainingController;
  const fit = useDatasetFit({
    training,
    loggedIn: true,
    readOnly,
    doc,
    nodes,
    analysis,
    analyzing: false,
    specs: SPECS,
    canvasReady: true,
    selectedInput,
    updateNode,
    deleteNode,
    startNetwork: start,
  });
  onFit(fit);
  return (
    <output data-testid="params">
      {JSON.stringify(Object.fromEntries(nodes.map((n) => [n.data.type, n.data.params])))}
    </output>
  );
}

let fit: DatasetFit;
const capture = (value: DatasetFit) => {
  fit = value;
};
const params = () =>
  JSON.parse(screen.getByTestId('params').textContent!) as Record<string, Params>;
const pause = (ms = 600) => act(() => new Promise((resolve) => setTimeout(resolve, ms)));

/** Render the message of the last notification and click its Undo button. */
async function undoLastNotification() {
  const calls = vi.mocked(notifications.show).mock.calls;
  const { message } = calls[calls.length - 1][0];
  render(<div data-testid="notification">{message}</div>);
  await userEvent.click(screen.getByRole('button', { name: 'Undo' }));
}

beforeEach(() => {
  vi.clearAllMocks();
  vi.spyOn(notifications, 'show').mockImplementation(() => 'id');
  vi.spyOn(notifications, 'hide').mockImplementation(() => '');
  vi.mocked(api.training.check).mockImplementation((graph) => Promise.resolve(serverCheck(graph)));
});

describe('useDatasetFit', () => {
  it('adapts the Input once when a dataset is chosen, and can undo it', async () => {
    const { rerender } = render(<Harness onFit={capture} />);
    await pause();
    // Opening a network with a remembered dataset changes nothing.
    expect(api.training.check).not.toHaveBeenCalled();
    expect(params().Input).toEqual({ shape: [784] });

    rerender(<Harness selection={1} opened onFit={capture} />);
    await waitFor(() => expect(params().Input).toEqual({ shape: [4] }));
    expect(params().Linear).toEqual({ out_features: 3 });
    expect(notifications.show).toHaveBeenCalledTimes(1);
    const { message } = vi.mocked(notifications.show).mock.calls[0][0];
    render(<div>{message}</div>);
    expect(
      screen.getByText('Input set to 4 features, last layer to 3 outputs from “iris”.')
    ).toBeInTheDocument();

    await userEvent.click(screen.getByRole('button', { name: 'Undo' }));
    expect(params()).toEqual({ Input: { shape: [784] }, Linear: { out_features: 10 } });

    // Only once per selection: the network does not fit again, but stays as it is.
    await pause();
    await waitFor(() => expect(fit.controller?.adapt).not.toBeNull());
    expect(params().Input).toEqual({ shape: [784] });
    expect(notifications.show).toHaveBeenCalledTimes(1);

    // The one-click action applies the fixes again.
    act(() => fit.controller!.adapt!());
    expect(params()).toEqual({ Input: { shape: [4] }, Linear: { out_features: 3 } });
    expect(notifications.show).toHaveBeenCalledTimes(2);
  });

  it('keeps changes made by hand when undoing', async () => {
    const { rerender } = render(<Harness onFit={capture} />);
    rerender(<Harness selection={1} onFit={capture} />);
    await waitFor(() => expect(params().Input).toEqual({ shape: [4] }));
    act(() => editByHand('fc', { params: { out_features: 5 } }));
    await undoLastNotification();
    expect(params().Input).toEqual({ shape: [784] });
    expect(params().Linear).toEqual({ out_features: 5 });
  });

  it('shows in the Inspector that the Input matches the data', async () => {
    const { rerender } = render(<Harness onFit={capture} />);
    rerender(<Harness selection={1} onFit={capture} />);
    await waitFor(() => expect(params().Input).toEqual({ shape: [4] }));
    rerender(<Harness selection={1} selectedInput="in" onFit={capture} />);
    await waitFor(() => expect(fit.inputFit?.matches).toBe(true));
    expect(fit.inputFit).toMatchObject({ nodeId: 'in', datasetName: 'iris', apply: null });
  });

  it('only offers changes when the network fits already', async () => {
    vi.mocked(api.training.check).mockResolvedValue({
      ok: true,
      errors: [],
      warnings: ['The network outputs 10 values, but iris has only 3 classes.'],
      plan: null,
      fixes: [OUTPUT_FIX],
    });
    const { rerender } = render(<Harness onFit={capture} />);
    rerender(<Harness selection={1} opened onFit={capture} />);
    await waitFor(() => expect(fit.controller?.adapt).toBeTruthy());
    await pause();
    expect(params().Linear).toEqual({ out_features: 10 });
    expect(notifications.show).not.toHaveBeenCalled();
    act(() => fit.controller!.adapt!());
    expect(params().Linear).toEqual({ out_features: 3 });
  });

  it('does not change other layers on its own', async () => {
    vi.mocked(api.training.check).mockResolvedValue({
      ...serverCheck(MLP),
      fixes: [INPUT_FIX, { ...OUTPUT_FIX, type: 'Conv2d', params: { out_channels: 3 } }],
    });
    const { rerender } = render(<Harness onFit={capture} />);
    rerender(<Harness selection={1} opened onFit={capture} />);
    await waitFor(() => expect(fit.controller?.compatibility.fresh).toBe(true));
    await pause();
    expect(params().Input).toEqual({ shape: [784] });
    expect(fit.controller?.adapt).toBeNull();
  });

  it('never touches the network of a viewer', async () => {
    const { rerender } = render(<Harness readOnly onFit={capture} />);
    rerender(<Harness readOnly selection={1} opened selectedInput="in" onFit={capture} />);
    await pause();
    expect(api.training.check).not.toHaveBeenCalled();
    expect(params().Input).toEqual({ shape: [784] });
    expect(fit.controller).toBeNull();
    expect(fit.inputFit).toBeNull();
    expect(notifications.show).not.toHaveBeenCalled();
  });

  it('ignores demo data', async () => {
    const demo: SourceChoice = { tab: 'random', builtin: null, upload: null };
    const { rerender } = render(<Harness source={demo} selectedInput="in" onFit={capture} />);
    rerender(<Harness source={demo} selection={1} selectedInput="in" onFit={capture} />);
    await pause();
    expect(api.training.check).not.toHaveBeenCalled();
    expect(fit.inputFit).toBeNull();
  });

  it('offers to set the Input from the dataset in the Inspector', async () => {
    render(<Harness selectedInput="in" onFit={capture} />);
    await waitFor(() => expect(fit.inputFit).not.toBeNull());
    expect(fit.inputFit).toMatchObject({
      nodeId: 'in',
      datasetName: 'iris',
      matches: false,
      summary: 'Input shape 784 → 4',
    });
    act(() => fit.inputFit!.apply!());
    // Only the Input: the Inspector shows the Input layer.
    expect(params()).toEqual({ Input: { shape: [4] }, Linear: { out_features: 10 } });
    expect(vi.mocked(notifications.show).mock.calls[0][0].message).toBeTruthy();
    await waitFor(() => expect(fit.inputFit?.matches).toBe(true));
  });

  it('starts a network for the data on an empty canvas', async () => {
    const empty: GraphDocument = { version: 1, nodes: [], edges: [] };
    const { rerender } = render(<Harness initial={empty} onFit={capture} />);
    rerender(<Harness initial={empty} selection={1} onFit={capture} />);
    await waitFor(() => expect(startNetwork).toHaveBeenCalledTimes(1));
    expect(params()).toEqual({
      Input: { shape: [4], dtype: 'float' },
      Linear: { out_features: 3 },
      Activation: { function: 'relu' },
    });
    const graph = startNetwork.mock.calls[0][0] as GraphDocument;
    expect(graph.nodes.map((n) => n.type)).toEqual(['Input', 'Linear', 'Activation', 'Linear']);
    expect(graph.edges).toHaveLength(3);
    const { message } = vi.mocked(notifications.show).mock.calls[0][0];
    render(<div>{message}</div>);
    expect(
      screen.getByText('Started a network for “iris”: Input 4 features, 3 outputs.')
    ).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Undo' }));
    expect(params()).toEqual({});
  });
});
