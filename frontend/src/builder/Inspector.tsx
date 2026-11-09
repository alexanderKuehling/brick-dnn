import {
  IconAlertTriangle,
  IconCircleCheck,
  IconCopy,
  IconDatabase,
  IconInfoCircle,
  IconTrash,
} from '@tabler/icons-react';
import {
  ActionIcon,
  Alert,
  Badge,
  Button,
  Code,
  Divider,
  Fieldset,
  Group,
  Kbd,
  List,
  Paper,
  SimpleGrid,
  Stack,
  Text,
  TextInput,
  Title,
  Tooltip,
  UnstyledButton,
} from '@mantine/core';
import type { Analysis, LayerSpec, NodeAnalysis } from '@/api/types';
import { LayerComments } from '@/comments/LayerComments';
import { categoryColor } from '@/theme';
import { useBuilder, type InputFit } from './BuilderContext';
import { formatOutput, formatShape } from './format';
import { isParamVisible, type LayerFlowNode } from './graph';
import { inputSlots, isClassicalSpec, isKernelNode, kernelTerms, networkFamily } from './kinds';
import { ParamField } from './ParamField';

const GAUSSIAN_PROCESSES = ['GaussianProcessRegressor', 'GaussianProcessClassifier'];

/** The parameters of a layer: weights; hyperparameters of kernels; nothing yet for SVMs. */
function paramsStat(
  spec: LayerSpec,
  info: NodeAnalysis | undefined,
  analysis: Analysis | null
): { label: string; value: string } {
  const count = info?.num_params;
  if (!isClassicalSpec(spec)) {
    return { label: 'Parameters', value: count?.toLocaleString('en-US') ?? '–' };
  }
  if (GAUSSIAN_PROCESSES.includes(spec.type)) {
    // With kernel bricks, their hyperparameters are the ones the fit optimises.
    const total = count === 0 && analysis?.valid ? analysis.total_params : count;
    return { label: 'Fitted hyperparameters', value: total?.toLocaleString('en-US') ?? '–' };
  }
  if (count === null) {
    return { label: 'Support vectors', value: 'after the fit' };
  }
  if (inputSlots(spec).length > 0) {
    // A kernel sum or product has none of its own; the kernels it combines count theirs.
    return { label: 'Fitted hyperparameters', value: 'in its kernels' };
  }
  return { label: 'Fitted hyperparameters', value: count?.toLocaleString('en-US') ?? '–' };
}

/** A kernel as scikit-learn code, one top-level term per line. */
function KernelCode({ code, label }: { code: string; label: string }) {
  return (
    <Code
      block
      aria-label={label}
      style={{
        whiteSpace: 'pre-wrap',
        overflowWrap: 'anywhere',
        fontSize: 'var(--mantine-font-size-xs)',
      }}
    >
      {kernelTerms(code).join('\n')}
    </Code>
  );
}

function Stat({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <Paper withBorder p="xs" radius="sm">
      <Text size="xs" c="dimmed">
        {label}
      </Text>
      <Text fw={600} ff="monospace" size="sm">
        {value}
      </Text>
    </Paper>
  );
}

interface InspectorProps {
  node: LayerFlowNode | null;
  nodes: LayerFlowNode[];
  /** Tips below the model summary instead of the ones for the builder or a shared network. */
  tips?: React.ReactNode[];
}

export function Inspector({ node, nodes, tips }: InspectorProps) {
  return node ? <LayerInspector node={node} /> : <ModelSummary nodes={nodes} tips={tips} />;
}

function LayerInspector({ node }: { node: LayerFlowNode }) {
  const {
    specs,
    analysis,
    errorsByNode,
    warningsByNode,
    readOnly,
    updateNode,
    deleteNode,
    duplicateNode,
    inputFit,
  } = useBuilder();
  const spec = specs[node.data.type];
  const info = analysis?.nodes[node.id];
  const errors = errorsByNode[node.id] ?? [];
  const warnings = warningsByNode[node.id] ?? [];

  if (!spec) {
    return (
      <Stack p="md">
        <Alert color="red">Unknown layer type “{node.data.type}”.</Alert>
      </Stack>
    );
  }

  const color = categoryColor(spec.category);
  const kernel = isKernelNode(spec, info);
  const params = paramsStat(spec, info, analysis);
  const gaussianProcess = GAUSSIAN_PROCESSES.includes(spec.type);
  const kernelCode = info?.kernel ?? null;

  return (
    <Stack p="md" gap="sm">
      <Group justify="space-between" wrap="nowrap">
        <Group gap="xs" wrap="nowrap">
          <span
            style={{
              width: 12,
              height: 12,
              borderRadius: kernel ? 2 : 3,
              background: `var(--mantine-color-${color}-6)`,
              transform: kernel ? 'rotate(45deg) scale(0.85)' : undefined,
            }}
          />
          <Title order={4}>{spec.label}</Title>
        </Group>
        {!readOnly && (
          <Group gap={4}>
            <Tooltip label="Duplicate">
              <ActionIcon
                variant="subtle"
                color="gray"
                onClick={() => duplicateNode(node.id)}
                aria-label="Duplicate layer"
              >
                <IconCopy size={16} />
              </ActionIcon>
            </Tooltip>
            <Tooltip label="Delete (Del)">
              <ActionIcon
                variant="subtle"
                color="red"
                onClick={() => deleteNode(node.id)}
                aria-label="Delete layer"
              >
                <IconTrash size={16} />
              </ActionIcon>
            </Tooltip>
          </Group>
        )}
      </Group>
      <Text size="sm" c="dimmed">
        {spec.description}
      </Text>

      {errors.map((message) => (
        <Alert key={message} color="red" variant="light" icon={<IconAlertTriangle size={18} />}>
          {message}
        </Alert>
      ))}
      {warnings.map((message) => (
        <Alert key={message} color="yellow" variant="light" icon={<IconInfoCircle size={18} />}>
          {message}
        </Alert>
      ))}

      <SimpleGrid cols={2} spacing="xs">
        <Stat
          label={kernel || info?.task === 'classification' ? 'Output' : 'Output shape'}
          value={formatOutput(info, kernel)}
        />
        <Stat label={params.label} value={params.value} />
      </SimpleGrid>
      {kernelCode && (kernel || gaussianProcess) && (
        <Stack gap={4}>
          <Text size="xs" c="dimmed">
            {kernel
              ? 'Kernel (scikit-learn)'
              : info?.num_params
                ? 'Default kernel (scikit-learn). Connect kernel bricks to the kernel input to choose your own.'
                : 'Kernel (scikit-learn)'}
          </Text>
          <KernelCode code={kernelCode} label="Kernel as scikit-learn code" />
        </Stack>
      )}
      {inputFit && inputFit.nodeId === node.id && !readOnly && <DatasetFit fit={inputFit} />}

      {/* View only: the settings are shown, but every field is disabled. */}
      <Fieldset variant="unstyled" disabled={readOnly} p={0} m={0}>
        <Stack gap="sm">
          {spec.params.length > 0 && <Divider label="Settings" labelPosition="left" />}
          {spec.params
            .filter((param) => isParamVisible(param, node.data.params))
            .map((param) => (
              <ParamField
                key={`${node.id}:${param.name}`}
                param={param}
                value={
                  // Missing (not cleared) values show the default the backend will use.
                  node.data.params[param.name] === undefined
                    ? param.auto
                      ? null
                      : param.default
                    : node.data.params[param.name]
                }
                inferred={info?.params?.[param.name]}
                onChange={(value) =>
                  updateNode(node.id, { params: { ...node.data.params, [param.name]: value } })
                }
              />
            ))}
          {spec.params.length === 0 && (
            <Text size="sm" c="dimmed">
              This layer has no settings.
            </Text>
          )}

          <Divider label="Code" labelPosition="left" />
          <TextInput
            label="Name in code"
            description="Optional variable name, e.g. classifier"
            placeholder="automatic"
            value={node.data.name ?? ''}
            onChange={(event) =>
              updateNode(node.id, { name: event.currentTarget.value || undefined })
            }
          />
        </Stack>
      </Fieldset>
      {/* Saved networks: comments on this layer (viewers may comment too). */}
      <LayerComments key={node.id} nodeId={node.id} />
    </Stack>
  );
}

/** The Input layer and the dataset chosen in the training panel. */
function DatasetFit({ fit }: { fit: InputFit }) {
  if (fit.matches) {
    return (
      <Badge
        color="green"
        variant="light"
        leftSection={<IconCircleCheck size={12} />}
        style={{ alignSelf: 'flex-start', textTransform: 'none' }}
      >
        Matches “{fit.datasetName}”
      </Badge>
    );
  }
  return (
    <Paper withBorder p="xs" radius="sm">
      <Group justify="space-between" gap="xs">
        <Text size="xs" c="dimmed" style={{ flex: '1 1 140px' }}>
          Training data “{fit.datasetName}”: {fit.summary}
        </Text>
        {fit.apply && (
          <Button
            size="compact-xs"
            variant="light"
            leftSection={<IconDatabase size={12} />}
            onClick={fit.apply}
          >
            Set from dataset
          </Button>
        )}
      </Group>
    </Paper>
  );
}

function ModelSummary({ nodes, tips }: { nodes: LayerFlowNode[]; tips?: React.ReactNode[] }) {
  const { analysis, specs, readOnly, focusNode } = useBuilder();
  const classical = networkFamily(analysis) === 'classical' ? (analysis?.classical ?? null) : null;
  const labelOf = (id: string | null) => {
    const node = nodes.find((n) => n.id === id);
    if (!node) {
      return 'Graph';
    }
    return node.data.name || specs[node.data.type]?.label || node.data.type;
  };

  return (
    <Stack p="md" gap="sm">
      <Title order={4}>Model</Title>
      {nodes.length === 0 && (
        <Text size="sm" c="dimmed">
          The canvas is empty. Start with an <b>Input</b> layer from the palette or load an example.
        </Text>
      )}

      {analysis && nodes.length > 0 && (
        <>
          {analysis.valid ? (
            <Alert color="green" variant="light" icon={<IconCircleCheck size={18} />}>
              The network is valid and ready to export.
            </Alert>
          ) : (
            <Alert color="red" variant="light" icon={<IconAlertTriangle size={18} />}>
              {analysis.errors.length === 1
                ? '1 problem needs fixing before export.'
                : `${analysis.errors.length} problems need fixing before export.`}
            </Alert>
          )}

          {analysis.valid && (
            <SimpleGrid cols={1} spacing="xs">
              {classical ? (
                <>
                  <Stat
                    label="Classical model (scikit-learn)"
                    value={specs[classical.type]?.label ?? classical.estimator}
                  />
                  <Stat
                    label={
                      analysis.total_params === null
                        ? 'Parameters'
                        : 'Hyperparameters the fit optimises'
                    }
                    value={
                      analysis.total_params === null
                        ? 'support vectors, after the fit'
                        : analysis.total_params.toLocaleString('en-US')
                    }
                  />
                </>
              ) : (
                <Stat
                  label="Trainable parameters"
                  value={analysis.total_params?.toLocaleString('en-US') ?? '–'}
                />
              )}
              {analysis.inputs.map((id) => (
                <Stat
                  key={`in-${id}`}
                  label={`Input · ${labelOf(id)}`}
                  value={`batch × ${formatShape(analysis.nodes[id]?.output_shape)}`}
                />
              ))}
              {analysis.outputs.map((id) => (
                <Stat
                  key={`out-${id}`}
                  label={`Output · ${labelOf(id)}`}
                  value={
                    formatOutput(analysis.nodes[id]) === 'classes from data'
                      ? 'one class per sample'
                      : `batch × ${formatShape(analysis.nodes[id]?.output_shape)}`
                  }
                />
              ))}
              {classical?.kernel && (
                <Stack gap={4}>
                  <Text size="xs" c="dimmed">
                    Kernel (starting values, optimised by the fit)
                  </Text>
                  <KernelCode code={classical.kernel} label="Kernel of the model" />
                </Stack>
              )}
            </SimpleGrid>
          )}

          {analysis.errors.length > 0 && (
            <Stack gap={6}>
              {analysis.errors.map((issue, index) => (
                <UnstyledButton
                  key={index}
                  onClick={() => issue.node && focusNode(issue.node)}
                  disabled={!issue.node}
                >
                  <Text size="sm">
                    <Text span fw={600} c="red">
                      {labelOf(issue.node)}:
                    </Text>{' '}
                    {issue.message}
                  </Text>
                </UnstyledButton>
              ))}
            </Stack>
          )}

          {analysis.warnings.map((issue, index) => (
            <Alert key={index} color="yellow" variant="light" icon={<IconInfoCircle size={18} />}>
              {issue.message}
            </Alert>
          ))}
        </>
      )}

      <Divider label="Tips" labelPosition="left" mt="sm" />
      {tips ? (
        <List size="sm" spacing={4} c="dimmed">
          {tips.map((tip, index) => (
            <List.Item key={index}>{tip}</List.Item>
          ))}
        </List>
      ) : readOnly ? (
        <List size="sm" spacing={4} c="dimmed">
          <List.Item>You can view this network, but not change it.</List.Item>
          <List.Item>Click a layer to see its settings here.</List.Item>
          <List.Item>
            <b>Duplicate to my networks</b> makes a copy that you can change.
          </List.Item>
        </List>
      ) : (
        <List size="sm" spacing={4} c="dimmed">
          <List.Item>Click a layer to edit its settings here.</List.Item>
          <List.Item>
            Drag from the right dot of a layer to the left dot of another to connect them.
          </List.Item>
          <List.Item>
            Fields marked <i>auto</i> are inferred from the incoming tensor.
          </List.Item>
          {classical ? (
            <>
              <List.Item>
                Connect the <b>Input</b> straight to the model. Kernels flow along the dashed
                connections into a <b>Kernel sum</b>, a <b>Kernel product</b> or the Gaussian
                process.
              </List.Item>
              <List.Item>
                Kernel values are starting values: the Gaussian process optimises them when it is
                trained, unless they are fixed.
              </List.Item>
            </>
          ) : (
            <List.Item>
              Use <b>Add</b> or <b>Concat</b> to merge branches (skip connections, multiple inputs).
            </List.Item>
          )}
          <List.Item>
            <Kbd>Del</Kbd> deletes the selection, <Kbd>Ctrl</Kbd> + <Kbd>S</Kbd> saves,{' '}
            <Kbd>Ctrl</Kbd> + <Kbd>E</Kbd> exports.
          </List.Item>
        </List>
      )}
    </Stack>
  );
}
