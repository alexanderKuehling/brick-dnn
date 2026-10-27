import {
  IconAlertTriangle,
  IconCircleCheck,
  IconCopy,
  IconInfoCircle,
  IconTrash,
} from '@tabler/icons-react';
import {
  ActionIcon,
  Alert,
  Divider,
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
import { categoryColor } from '@/theme';
import { useBuilder } from './BuilderContext';
import { formatShape } from './format';
import { isParamVisible, type LayerFlowNode } from './graph';
import { ParamField } from './ParamField';

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

export function Inspector({ node, nodes }: { node: LayerFlowNode | null; nodes: LayerFlowNode[] }) {
  return node ? <LayerInspector node={node} /> : <ModelSummary nodes={nodes} />;
}

function LayerInspector({ node }: { node: LayerFlowNode }) {
  const { specs, analysis, errorsByNode, warningsByNode, updateNode, deleteNode, duplicateNode } =
    useBuilder();
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

  return (
    <Stack p="md" gap="sm">
      <Group justify="space-between" wrap="nowrap">
        <Group gap="xs" wrap="nowrap">
          <span
            style={{
              width: 12,
              height: 12,
              borderRadius: 3,
              background: `var(--mantine-color-${color}-6)`,
            }}
          />
          <Title order={4}>{spec.label}</Title>
        </Group>
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
        <Stat label="Output shape" value={formatShape(info?.output_shape)} />
        <Stat label="Parameters" value={info?.num_params?.toLocaleString('en-US') ?? '–'} />
      </SimpleGrid>

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
        onChange={(event) => updateNode(node.id, { name: event.currentTarget.value || undefined })}
      />
    </Stack>
  );
}

function ModelSummary({ nodes }: { nodes: LayerFlowNode[] }) {
  const { analysis, specs, focusNode } = useBuilder();
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
              <Stat
                label="Trainable parameters"
                value={analysis.total_params?.toLocaleString('en-US') ?? '–'}
              />
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
                  value={`batch × ${formatShape(analysis.nodes[id]?.output_shape)}`}
                />
              ))}
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
      <List size="sm" spacing={4} c="dimmed">
        <List.Item>Click a layer to edit its settings here.</List.Item>
        <List.Item>
          Drag from the right dot of a layer to the left dot of another to connect them.
        </List.Item>
        <List.Item>
          Fields marked <i>auto</i> are inferred from the incoming tensor.
        </List.Item>
        <List.Item>
          Use <b>Add</b> or <b>Concat</b> to merge branches (skip connections, multiple inputs).
        </List.Item>
        <List.Item>
          <Kbd>Del</Kbd> deletes the selection, <Kbd>Ctrl</Kbd> + <Kbd>S</Kbd> saves,{' '}
          <Kbd>Ctrl</Kbd> + <Kbd>E</Kbd> exports.
        </List.Item>
      </List>
    </Stack>
  );
}
