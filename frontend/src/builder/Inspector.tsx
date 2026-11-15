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
import type { Analysis, LayerSpec, ModelMember, NodeAnalysis } from '@/api/types';
import { LayerComments } from '@/comments/LayerComments';
import { LayerLearning } from '@/explain/LayerLearning';
import { currentLocale, i18n, Trans, useTranslation } from '@/i18n';
import { categoryColor } from '@/theme';
import { useBuilder, type InputFit } from './BuilderContext';
import { formatOutput, formatShape } from './format';
import { isParamVisible, type LayerFlowNode } from './graph';
import {
  inputSlots,
  isBoosting,
  isClassicalSpec,
  isEnsemble,
  isKernelNode,
  kernelTerms,
  networkFamily,
} from './kinds';
import { ParamField } from './ParamField';

const GAUSSIAN_PROCESSES = ['GaussianProcessRegressor', 'GaussianProcessClassifier'];

/**
 * The parameters of a layer: weights; hyperparameters of kernels; nothing yet for SVMs and
 * gradient boosting (their support vectors or trees come from the fit); the models of an
 * ensemble.
 */
function paramsStat(
  spec: LayerSpec,
  info: NodeAnalysis | undefined,
  analysis: Analysis | null
): { label: string; value: string } {
  const count = info?.num_params;
  if (!isClassicalSpec(spec)) {
    return {
      label: i18n.t('builder.inspector.parameters'),
      value: count?.toLocaleString(currentLocale()) ?? '–',
    };
  }
  if (isEnsemble(spec.type)) {
    return {
      label: i18n.t('builder.inspector.models'),
      value: info?.members ? String(info.members.length) : '–',
    };
  }
  if (GAUSSIAN_PROCESSES.includes(spec.type)) {
    // With kernel bricks, their hyperparameters are the ones the fit optimises.
    const total = count === 0 && analysis?.valid ? analysis.total_params : count;
    return {
      label: i18n.t('builder.inspector.fittedHyperparameters'),
      value: total?.toLocaleString(currentLocale()) ?? '–',
    };
  }
  if (count === null) {
    const label = isBoosting(spec.type)
      ? i18n.t('builder.inspector.decisionTrees')
      : i18n.t('builder.inspector.supportVectors');
    return { label, value: i18n.t('builder.inspector.afterFit') };
  }
  if (inputSlots(spec).length > 0) {
    // A kernel sum or product has none of its own; the kernels it combines count theirs.
    return {
      label: i18n.t('builder.inspector.fittedHyperparameters'),
      value: i18n.t('builder.inspector.inItsKernels'),
    };
  }
  return {
    label: i18n.t('builder.inspector.fittedHyperparameters'),
    value: count?.toLocaleString(currentLocale()) ?? '–',
  };
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

/**
 * The models of an ensemble in the order of their connections, which is also the order of
 * its weights. A click selects the model on the canvas.
 */
function EnsembleMembers({
  members,
  weights,
  onSelect,
}: {
  members: ModelMember[];
  weights: unknown;
  onSelect: (node: string) => void;
}) {
  const { t } = useTranslation();
  const weighted = Array.isArray(weights) && weights.length === members.length;
  return (
    <Stack gap={4}>
      <Text size="xs" c="dimmed">
        {weighted ? t('builder.inspector.membersWeighted') : t('builder.inspector.members')}
      </Text>
      <List size="sm" spacing={2} type="ordered" aria-label={t('builder.inspector.membersLabel')}>
        {members.map((member, index) => (
          <List.Item key={member.node}>
            <UnstyledButton onClick={() => onSelect(member.node)}>
              <Text span size="sm" ff="monospace" fw={600}>
                {member.name}
              </Text>{' '}
              <Text span size="sm" c="dimmed">
                {member.label}
                {weighted
                  ? ` · ${t('builder.inspector.weight', { weight: String((weights as unknown[])[index]) })}`
                  : ''}
              </Text>
            </UnstyledButton>
          </List.Item>
        ))}
      </List>
    </Stack>
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
  const { t } = useTranslation();
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
    links,
    focusNode,
  } = useBuilder();
  const spec = specs[node.data.type];
  const info = analysis?.nodes[node.id];
  const errors = errorsByNode[node.id] ?? [];
  const warnings = warningsByNode[node.id] ?? [];

  if (!spec) {
    return (
      <Stack p="md">
        <Alert color="red">{t('builder.inspector.unknownType', { type: node.data.type })}</Alert>
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
            <Tooltip label={t('builder.inspector.duplicate')}>
              <ActionIcon
                variant="subtle"
                color="gray"
                onClick={() => duplicateNode(node.id)}
                aria-label={t('builder.inspector.duplicateLayer')}
              >
                <IconCopy size={16} />
              </ActionIcon>
            </Tooltip>
            <Tooltip label={t('builder.inspector.deleteTooltip')}>
              <ActionIcon
                variant="subtle"
                color="red"
                onClick={() => deleteNode(node.id)}
                aria-label={t('builder.inspector.deleteLayer')}
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
          label={
            kernel || info?.task === 'classification'
              ? t('builder.inspector.output')
              : t('builder.inspector.outputShape')
          }
          value={formatOutput(info, kernel)}
        />
        <Stat label={params.label} value={params.value} />
      </SimpleGrid>
      {/* For learners: parameter formula and receptive field (src/explain). */}
      <LayerLearning nodeId={node.id} analysis={analysis} links={links} />
      {kernelCode && (kernel || gaussianProcess) && (
        <Stack gap={4}>
          <Text size="xs" c="dimmed">
            {kernel
              ? t('builder.inspector.kernel')
              : info?.num_params
                ? t('builder.inspector.defaultKernel')
                : t('builder.inspector.kernel')}
          </Text>
          <KernelCode code={kernelCode} label={t('builder.inspector.kernelCode')} />
        </Stack>
      )}
      {info?.members && info.members.length > 0 && (
        <EnsembleMembers
          members={info.members}
          weights={node.data.params.weights}
          onSelect={focusNode}
        />
      )}
      {inputFit && inputFit.nodeId === node.id && !readOnly && <DatasetFit fit={inputFit} />}

      {/* View only: the settings are shown, but every field is disabled. */}
      <Fieldset variant="unstyled" disabled={readOnly} p={0} m={0}>
        <Stack gap="sm">
          {spec.params.length > 0 && (
            <Divider label={t('builder.inspector.settings')} labelPosition="left" />
          )}
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
              {t('builder.inspector.noSettings')}
            </Text>
          )}

          <Divider label={t('builder.inspector.code')} labelPosition="left" />
          <TextInput
            label={t('builder.inspector.nameInCode')}
            description={t('builder.inspector.nameInCodeHint')}
            placeholder={t('builder.inspector.automatic')}
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
  const { t } = useTranslation();
  if (fit.matches) {
    return (
      <Badge
        color="green"
        variant="light"
        leftSection={<IconCircleCheck size={12} />}
        style={{ alignSelf: 'flex-start', textTransform: 'none' }}
      >
        {t('builder.inspector.matches', { name: fit.datasetName })}
      </Badge>
    );
  }
  return (
    <Paper withBorder p="xs" radius="sm">
      <Group justify="space-between" gap="xs">
        <Text size="xs" c="dimmed" style={{ flex: '1 1 140px' }}>
          {t('builder.inspector.trainingData', { name: fit.datasetName })}: {fit.summary}
        </Text>
        {fit.apply && (
          <Button
            size="compact-xs"
            variant="light"
            leftSection={<IconDatabase size={12} />}
            onClick={fit.apply}
          >
            {t('builder.inspector.setFromDataset')}
          </Button>
        )}
      </Group>
    </Paper>
  );
}

function ModelSummary({ nodes, tips }: { nodes: LayerFlowNode[]; tips?: React.ReactNode[] }) {
  const { t } = useTranslation();
  const { analysis, specs, readOnly, focusNode } = useBuilder();
  const family = networkFamily(analysis);
  const classical = family !== 'neural' ? (analysis?.classical ?? null) : null;
  const hybrid = family === 'hybrid' ? (analysis?.hybrid ?? null) : null;
  const labelOf = (id: string | null) => {
    const node = nodes.find((n) => n.id === id);
    if (!node) {
      return t('builder.inspector.graph');
    }
    return node.data.name || specs[node.data.type]?.label || node.data.type;
  };

  return (
    <Stack p="md" gap="sm">
      <Title order={4}>{t('builder.inspector.model')}</Title>
      {nodes.length === 0 && (
        <Text size="sm" c="dimmed">
          <Trans i18nKey="builder.inspector.emptyCanvas" components={{ b: <b /> }} />
        </Text>
      )}

      {analysis && nodes.length > 0 && (
        <>
          {analysis.valid ? (
            <Alert color="green" variant="light" icon={<IconCircleCheck size={18} />}>
              {t('builder.inspector.valid')}
            </Alert>
          ) : (
            <Alert color="red" variant="light" icon={<IconAlertTriangle size={18} />}>
              {t('builder.inspector.problems', { count: analysis.errors.length })}
            </Alert>
          )}

          {analysis.valid && (
            <SimpleGrid cols={1} spacing="xs">
              {hybrid && (
                <Stat
                  label={t('builder.inspector.neuralLayers')}
                  value={t('builder.inspector.hybridFeatures', {
                    features: hybrid.features ?? '–',
                    from: labelOf(hybrid.boundary),
                    params: hybrid.neural_params?.toLocaleString(currentLocale()) ?? '–',
                  })}
                />
              )}
              {classical ? (
                <>
                  <Stat
                    label={t('builder.inspector.classicalModel')}
                    value={specs[classical.type]?.label ?? classical.estimator}
                  />
                  {classical.members ? (
                    <Stat
                      label={t('builder.inspector.modelCount', { count: classical.members.length })}
                      value={classical.members.map((member) => member.name).join(', ')}
                    />
                  ) : (
                    <Stat
                      label={
                        analysis.total_params === null
                          ? t('builder.inspector.parameters')
                          : t('builder.inspector.optimisedHyperparameters')
                      }
                      value={
                        analysis.total_params === null
                          ? isBoosting(classical.type)
                            ? t('builder.inspector.treesAfterFit')
                            : t('builder.inspector.vectorsAfterFit')
                          : analysis.total_params.toLocaleString(currentLocale())
                      }
                    />
                  )}
                </>
              ) : (
                <Stat
                  label={t('builder.inspector.trainableParameters')}
                  value={analysis.total_params?.toLocaleString(currentLocale()) ?? '–'}
                />
              )}
              {analysis.inputs.map((id) => (
                <Stat
                  key={`in-${id}`}
                  label={`${t('builder.inspector.input')} · ${labelOf(id)}`}
                  value={`batch × ${formatShape(analysis.nodes[id]?.output_shape)}`}
                />
              ))}
              {analysis.outputs.map((id) => (
                <Stat
                  key={`out-${id}`}
                  label={`${t('builder.inspector.output')} · ${labelOf(id)}`}
                  value={
                    formatOutput(analysis.nodes[id]) === t('builder.format.classesFromData')
                      ? t('builder.inspector.oneClassPerSample')
                      : `batch × ${formatShape(analysis.nodes[id]?.output_shape)}`
                  }
                />
              ))}
              {classical?.kernel && (
                <Stack gap={4}>
                  <Text size="xs" c="dimmed">
                    {t('builder.inspector.kernelStart')}
                  </Text>
                  <KernelCode
                    code={classical.kernel}
                    label={t('builder.inspector.kernelOfModel')}
                  />
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

      <Divider label={t('builder.inspector.tips')} labelPosition="left" mt="sm" />
      {tips ? (
        <List size="sm" spacing={4} c="dimmed">
          {tips.map((tip, index) => (
            <List.Item key={index}>{tip}</List.Item>
          ))}
        </List>
      ) : readOnly ? (
        <List size="sm" spacing={4} c="dimmed">
          <List.Item>{t('builder.inspector.tipViewOnly')}</List.Item>
          <List.Item>{t('builder.inspector.tipClickView')}</List.Item>
          <List.Item>
            <Trans i18nKey="builder.inspector.tipDuplicate" components={{ b: <b /> }} />
          </List.Item>
        </List>
      ) : (
        <List size="sm" spacing={4} c="dimmed">
          <List.Item>{t('builder.inspector.tipClickEdit')}</List.Item>
          <List.Item>{t('builder.inspector.tipConnect')}</List.Item>
          <List.Item>
            <Trans i18nKey="builder.inspector.tipAuto" components={{ i: <i /> }} />
          </List.Item>
          {classical ? (
            <>
              <List.Item>
                <Trans i18nKey="builder.inspector.tipClassical" components={{ b: <b /> }} />
              </List.Item>
              <List.Item>{t('builder.inspector.tipKernelValues')}</List.Item>
              <List.Item>{t('builder.inspector.tipEnsemble')}</List.Item>
            </>
          ) : (
            <List.Item>
              <Trans i18nKey="builder.inspector.tipMerge" components={{ b: <b /> }} />
            </List.Item>
          )}
          <List.Item>
            <Trans i18nKey="builder.inspector.tipKeys" components={{ k: <Kbd /> }} />
          </List.Item>
        </List>
      )}
    </Stack>
  );
}
