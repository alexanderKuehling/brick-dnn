import { IconUnlink } from '@tabler/icons-react';
import {
  ActionIcon,
  Anchor,
  Divider,
  Group,
  List,
  Select,
  Stack,
  Text,
  Tooltip,
} from '@mantine/core';
import { useTranslation } from '@/i18n';
import { layerName } from './a11y';
import { useBuilder } from './BuilderContext';
import type { LayerFlowNode } from './graph';

/**
 * The connections of the selected layer, in words: where its inputs come from and where its
 * output goes, each with a button that removes it, and two lists to add one. The way to
 * connect layers without dragging on the canvas (keyboard, screen readers, WCAG 2.1.1 and
 * 2.5.7). View only: the lists alone.
 */
export function Connections({ node, nodes }: { node: LayerFlowNode; nodes: LayerFlowNode[] }) {
  const { t } = useTranslation();
  const { specs, links = [], readOnly, canConnect, connect, disconnect, focusNode } = useBuilder();
  const byId = new Map(nodes.map((n) => [n.id, n]));
  const name = (id: string) => layerName(byId.get(id), specs) || id;
  const editable = !readOnly && Boolean(connect && disconnect && canConnect);
  const inputs = links.filter((link) => link.target === node.id).map((link) => link.source);
  const outputs = links.filter((link) => link.source === node.id).map((link) => link.target);
  // Two layers of the same type and without a name in code look alike: add their type.
  const option = (other: LayerFlowNode) => ({
    value: other.id,
    label: other.data.name
      ? `${other.data.name} (${specs[other.data.type]?.label})`
      : name(other.id),
  });
  const others = nodes.filter((other) => other.id !== node.id);
  const inputCandidates = editable
    ? others.filter((other) => canConnect!(other.id, node.id)).map(option)
    : [];
  const outputCandidates = editable
    ? others.filter((other) => canConnect!(node.id, other.id)).map(option)
    : [];

  const list = (ids: string[], direction: 'in' | 'out', label: string) => (
    <Stack gap={2}>
      <Text size="xs" c="dimmed" id={`${node.id}-${direction}-label`}>
        {label}
      </Text>
      {ids.length === 0 ? (
        <Text size="sm">{t('a11y.connections.none')}</Text>
      ) : (
        <List
          size="sm"
          spacing={2}
          listStyleType="none"
          aria-labelledby={`${node.id}-${direction}-label`}
        >
          {ids.map((id) => {
            const [source, target] = direction === 'in' ? [id, node.id] : [node.id, id];
            return (
              <List.Item key={id}>
                <Group gap={4} wrap="nowrap">
                  <Anchor
                    component="button"
                    type="button"
                    size="sm"
                    underline="hover"
                    onClick={() => focusNode(id)}
                    title={t('a11y.connections.select', { name: name(id) })}
                  >
                    {name(id)}
                  </Anchor>
                  {editable && (
                    <Tooltip
                      label={t('a11y.connections.remove', {
                        source: name(source),
                        target: name(target),
                      })}
                    >
                      <ActionIcon
                        variant="subtle"
                        color="gray"
                        size="sm"
                        aria-label={t('a11y.connections.remove', {
                          source: name(source),
                          target: name(target),
                        })}
                        onClick={() => disconnect!(source, target)}
                      >
                        <IconUnlink size={14} />
                      </ActionIcon>
                    </Tooltip>
                  )}
                </Group>
              </List.Item>
            );
          })}
        </List>
      )}
    </Stack>
  );

  const add = (direction: 'in' | 'out') => {
    const candidates = direction === 'in' ? inputCandidates : outputCandidates;
    return (
      <Select
        size="xs"
        label={
          direction === 'in' ? t('a11y.connections.addInput') : t('a11y.connections.addOutput')
        }
        description={
          candidates.length === 0
            ? t('a11y.connections.noCandidates')
            : direction === 'in'
              ? t('a11y.connections.addInputHint')
              : t('a11y.connections.addOutputHint')
        }
        data={candidates}
        value={null}
        disabled={candidates.length === 0}
        searchable
        allowDeselect={false}
        onChange={(id) => {
          if (id) {
            if (direction === 'in') {
              connect!(id, node.id);
            } else {
              connect!(node.id, id);
            }
          }
        }}
        comboboxProps={{ withinPortal: true }}
      />
    );
  };

  return (
    <Stack gap="xs" data-testid="connections">
      <Divider label={t('a11y.connections.title')} labelPosition="left" />
      {list(inputs, 'in', t('a11y.connections.inputs'))}
      {editable && add('in')}
      {list(outputs, 'out', t('a11y.connections.outputs'))}
      {editable && add('out')}
    </Stack>
  );
}
