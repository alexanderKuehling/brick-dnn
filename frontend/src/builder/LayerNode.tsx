import { memo } from 'react';
import { IconAlertTriangle, IconInfoCircle } from '@tabler/icons-react';
import { Handle, Position, type NodeProps } from '@xyflow/react';
import { Text, Tooltip } from '@mantine/core';
import { categoryColor } from '@/theme';
import { useBuilder } from './BuilderContext';
import { formatCount, formatShape, summarize } from './format';
import type { LayerFlowNode } from './graph';
import classes from './LayerNode.module.css';

function LayerNodeView({ id, data, selected }: NodeProps<LayerFlowNode>) {
  const { specs, analysis, errorsByNode, warningsByNode } = useBuilder();
  const spec = specs[data.type];
  const info = analysis?.nodes[id];
  const errors = errorsByNode[id] ?? [];
  const warnings = warningsByNode[id] ?? [];
  const color = categoryColor(spec?.category);
  const params = info?.params ?? data.params;

  const style = {
    '--node-color': `var(--mantine-color-${color}-6)`,
  } as React.CSSProperties;

  return (
    <div
      className={classes.node}
      style={style}
      data-selected={selected || undefined}
      data-error={errors.length > 0 || undefined}
      data-testid={`layer-${data.type}`}
    >
      {spec?.max_inputs !== 0 && (
        <Handle type="target" position={Position.Left} className={classes.handle} />
      )}
      <div className={classes.header}>
        <span className={classes.dot} />
        <span className={classes.title}>{data.name || spec?.label || data.type}</span>
        {data.name && (
          <Text size="xs" c="dimmed">
            {spec?.label ?? data.type}
          </Text>
        )}
        {errors.length > 0 && (
          <Tooltip label={errors.join(' ')} multiline w={280} withArrow>
            <IconAlertTriangle size={16} color="var(--mantine-color-red-6)" aria-label="Error" />
          </Tooltip>
        )}
        {errors.length === 0 && warnings.length > 0 && (
          <Tooltip label={warnings.join(' ')} multiline w={280} withArrow>
            <IconInfoCircle size={16} color="var(--mantine-color-yellow-7)" aria-label="Warning" />
          </Tooltip>
        )}
      </div>
      <div className={classes.body}>
        <Text size="sm" c="dimmed" truncate="end">
          {spec ? summarize(data.type, params) || ' ' : `Unknown layer "${data.type}"`}
        </Text>
        {errors[0] && (
          <Text size="xs" c="red" mt={4} lineClamp={3}>
            {errors[0]}
          </Text>
        )}
        <div className={classes.footer}>
          <span className={classes.shape} title="Output shape (batch dimension omitted)">
            {info?.output_shape ? formatShape(info.output_shape) : '?'}
          </span>
          {Boolean(info?.num_params) && <span>{formatCount(info?.num_params)} params</span>}
        </div>
      </div>
      <Handle type="source" position={Position.Right} className={classes.handle} />
    </div>
  );
}

export const LayerNode = memo(LayerNodeView);
