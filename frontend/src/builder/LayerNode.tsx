import { memo } from 'react';
import { IconAlertTriangle, IconInfoCircle } from '@tabler/icons-react';
import { Handle, Position, useNodeConnections, type NodeProps } from '@xyflow/react';
import { Text, Tooltip } from '@mantine/core';
import { CommentMarker } from '@/comments/CommentMarker';
import { useRemoteSelection } from '@/live/presence';
import { categoryColor } from '@/theme';
import { useBuilder } from './BuilderContext';
import { formatCount, formatOutput, hyperparameters, summarize } from './format';
import type { LayerFlowNode } from './graph';
import { inputSlots, isClassicalSpec, isKernelNode, outputKind, slotHandle } from './kinds';
import classes from './LayerNode.module.css';

/** Vertical position (percent) of the input handle `index` of `count`. */
function handleTop(index: number, count: number): string {
  return `${Math.round(36 + (index * 40) / Math.max(1, count - 1))}%`;
}

/** What the footer says about the parameters: a count, hyperparameters, or nothing. */
function paramsText(count: number | null | undefined, classical: boolean): string | null {
  if (!count) {
    return null;
  }
  return classical ? hyperparameters(count) : `${formatCount(count)} params`;
}

function LayerNodeView({ id, data, selected }: NodeProps<LayerFlowNode>) {
  const { specs, analysis, errorsByNode, warningsByNode, presence } = useBuilder();
  const spec = specs[data.type];
  const info = analysis?.nodes[id];
  const errors = errorsByNode[id] ?? [];
  const warnings = warningsByNode[id] ?? [];
  const color = categoryColor(spec?.category);
  const params = info?.params ?? data.params;
  const slots = inputSlots(spec);
  const kernel = isKernelNode(spec, info);
  const classical = isClassicalSpec(spec);
  // Gaussian processes: is a kernel connected, or do they use their default kernel?
  const kernelConnections = useNodeConnections({ handleType: 'target', handleId: 'kernel' });
  const output = formatOutput(info, kernel);
  const outputTitle = kernel
    ? 'Output: a kernel for a Gaussian process'
    : output === 'classes from data'
      ? 'One class per sample; the classes come from the training data'
      : 'Output shape (batch dimension omitted)';
  // Live editing: others who have this layer selected (outline and name tag in their colour).
  const remote = useRemoteSelection(presence ?? null, id);

  const style = {
    '--node-color': `var(--mantine-color-${color}-6)`,
    ...(remote.length > 0 ? { '--remote-color': `var(--mantine-color-${remote[0].color}-6)` } : {}),
  } as React.CSSProperties;

  return (
    <div
      className={classes.node}
      style={style}
      data-selected={selected || undefined}
      data-error={errors.length > 0 || undefined}
      data-remote={remote.length > 0 || undefined}
      data-testid={`layer-${data.type}`}
    >
      {remote.length > 0 && (
        <span className={classes.remoteTag} title={remote.map((p) => p.username).join(', ')}>
          {remote[0].username}
          {remote.length > 1 && ` +${remote.length - 1}`}
        </span>
      )}
      {slots.map((slot, index) => (
        <Handle
          key={slot.name}
          type="target"
          position={Position.Left}
          id={slotHandle(slots, slot)}
          className={classes.handle}
          data-kind={slot.kind}
          style={slots.length > 1 ? { top: handleTop(index, slots.length) } : undefined}
          title={slots.length > 1 ? `${slot.label} (${slot.kind})` : undefined}
        >
          {slots.length > 1 && (
            <span className={classes.slotLabel}>{slot.label.toLowerCase()}</span>
          )}
        </Handle>
      ))}
      <div className={classes.header}>
        <span className={classes.dot} data-kernel={kernel || undefined} />
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
        <CommentMarker nodeId={id} />
      </div>
      <div className={classes.body}>
        <Text size="sm" c="dimmed" truncate="end">
          {spec
            ? summarize(data.type, params, {
                info,
                kernelConnected: kernelConnections.length > 0,
              }) || ' '
            : `Unknown layer "${data.type}"`}
        </Text>
        {errors[0] && (
          <Text size="xs" c="red" mt={4} lineClamp={3}>
            {errors[0]}
          </Text>
        )}
        <div className={classes.footer}>
          <span
            className={classes.shape}
            title={outputTitle}
            data-kind={kernel ? 'kernel' : undefined}
          >
            {output}
          </span>
          {paramsText(info?.num_params, classical) && (
            <span>{paramsText(info?.num_params, classical)}</span>
          )}
        </div>
      </div>
      <Handle
        type="source"
        position={Position.Right}
        className={classes.handle}
        data-kind={kernel ? 'kernel' : (info?.output_kind ?? outputKind(spec))}
      />
    </div>
  );
}

export const LayerNode = memo(LayerNodeView);
