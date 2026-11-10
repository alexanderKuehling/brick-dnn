/**
 * Kernel hyperparameters that a Gaussian process fit left at one of their bounds (the server
 * finds them after the fit, see brickdnn/training/kernel_bounds.py). scikit-learn only warns
 * about this in the log; here it says which brick, what it may mean and how to change it, with
 * a button that sets the suggested bounds on the brick in the builder (and can undo it).
 */
import { useContext, useState } from 'react';
import { IconAlertTriangle } from '@tabler/icons-react';
import { Alert, Button, Group, Stack, Text } from '@mantine/core';
import type { KernelBound, Params } from '@/api/types';
import { BuilderContext } from '@/builder/BuilderContext';

/** 1e-05, 0.001, 50, 1e+05: short, like the server's messages. */
export function formatBound(value: unknown): string {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return String(value);
  }
  const abs = Math.abs(value);
  if (abs !== 0 && (abs < 1e-3 || abs >= 1e5)) {
    return value.toExponential().replace(/e([+-])(\d)$/, 'e$10$2');
  }
  return String(Number(value.toPrecision(3)));
}

function boundsText(bound: KernelBound): string {
  const suggestion = bound.suggestion ?? {};
  const low = suggestion[`${bound.parameter}_min`];
  const high = suggestion[`${bound.parameter}_max`];
  return `${bound.label}: custom bounds ${formatBound(low)} … ${formatBound(high)}`;
}

const keyOf = (bound: KernelBound) => `${bound.node}:${bound.parameter}:${bound.side}`;

export function KernelBoundsHint({ bounds }: { bounds: KernelBound[] }) {
  const builder = useContext(BuilderContext);
  // The parameters of each brick before a suggestion was applied (for undo).
  const [applied, setApplied] = useState<Record<string, { node: string; before: Params }>>({});

  if (bounds.length === 0) {
    return null;
  }
  const canApply = (bound: KernelBound) =>
    Boolean(
      builder?.patchParams &&
      !builder.readOnly &&
      bound.node &&
      bound.suggestion &&
      builder.analysis?.nodes[bound.node]?.type === bound.type
    );

  const apply = (bound: KernelBound) => {
    if (!builder?.patchParams || !bound.node || !bound.suggestion) {
      return;
    }
    const before = builder.patchParams(bound.node, bound.type, bound.suggestion);
    if (before) {
      setApplied((current) => ({ ...current, [keyOf(bound)]: { node: bound.node!, before } }));
      builder.focusNode(bound.node);
    }
  };

  const undo = (bound: KernelBound) => {
    const entry = applied[keyOf(bound)];
    if (!builder || !entry) {
      return;
    }
    builder.updateNode(entry.node, { params: entry.before });
    setApplied(({ [keyOf(bound)]: _, ...rest }) => rest);
  };

  return (
    <Alert
      color="yellow"
      variant="light"
      icon={<IconAlertTriangle size={18} />}
      title={
        bounds.length === 1
          ? 'A kernel hyperparameter ended at its bound'
          : 'Kernel hyperparameters ended at their bounds'
      }
      aria-label="Kernel hyperparameters at a bound"
    >
      <Stack gap="xs">
        {bounds.map((bound) => {
          const done = applied[keyOf(bound)];
          return (
            <div key={keyOf(bound)}>
              <Text size="sm">
                {bound.message}
                {bound.classes?.length ? ` (classes: ${bound.classes.join(', ')})` : ''}
              </Text>
              {canApply(bound) && (
                <Group gap="xs" mt={4} wrap="wrap">
                  {done ? (
                    <>
                      <Text size="xs" c="dimmed">
                        Set on {bound.name}: {boundsText(bound)}. Train again to see the effect.
                      </Text>
                      <Button size="compact-xs" variant="subtle" onClick={() => undo(bound)}>
                        Undo
                      </Button>
                    </>
                  ) : (
                    <Button size="compact-xs" variant="light" onClick={() => apply(bound)}>
                      Set {bound.side === 'lower' ? 'a higher lower' : 'a higher upper'} bound on{' '}
                      {bound.name}
                    </Button>
                  )}
                </Group>
              )}
            </div>
          );
        })}
      </Stack>
    </Alert>
  );
}
