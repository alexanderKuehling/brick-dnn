import { useEffect, useState } from 'react';
import { IconStack2 } from '@tabler/icons-react';
import {
  Alert,
  Button,
  Group,
  Modal,
  NumberInput,
  SegmentedControl,
  Stack,
  Table,
  Text,
  TextInput,
} from '@mantine/core';
import { api, type EnsembleMethod, type RunDetail, type RunSummary } from '@/api';
import { defaultEnsembleName, METHOD_LABELS, methodText, shares } from './ensemble';
import { formatPercent } from './format';

interface CombineDialogProps {
  opened: boolean;
  /** The selected finished runs, in the order they were selected. */
  runs: RunSummary[];
  /** The open network the ensemble belongs to (one the user owns or edits), if any. */
  project: number | null;
  onClose: () => void;
  onCreated: (run: RunDetail) => void;
}

/**
 * "Combine into an ensemble": the name, the method (weighted average or stacking) and, for the
 * average, a weight per run. The server checks that the runs fit together and says why not.
 */
export function CombineDialog({ opened, runs, project, onClose, onCreated }: CombineDialogProps) {
  const [name, setName] = useState('');
  const [method, setMethod] = useState<EnsembleMethod>('average');
  const [weights, setWeights] = useState<number[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const key = runs.map((run) => run.id).join(',');

  useEffect(() => {
    if (opened) {
      setName(defaultEnsembleName(runs));
      setMethod('average');
      setWeights(runs.map(() => 1));
      setError(null);
    }
  }, [opened, key]);

  const task = runs[0]?.task;
  const shown = shares(weights.length === runs.length ? weights : runs.map(() => 1));
  const noWeight = method === 'average' && weights.every((w) => !(w > 0));

  const create = async () => {
    setBusy(true);
    setError(null);
    try {
      const run = await api.training.ensembles.create({
        members: runs.map((r) => r.id),
        method,
        ...(method === 'average' ? { weights } : {}),
        name: name.trim() || undefined,
        project,
      });
      onCreated(run);
      onClose();
    } catch (reason) {
      setError((reason as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <Modal
      opened={opened}
      onClose={onClose}
      title={
        <Group gap="xs">
          <IconStack2 size={18} />
          <Text fw={600}>Combine into an ensemble</Text>
        </Group>
      }
      size="lg"
    >
      <Stack gap="sm">
        <Text size="sm" c="dimmed">
          The ensemble predicts with all {runs.length} runs and combines what they predict. It is
          measured on the validation and test rows of their dataset, next to each run on its own.
        </Text>
        <TextInput
          label="Name"
          value={name}
          maxLength={120}
          onChange={(event) => setName(event.currentTarget.value)}
        />
        <div>
          <Text size="sm" fw={500} mb={4}>
            Method
          </Text>
          <SegmentedControl
            fullWidth
            aria-label="Method"
            value={method}
            onChange={(value) => setMethod(value as EnsembleMethod)}
            data={[
              { value: 'average', label: METHOD_LABELS.average },
              { value: 'stacking', label: METHOD_LABELS.stacking },
            ]}
          />
          <Text size="xs" c="dimmed" mt={4}>
            {methodText(method, task)}
            {method === 'stacking' &&
              ' It needs validation rows; without test rows its metrics are optimistic.'}
          </Text>
        </div>
        <Table fz="sm" aria-label="Members">
          <Table.Thead>
            <Table.Tr>
              <Table.Th>Run</Table.Th>
              {method === 'average' && <Table.Th w={120}>Weight</Table.Th>}
              {method === 'average' && <Table.Th w={70}>Share</Table.Th>}
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            {runs.map((run, index) => (
              <Table.Tr key={run.id}>
                <Table.Td>
                  <Text size="sm" fw={500}>
                    {run.name}
                  </Text>
                  <Text size="xs" c="dimmed">
                    {[
                      run.family === 'classical'
                        ? 'classical model'
                        : run.family === 'hybrid'
                          ? 'hybrid network'
                          : 'neural network',
                      run.dataset.name,
                    ]
                      .filter(Boolean)
                      .join(' · ')}
                  </Text>
                </Table.Td>
                {method === 'average' && (
                  <Table.Td>
                    <NumberInput
                      size="xs"
                      min={0}
                      step={0.5}
                      decimalScale={3}
                      aria-label={`Weight of ${run.name}`}
                      value={weights[index] ?? 1}
                      onChange={(value) =>
                        setWeights((current) =>
                          current.map((w, i) => (i === index ? Number(value) || 0 : w))
                        )
                      }
                    />
                  </Table.Td>
                )}
                {method === 'average' && (
                  <Table.Td>
                    <Text size="xs" c="dimmed">
                      {formatPercent(shown[index], 0)}
                    </Text>
                  </Table.Td>
                )}
              </Table.Tr>
            ))}
          </Table.Tbody>
        </Table>
        {noWeight && (
          <Text size="xs" c="red">
            At least one weight must be above 0.
          </Text>
        )}
        {error && (
          <Alert color="red" variant="light" title="Cannot combine these runs">
            <Text size="sm">{error}</Text>
          </Alert>
        )}
        <Group justify="flex-end" gap="xs">
          <Button variant="default" onClick={onClose}>
            Cancel
          </Button>
          <Button
            leftSection={<IconStack2 size={16} />}
            loading={busy}
            disabled={noWeight}
            onClick={() => void create()}
          >
            Create ensemble
          </Button>
        </Group>
      </Stack>
    </Modal>
  );
}
