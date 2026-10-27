import { useEffect, useMemo, useState } from 'react';
import { IconAlertTriangle, IconDownload } from '@tabler/icons-react';
import {
  Alert,
  Button,
  Center,
  Group,
  Loader,
  Modal,
  ScrollArea,
  Switch,
  Text,
  TextInput,
} from '@mantine/core';
import { useDebouncedValue } from '@mantine/hooks';
import { api, type GenerateOptions, type GraphDocument } from '@/api';
import { CodeView } from './CodeView';
import { downloadText } from './download';
import { semanticKey } from './graph';

interface CodeModalProps {
  opened: boolean;
  onClose: () => void;
  doc: GraphDocument;
  filename: string;
  options: GenerateOptions;
  onOptionsChange: (options: GenerateOptions) => void;
}

export default function CodeModal({
  opened,
  onClose,
  doc,
  filename,
  options,
  onOptionsChange,
}: CodeModalProps) {
  const [debounced] = useDebouncedValue(options, 300);
  const key = useMemo(() => semanticKey(doc), [doc]);
  const [code, setCode] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!opened) {
      return undefined;
    }
    let cancelled = false;
    setError(null);
    api
      .generate(JSON.parse(key) as GraphDocument, { ...debounced, filename })
      .then((result) => !cancelled && setCode(result.code))
      .catch((err: Error) => {
        if (!cancelled) {
          setCode(null);
          setError(err.message);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [opened, key, debounced, filename]);

  return (
    <Modal opened={opened} onClose={onClose} title="PyTorch code" size="xl">
      <Group align="flex-end" mb="md" gap="md">
        <TextInput
          label="Class name"
          value={options.class_name ?? ''}
          onChange={(event) =>
            onOptionsChange({ ...options, class_name: event.currentTarget.value })
          }
          w={220}
        />
        <Switch
          label="Runnable script (__main__)"
          checked={options.include_main ?? true}
          onChange={(event) =>
            onOptionsChange({ ...options, include_main: event.currentTarget.checked })
          }
        />
        <Switch
          label="CSV training loop"
          disabled={options.include_main === false}
          checked={options.include_training ?? true}
          onChange={(event) =>
            onOptionsChange({ ...options, include_training: event.currentTarget.checked })
          }
        />
      </Group>
      <Text size="xs" c="dimmed" mb="xs">
        The training loop is added for models with one flat input and one flat output (tabular
        data).
      </Text>

      {error && (
        <Alert color="red" icon={<IconAlertTriangle size={18} />} mb="md">
          {error}
        </Alert>
      )}
      {!code && !error && (
        <Center h={200}>
          <Loader />
        </Center>
      )}
      {code && (
        <ScrollArea.Autosize mah="60vh" type="auto">
          <CodeView code={code} />
        </ScrollArea.Autosize>
      )}

      <Group justify="flex-end" mt="md">
        <Button variant="default" onClick={onClose}>
          Close
        </Button>
        <Button
          leftSection={<IconDownload size={16} />}
          disabled={!code}
          onClick={() => code && downloadText(`${filename}.py`, code, 'text/x-python')}
        >
          Download {filename}.py
        </Button>
      </Group>
    </Modal>
  );
}
