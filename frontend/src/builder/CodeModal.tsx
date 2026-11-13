import { useEffect, useMemo, useState } from 'react';
import { IconAlertTriangle, IconDownload } from '@tabler/icons-react';
import {
  Alert,
  Box,
  Button,
  Center,
  Group,
  Loader,
  Modal,
  ScrollArea,
  Switch,
  Text,
  TextInput,
  useMantineTheme,
} from '@mantine/core';
import { useDebouncedValue, useMediaQuery } from '@mantine/hooks';
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
  /** classical: the network is a scikit-learn model (Gaussian process, SVM, gradient boosting). */
  family?: 'neural' | 'classical';
}

export default function CodeModal({
  opened,
  onClose,
  doc,
  filename,
  options,
  onOptionsChange,
  family = 'neural',
}: CodeModalProps) {
  const classical = family === 'classical';
  const [debounced] = useDebouncedValue(options, 300);
  const key = useMemo(() => semanticKey(doc), [doc]);
  const [code, setCode] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const theme = useMantineTheme();
  // Phones (also in landscape) get the whole screen, with the code filling the free height.
  const fullScreen = Boolean(
    useMediaQuery(`(max-width: ${theme.breakpoints.sm}), (max-height: 500px)`)
  );

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
    <Modal
      opened={opened}
      onClose={onClose}
      title={classical ? 'scikit-learn code' : 'PyTorch code'}
      size="xl"
      fullScreen={fullScreen}
      styles={
        fullScreen
          ? {
              content: { display: 'flex', flexDirection: 'column' },
              body: { flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column' },
            }
          : undefined
      }
    >
      <Group align="flex-end" mb="md" gap="md">
        {/* A classical model is built by make_model(): there is no class to name. */}
        {!classical && (
          <TextInput
            label="Class name"
            value={options.class_name ?? ''}
            onChange={(event) =>
              onOptionsChange({ ...options, class_name: event.currentTarget.value })
            }
            w={220}
          />
        )}
        <Switch
          label="Runnable script (__main__)"
          checked={options.include_main ?? true}
          onChange={(event) =>
            onOptionsChange({ ...options, include_main: event.currentTarget.checked })
          }
        />
        <Switch
          label={classical ? 'Fit to a CSV file (--csv)' : 'CSV training loop'}
          disabled={options.include_main === false}
          checked={options.include_training ?? true}
          onChange={(event) =>
            onOptionsChange({ ...options, include_training: event.currentTarget.checked })
          }
        />
      </Group>
      <Text size="xs" c="dimmed" mb="xs">
        {classical
          ? 'make_model() returns a scikit-learn pipeline that standardises the features. The script fits it to a CSV file, saves it with joblib (--save) and predicts new rows (--predict), also with a model trained in Brick DNN.'
          : 'The training loop is added for models with one flat input and one flat output (tabular data).'}
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
      {code &&
        (fullScreen ? (
          <ScrollArea type="auto" style={{ flex: 1, minHeight: 120 }}>
            <CodeView code={code} />
          </ScrollArea>
        ) : (
          <ScrollArea.Autosize mah="60vh" type="auto">
            <CodeView code={code} />
          </ScrollArea.Autosize>
        ))}

      <Group justify="flex-end" mt="md">
        <Button variant="default" onClick={onClose}>
          Close
        </Button>
        <Button
          leftSection={<IconDownload size={16} />}
          disabled={!code}
          onClick={() => code && downloadText(`${filename}.py`, code, 'text/x-python')}
        >
          {/* One inline span: the button label is a flex box, which would drop the space. */}
          <span>
            Download{' '}
            <Box component="span" visibleFrom="xs">
              {filename}
            </Box>
            .py
          </span>
        </Button>
      </Group>
    </Modal>
  );
}
