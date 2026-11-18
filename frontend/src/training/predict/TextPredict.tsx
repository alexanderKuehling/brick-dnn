import { useState } from 'react';
import { IconWand } from '@tabler/icons-react';
import { Badge, Button, Group, Paper, Stack, Text, Textarea } from '@mantine/core';
import type { Prediction, PredictionForm, PredictionResult } from '@/api';
import { currentLocale, i18n, useTranslation } from '@/i18n';
import { Messages, ResultCard } from './PredictionResults';
import { isProblem, predictAll, type PredictionProblem } from './requests';

/** The words a network read, as small badges; words it does not know show as `<unk>`. */
export function TokenList({ tokens }: { tokens: string[] }) {
  return (
    <Group gap={4} aria-label={i18n.t('ui.training.TextPredict.tokens')}>
      {tokens.map((token, index) => (
        <Badge
          // Tokens repeat, so their position is part of the key.
          key={`${index}-${token}`}
          size="sm"
          radius="sm"
          variant={token === '<unk>' ? 'filled' : 'light'}
          color={token === '<unk>' ? 'orange' : 'gray'}
          tt="none"
        >
          {token}
        </Badge>
      ))}
    </Group>
  );
}

/**
 * Try a network trained on a built-in text dataset: type texts (one per line) or pick test
 * texts of the dataset. The server splits them into words and looks them up in the
 * vocabulary of the training; the answer shows the words the network read.
 */
export function TextPredict({ runId, form }: { runId: string; form: PredictionForm }) {
  useTranslation();
  const text = form.text!;
  const [value, setValue] = useState(text.examples[0]?.text ?? '');
  const [busy, setBusy] = useState(false);
  const [problem, setProblem] = useState<PredictionProblem | null>(null);
  const [results, setResults] = useState<{
    texts: string[];
    predictions: Prediction[];
    task: PredictionResult['task'];
  } | null>(null);

  const lines = value
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean)
    .slice(0, form.max_rows);
  const labels = new Map(text.examples.map((example) => [example.text, example.class]));

  const predict = async () => {
    setBusy(true);
    setProblem(null);
    setResults(null);
    try {
      const answer = await predictAll(
        runId,
        lines,
        { top_k: Math.min(3, form.max_top_k) },
        form.max_rows
      );
      setResults({ texts: lines, predictions: answer.predictions, task: answer.task });
    } catch (error) {
      setProblem(isProblem(error) ? error : { message: String(error), errors: [], budget: false });
    } finally {
      setBusy(false);
    }
  };

  return (
    <Stack gap="sm">
      <Text size="xs" c="dimmed">
        {i18n.t('ui.training.TextPredict.howTextsAreRead', {
          words: text.vocab_size.toLocaleString(currentLocale()),
          length: text.length,
        })}
      </Text>
      <Textarea
        label={i18n.t('ui.training.TextPredict.texts')}
        description={i18n.t('ui.training.TextPredict.oneTextPerLine')}
        value={value}
        onChange={(event) => setValue(event.currentTarget.value)}
        autosize
        minRows={2}
        maxRows={8}
      />
      {text.examples.length > 0 && (
        <Group gap={6} align="center">
          <Text size="xs" c="dimmed">
            {i18n.t('ui.training.TextPredict.testTexts')}
          </Text>
          {text.examples.map((example, index) => (
            <Button
              key={example.text}
              size="compact-xs"
              variant="default"
              title={example.text}
              onClick={() => setValue(example.text)}
            >
              {i18n.t('ui.training.TextPredict.example', { number: index + 1 })}
            </Button>
          ))}
          <Button
            size="compact-xs"
            variant="subtle"
            onClick={() => setValue(text.examples.map((example) => example.text).join('\n'))}
          >
            {i18n.t('ui.training.TextPredict.all')}
          </Button>
        </Group>
      )}
      <Group>
        <Button
          leftSection={<IconWand size={16} />}
          loading={busy}
          disabled={lines.length === 0}
          onClick={() => void predict()}
        >
          {i18n.t('ui.training.TextPredict.predict')}
        </Button>
      </Group>
      {problem && (
        <Messages
          color={problem.budget ? 'yellow' : 'red'}
          title={
            problem.budget
              ? i18n.t('ui.training.TextPredict.noComputeLeft')
              : i18n.t('ui.training.TextPredict.cannotPredict')
          }
          messages={problem.errors.length > 0 ? problem.errors : [problem.message]}
        />
      )}
      {results?.predictions.map((prediction, index) => (
        <Paper
          // One card per line of the request, in its order.
          key={index}
          withBorder
          p="xs"
          radius="sm"
        >
          <Stack gap={6}>
            <Text size="xs" c="dimmed">
              {i18n.t('ui.training.TextPredict.whatTheNetworkRead')}
            </Text>
            <TokenList tokens={prediction.tokens ?? []} />
            {(prediction.unknown_words?.length ?? 0) > 0 && (
              <Text size="xs" c="orange">
                {i18n.t('ui.training.TextPredict.unknownWords', {
                  words: (prediction.unknown_words ?? []).join(', '),
                })}
              </Text>
            )}
            <ResultCard
              prediction={prediction}
              task={results.task}
              actual={labels.get(results.texts[index]) ?? null}
            />
          </Stack>
        </Paper>
      ))}
    </Stack>
  );
}
