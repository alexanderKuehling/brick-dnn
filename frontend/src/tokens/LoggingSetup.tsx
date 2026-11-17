import { Stack, Text } from '@mantine/core';
import { apiBaseUrl } from '@/api';
import { useTranslation } from '@/i18n';
import { Snippet } from './Snippet';

const TOKEN_PLACEHOLDER = '<your token>';

/** The commands that log a training on the user's own machine (the exported script). */
export function loggingCommands(token: string | null, url: string): string {
  return [
    `export BRICKDNN_TOKEN=${token ?? TOKEN_PLACEHOLDER}`,
    `export BRICKDNN_URL=${url}`,
    'python my_network.py --csv data.csv --epochs 20 --test-split 0.2',
  ].join('\n');
}

/** Logging from a training loop of one's own with the client of the brickdnn package. */
export function loggingPython(): string {
  return [
    'from brickdnn.tracking import BrickDNNRun  # pip install brickdnn (see the README)',
    '',
    'with BrickDNNRun.start(name="My run", config={"learning_rate": 1e-3}) as run:',
    '    for epoch in range(1, 21):',
    '        ...  # train and validate',
    '        run.log(epoch=epoch, train_loss=train_loss, val_loss=val_loss)',
    '    run.finish(summary={"test_accuracy": accuracy})',
  ].join('\n');
}

/**
 * How to log a training that runs elsewhere (a laptop, a GPU server) into Brick DNN: the
 * exported script logs when BRICKDNN_TOKEN is set; the token never goes into the script.
 */
export function LoggingSetup({ token = null }: { token?: string | null }) {
  const { t } = useTranslation();
  const url = apiBaseUrl();
  return (
    <Stack gap="sm">
      {(['step1', 'step2', 'step3'] as const).map((step, index) => (
        <Text key={step} size="sm" maw={720}>
          <b>{index + 1}.</b> {t(`experiments.logging.${step}`)}
        </Text>
      ))}
      <Snippet
        value={loggingCommands(token, url)}
        copyLabel={t('experiments.logging.copyCommands')}
      />
      <Text size="sm" maw={720}>
        {t('experiments.logging.ownLoop')}
      </Text>
      <Snippet value={loggingPython()} copyLabel={t('experiments.logging.copyPython')} />
      <Text size="xs" c="dimmed" maw={720}>
        {t('experiments.logging.privacy')}
      </Text>
    </Stack>
  );
}
