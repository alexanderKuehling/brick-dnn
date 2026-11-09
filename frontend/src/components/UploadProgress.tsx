import { Button, Group, Progress, Stack, Text } from '@mantine/core';
import { formatBytes, formatBytesOf } from '@/training/format';
import { PROGRESS_INTERVAL, type UploadBytes } from './useUploadProgress';

interface UploadProgressProps {
  bytes: UploadBytes;
  fileName: string;
  /** Shown once the file is sent, while the server still works on it. */
  processingText?: string;
  /** Offers to cancel the upload while it is sent (later the server finishes it anyway). */
  onCancel?: () => void;
}

/**
 * A running upload: a bar with the percentage and the bytes sent, then a striped bar while the
 * server processes the file (from useUploadProgress).
 */
export function UploadProgress({
  bytes,
  fileName,
  processingText = 'Processing the file on the server …',
  onCancel,
}: UploadProgressProps) {
  const { sent, total } = bytes;
  const processing = total > 0 && sent >= total;
  const share = total > 0 ? Math.min(sent / total, 1) : 0;
  const percent = Math.floor(share * 100);
  return (
    <Stack gap={6}>
      <Group justify="space-between" align="flex-start" wrap="nowrap" gap="xs">
        <Text size="sm" role="status" style={{ minWidth: 0, overflowWrap: 'anywhere' }}>
          {processing ? processingText : `Uploading “${fileName}” …`}
        </Text>
        {!processing && (
          <Text size="sm" fw={600} style={{ flexShrink: 0 }} aria-hidden>
            {percent}%
          </Text>
        )}
      </Group>
      <Progress.Root size="md" transitionDuration={PROGRESS_INTERVAL}>
        {/* While the server works, the bar shows no value (indeterminate). */}
        <Progress.Section
          value={processing ? 100 : share * 100}
          striped={processing}
          animated={processing}
          withAria={false}
          role="progressbar"
          aria-label={`Upload of “${fileName}”`}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={processing ? undefined : percent}
          aria-valuetext={
            processing ? processingText : `${percent}%, ${formatBytesOf(sent, total)}`
          }
        />
      </Progress.Root>
      <Group justify="space-between" gap="xs" mih={22}>
        <Text size="xs" c="dimmed">
          {processing ? `${formatBytes(total)} sent` : formatBytesOf(sent, total)}
        </Text>
        {onCancel && !processing && (
          <Button variant="subtle" color="gray" size="compact-xs" onClick={onCancel}>
            Cancel upload
          </Button>
        )}
      </Group>
    </Stack>
  );
}
