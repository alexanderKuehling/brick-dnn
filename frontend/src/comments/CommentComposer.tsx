import { useState } from 'react';
import { Button, Group, Textarea } from '@mantine/core';
import { ApiError } from '@/api';

/** The server's message for the text, or the general one. */
export function commentError(error: unknown): string {
  if (error instanceof ApiError) {
    const text = (error.data as { text?: unknown } | null)?.text;
    if (Array.isArray(text) && text.length > 0) {
      return text.map(String).join(' ');
    }
  }
  return (error as Error).message;
}

export const MAX_COMMENT_LENGTH = 5000;

interface CommentComposerProps {
  /** The field's accessible name. */
  label: string;
  placeholder?: string;
  submitLabel: string;
  initialText?: string;
  autoFocus?: boolean;
  /** Saves the text; a rejection is shown below the field and keeps the text. */
  onSubmit: (text: string) => Promise<unknown>;
  /** Shows a cancel button. */
  onCancel?: () => void;
}

/** A text field for a comment, a reply or an edit. Ctrl+Enter (⌘+Enter) sends it. */
export function CommentComposer({
  label,
  placeholder,
  submitLabel,
  initialText = '',
  autoFocus = false,
  onSubmit,
  onCancel,
}: CommentComposerProps) {
  const [text, setText] = useState(initialText);
  const [error, setError] = useState<string | null>(null);
  const [sending, setSending] = useState(false);
  const empty = text.trim() === '';

  const submit = async () => {
    if (empty || sending) {
      return;
    }
    setSending(true);
    setError(null);
    try {
      await onSubmit(text.trim());
      setText('');
    } catch (err) {
      setError(commentError(err));
    } finally {
      setSending(false);
    }
  };

  return (
    <form
      onSubmit={(event) => {
        event.preventDefault();
        void submit();
      }}
    >
      <Textarea
        aria-label={label}
        placeholder={placeholder ?? label}
        value={text}
        onChange={(event) => setText(event.currentTarget.value)}
        onKeyDown={(event) => {
          if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) {
            event.preventDefault();
            void submit();
          } else if (event.key === 'Escape' && onCancel) {
            onCancel();
          }
        }}
        // Escape cancels the edit, it does not close the drawer around it.
        data-mantine-stop-propagation={onCancel ? 'true' : undefined}
        error={error}
        autosize
        minRows={2}
        maxRows={8}
        maxLength={MAX_COMMENT_LENGTH}
        autoFocus={autoFocus}
        data-autofocus={autoFocus || undefined}
      />
      <Group justify="flex-end" gap="xs" mt={6}>
        {onCancel && (
          <Button variant="subtle" color="gray" size="xs" onClick={onCancel}>
            Cancel
          </Button>
        )}
        <Button type="submit" size="xs" loading={sending} disabled={empty}>
          {submitLabel}
        </Button>
      </Group>
    </form>
  );
}
