import { IconCheck, IconCopy } from '@tabler/icons-react';
import { ActionIcon, CopyButton, Tooltip } from '@mantine/core';
import { i18n, useTranslation } from '@/i18n';
import classes from './Tokens.module.css';

interface SnippetProps {
  value: string;
  /** Tooltip and accessible name of the copy button, e.g. "Copy command". */
  copyLabel: string;
}

/** A command, URL or configuration in monospace with a copy button. Long lines wrap. */
export function Snippet({ value, copyLabel }: SnippetProps) {
  useTranslation();
  return (
    <div className={classes.snippet}>
      <pre className={classes.code}>{value}</pre>
      <CopyButton value={value} timeout={1500}>
        {({ copied, copy }) => (
          <Tooltip
            label={copied ? i18n.t('ui.tokens.Snippet.copied') : copyLabel}
            withArrow
            position="left"
          >
            <ActionIcon variant="default" onClick={copy} aria-label={copyLabel}>
              {copied ? <IconCheck size={16} /> : <IconCopy size={16} />}
            </ActionIcon>
          </Tooltip>
        )}
      </CopyButton>
    </div>
  );
}
