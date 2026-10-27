import { useMemo } from 'react';
import { IconCheck, IconCopy } from '@tabler/icons-react';
import hljs from 'highlight.js/lib/core';
import python from 'highlight.js/lib/languages/python';
import { ActionIcon, Box, CopyButton, Tooltip } from '@mantine/core';
import classes from './CodeView.module.css';

hljs.registerLanguage('python', python);

/** Syntax-highlighted Python with a copy button. Only Python is bundled to keep this small. */
export function CodeView({ code }: { code: string }) {
  // highlight.js escapes the input, so the resulting HTML is safe to inject.
  const html = useMemo(() => hljs.highlight(code, { language: 'python' }).value, [code]);

  return (
    <Box pos="relative">
      <CopyButton value={code} timeout={1500}>
        {({ copied, copy }) => (
          <Tooltip label={copied ? 'Copied' : 'Copy code'} withArrow position="left">
            <ActionIcon
              variant="default"
              onClick={copy}
              aria-label="Copy code"
              pos="absolute"
              top={10}
              right={10}
            >
              {copied ? <IconCheck size={16} /> : <IconCopy size={16} />}
            </ActionIcon>
          </Tooltip>
        )}
      </CopyButton>
      <pre className={classes.code}>
        <code dangerouslySetInnerHTML={{ __html: html }} />
      </pre>
    </Box>
  );
}
