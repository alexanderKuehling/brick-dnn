import { Code, Stack, Tabs, Text } from '@mantine/core';
import { apiBaseUrl } from '@/api';
import { claudeCodeCommand, curlCommand, mcpJsonConfig, mcpUrl, TOKEN_PLACEHOLDER } from './setup';
import { Snippet } from './Snippet';

interface McpSetupProps {
  /** A token that was just created; without one, the snippets have a placeholder. */
  token?: string | null;
}

/** The MCP server URL and how to add it to an assistant (or use a token in a script). */
export function McpSetup({ token }: McpSetupProps) {
  const apiUrl = apiBaseUrl();
  const url = mcpUrl(apiUrl);
  const secret = token || TOKEN_PLACEHOLDER;

  return (
    <Stack gap="md">
      <div>
        <Text size="sm" fw={500} mb={6}>
          MCP server URL
        </Text>
        <Snippet value={url} copyLabel="Copy MCP server URL" />
      </div>

      <Tabs defaultValue="claude-code">
        <Tabs.List>
          <Tabs.Tab value="claude-code">Claude Code</Tabs.Tab>
          <Tabs.Tab value="json">Other MCP clients</Tabs.Tab>
        </Tabs.List>
        <Tabs.Panel value="claude-code" pt="sm">
          <Text size="sm" mb={6}>
            Run this in a terminal:
          </Text>
          <Snippet value={claudeCodeCommand(url, secret)} copyLabel="Copy command" />
        </Tabs.Panel>
        <Tabs.Panel value="json" pt="sm">
          <Text size="sm" mb={6}>
            For clients that support remote MCP servers with headers. Where this goes depends on the
            client; see its documentation.
          </Text>
          <Snippet value={mcpJsonConfig(url, secret)} copyLabel="Copy configuration" />
        </Tabs.Panel>
      </Tabs>

      <div>
        <Text size="sm" mb={6}>
          Your own scripts can use a token with the REST API, too:
        </Text>
        <Snippet value={curlCommand(apiUrl, secret)} copyLabel="Copy curl command" />
      </div>

      {!token && (
        <Text size="xs" c="dimmed">
          Replace <Code>{TOKEN_PLACEHOLDER}</Code> with one of your tokens. A token is only shown
          once, right after you create it.
        </Text>
      )}
    </Stack>
  );
}
