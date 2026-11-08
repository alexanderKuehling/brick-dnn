import { Code, Stack, Tabs, Text } from '@mantine/core';
import { apiBaseUrl } from '@/api';
import {
  claudeCodeCommand,
  curlCommand,
  isLocalUrl,
  mcpJsonConfig,
  mcpUrl,
  TOKEN_PLACEHOLDER,
} from './setup';
import { Snippet } from './Snippet';

interface McpSetupProps {
  /**
   * A token that was just created: the instructions then only show how to use it. Without one,
   * they start with signing in (OAuth), and the token snippets have a placeholder.
   */
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

      <Tabs defaultValue={token ? 'claude-code' : 'claude-app'}>
        <Tabs.List>
          {!token && <Tabs.Tab value="claude-app">Claude app</Tabs.Tab>}
          <Tabs.Tab value="claude-code">Claude Code</Tabs.Tab>
          <Tabs.Tab value="json">Other clients</Tabs.Tab>
        </Tabs.List>
        {!token && (
          <Tabs.Panel value="claude-app" pt="sm">
            <Text size="sm">
              In Claude: <strong>Settings → Connectors → Add custom connector</strong>, and paste
              the MCP server URL. Claude then asks you to sign in to Brick DNN and allow access — no
              token needed.
            </Text>
            {isLocalUrl(url) && (
              <Text size="xs" c="dimmed" mt="xs">
                Claude connects to the server over the internet, so this does not work while Brick
                DNN only runs on your own computer. Use Claude Code there.
              </Text>
            )}
          </Tabs.Panel>
        )}
        <Tabs.Panel value="claude-code" pt="sm">
          {token ? (
            <>
              <Text size="sm" mb={6}>
                Run this in a terminal:
              </Text>
              <Snippet value={claudeCodeCommand(url, secret)} copyLabel="Copy command" />
            </>
          ) : (
            <Stack gap="sm">
              <div>
                <Text size="sm" mb={6}>
                  Run this in a terminal, then type <Code>/mcp</Code> in Claude Code and sign in to
                  Brick DNN in the browser:
                </Text>
                <Snippet value={claudeCodeCommand(url)} copyLabel="Copy command" />
              </div>
              <div>
                <Text size="sm" mb={6}>
                  Or use an API token instead of signing in:
                </Text>
                <Snippet
                  value={claudeCodeCommand(url, secret)}
                  copyLabel="Copy command with token"
                />
              </div>
            </Stack>
          )}
        </Tabs.Panel>
        <Tabs.Panel value="json" pt="sm">
          <Text size="sm" mb={6}>
            {!token && 'Clients that can sign in with OAuth only need the MCP server URL. '}
            For clients that support remote MCP servers with headers, use a token. Where this goes
            depends on the client; see its documentation.
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
