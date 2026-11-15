import { Code, Stack, Tabs, Text } from '@mantine/core';
import { apiBaseUrl } from '@/api';
import { i18n, Trans, useTranslation } from '@/i18n';
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
  useTranslation();
  const apiUrl = apiBaseUrl();
  const url = mcpUrl(apiUrl);
  const secret = token || TOKEN_PLACEHOLDER;

  return (
    <Stack gap="md">
      <div>
        <Text size="sm" fw={500} mb={6}>
          {i18n.t('ui.tokens.McpSetup.mcpServerUrl')}
        </Text>
        <Snippet value={url} copyLabel="Copy MCP server URL" />
      </div>

      <Tabs defaultValue={token ? 'claude-code' : 'claude-app'}>
        <Tabs.List>
          {!token && (
            <Tabs.Tab value="claude-app">{i18n.t('ui.tokens.McpSetup.claudeApp')}</Tabs.Tab>
          )}
          <Tabs.Tab value="claude-code">{i18n.t('ui.tokens.McpSetup.claudeCode')}</Tabs.Tab>
          <Tabs.Tab value="json">{i18n.t('ui.tokens.McpSetup.otherClients')}</Tabs.Tab>
        </Tabs.List>
        {!token && (
          <Tabs.Panel value="claude-app" pt="sm">
            <Text size="sm">
              <Trans
                i18nKey="ui.tokens.McpSetup.inClaudeSettingsConnectorsAdd"
                components={[<strong />]}
              />
            </Text>
            {isLocalUrl(url) && (
              <Text size="xs" c="dimmed" mt="xs">
                {i18n.t('ui.tokens.McpSetup.claudeConnectsToTheServer')}
              </Text>
            )}
          </Tabs.Panel>
        )}
        <Tabs.Panel value="claude-code" pt="sm">
          {token ? (
            <>
              <Text size="sm" mb={6}>
                {i18n.t('ui.tokens.McpSetup.runThisInATerminal')}
              </Text>
              <Snippet value={claudeCodeCommand(url, secret)} copyLabel="Copy command" />
            </>
          ) : (
            <Stack gap="sm">
              <div>
                <Text size="sm" mb={6}>
                  <Trans i18nKey="ui.tokens.McpSetup.runThisInATerminal2" components={[<Code />]} />
                </Text>
                <Snippet value={claudeCodeCommand(url)} copyLabel="Copy command" />
              </div>
              <div>
                <Text size="sm" mb={6}>
                  {i18n.t('ui.tokens.McpSetup.orUseAnApiToken')}
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
            {!token && i18n.t('ui.tokens.McpSetup.clientsThatCanSignIn')}
            {i18n.t('ui.tokens.McpSetup.forClientsThatSupportRemote')}
          </Text>
          <Snippet value={mcpJsonConfig(url, secret)} copyLabel="Copy configuration" />
        </Tabs.Panel>
      </Tabs>

      <div>
        <Text size="sm" mb={6}>
          {i18n.t('ui.tokens.McpSetup.yourOwnScriptsCanUse')}
        </Text>
        <Snippet value={curlCommand(apiUrl, secret)} copyLabel="Copy curl command" />
      </div>

      {!token && (
        <Text size="xs" c="dimmed">
          <Trans
            i18nKey="ui.tokens.McpSetup.replaceWithOneOfYour"
            values={{ TOKEN_PLACEHOLDER }}
            components={[<Code />]}
          />
        </Text>
      )}
    </Stack>
  );
}
