/** Shown in the setup snippets instead of a token. */
export const TOKEN_PLACEHOLDER = '<YOUR_TOKEN>';

/** The name of the MCP server in the assistants' configuration. */
const SERVER_NAME = 'brick-dnn';

/** The MCP endpoint (Streamable HTTP) of the API at `apiUrl`. */
export function mcpUrl(apiUrl: string): string {
  return `${apiUrl}/api/mcp/`;
}

/**
 * Adds the MCP server to Claude Code. Without a token, Claude Code asks you to sign in in the
 * browser (OAuth) when you connect it with /mcp.
 */
export function claudeCodeCommand(url: string, token?: string): string {
  const command = `claude mcp add --transport http ${SERVER_NAME} ${url}`;
  return token ? `${command} --header "Authorization: Bearer ${token}"` : command;
}

/** Whether `url` is on this computer, where apps that connect from the internet cannot reach it. */
export function isLocalUrl(url: string): boolean {
  try {
    const { hostname } = new URL(url);
    return (
      hostname === 'localhost' ||
      hostname.endsWith('.localhost') ||
      hostname === '[::1]' ||
      /^127\.\d+\.\d+\.\d+$/.test(hostname)
    );
  } catch {
    return false;
  }
}

/** The usual JSON configuration of MCP clients that support remote servers with headers. */
export function mcpJsonConfig(url: string, token: string): string {
  const config = {
    mcpServers: {
      [SERVER_NAME]: { type: 'http', url, headers: { Authorization: `Bearer ${token}` } },
    },
  };
  return JSON.stringify(config, null, 2);
}

/** A REST API call with the token, for scripts. */
export function curlCommand(apiUrl: string, token: string): string {
  return `curl -H "Authorization: Bearer ${token}" ${apiUrl}/api/projects/`;
}
