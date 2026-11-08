/** Shown in the setup snippets instead of a token. */
export const TOKEN_PLACEHOLDER = '<YOUR_TOKEN>';

/** The name of the MCP server in the assistants' configuration. */
const SERVER_NAME = 'brick-dnn';

/** The MCP endpoint (Streamable HTTP) of the API at `apiUrl`. */
export function mcpUrl(apiUrl: string): string {
  return `${apiUrl}/api/mcp/`;
}

/** Adds the MCP server to Claude Code. */
export function claudeCodeCommand(url: string, token: string): string {
  return `claude mcp add --transport http ${SERVER_NAME} ${url} --header "Authorization: Bearer ${token}"`;
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
