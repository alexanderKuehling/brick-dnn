import { fileURLToPath } from 'node:url';
import react from '@vitejs/plugin-react';
import { defineConfig, loadEnv, searchForWorkspaceRoot } from 'vite';

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');
  // During development, API calls go to the Django dev server via this proxy.
  const backend = env.BACKEND_URL || 'http://127.0.0.1:8000';

  return {
    plugins: [react()],
    // The online demo (`--mode demo`) uses relative URLs, so it works under any path.
    base: mode === 'demo' ? './' : '/',
    resolve: {
      tsconfigPaths: true,
    },
    // The demo's worker loads Pyodide with a dynamic import, which needs an ES module worker.
    worker: {
      format: 'es',
    },
    server: {
      port: 5173,
      proxy: {
        '/api': backend,
        '/admin': backend,
        '/static': backend,
        // OAuth metadata for MCP clients (docs/mcp.md).
        '/.well-known/oauth-protected-resource': backend,
        '/.well-known/oauth-authorization-server': backend,
      },
      fs: {
        // The demo bundles the Python sources of the code generator from ../backend.
        allow: [
          searchForWorkspaceRoot(process.cwd()),
          fileURLToPath(new URL('../backend/brickdnn', import.meta.url)),
        ],
      },
    },
    build: {
      chunkSizeWarningLimit: 1500,
    },
    test: {
      globals: true,
      environment: 'jsdom',
      setupFiles: './vitest.setup.mjs',
      // Tests that click through whole forms take about a second alone, but several times
      // longer when all files run in parallel on a small machine (or CI runner).
      testTimeout: 15000,
    },
  };
});
