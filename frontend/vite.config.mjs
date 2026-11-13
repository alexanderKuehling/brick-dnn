import react from '@vitejs/plugin-react';
import { defineConfig, loadEnv } from 'vite';

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');
  // During development, API calls go to the Django dev server via this proxy. It keeps the
  // Host header (like nginx in production), so that Django builds URLs for this address,
  // e.g. the OAuth endpoints MCP clients sign in with.
  const backend = { target: env.BACKEND_URL || 'http://127.0.0.1:8000', changeOrigin: false };

  return {
    plugins: [react()],
    resolve: {
      tsconfigPaths: true,
    },
    server: {
      port: 5173,
      proxy: {
        // Live editing (WebSockets); `manage.py runserver` serves them too.
        '/api/ws': { ...backend, ws: true },
        '/api': backend,
        '/admin': backend,
        '/static': backend,
        // OAuth metadata for MCP clients (docs/mcp.md).
        '/.well-known/oauth-protected-resource': backend,
        '/.well-known/oauth-authorization-server': backend,
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
