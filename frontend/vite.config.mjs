import react from '@vitejs/plugin-react';
import { defineConfig, loadEnv } from 'vite';
import tsconfigPaths from 'vite-tsconfig-paths';

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');
  // During development, API calls go to the Django dev server via this proxy.
  const backend = env.BACKEND_URL || 'http://127.0.0.1:8000';

  return {
    plugins: [react(), tsconfigPaths()],
    server: {
      port: 5173,
      proxy: {
        '/api': backend,
        '/admin': backend,
        '/static': backend,
      },
    },
    build: {
      chunkSizeWarningLimit: 1500,
    },
    test: {
      globals: true,
      environment: 'jsdom',
      setupFiles: './vitest.setup.mjs',
    },
  };
});
