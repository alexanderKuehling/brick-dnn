import { sentryVitePlugin } from '@sentry/vite-plugin';
import react from '@vitejs/plugin-react';
import { defineConfig, loadEnv } from 'vite';
import { landingPages } from './landing/plugin.ts';

// The oldest browsers the app supports (README.md, "Supported browsers"): the first versions
// with everything its code and styles use, e.g. structuredClone, Array.at and CSS :has().
// Vite rewrites newer syntax for them (JavaScript, and CSS such as range media queries, which
// Safari only knows from 16.4), but adds no missing functions: src/polyfills.ts has those that
// dependencies need.
const BROWSERS = ['chrome105', 'edge105', 'firefox121', 'safari15.4', 'ios15.4'];

// Sentry (docs/monitoring.md, "Readable stack traces in the browser"): Brick DNN's Sentry
// organisation in the EU region and its project for the browser app.
const SENTRY = { org: 'brickdnn', project: 'brickdnn-frontend', url: 'https://de.sentry.io' };

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');
  // During development, API calls go to the Django dev server via this proxy. It keeps the
  // Host header (like nginx in production), so that Django builds URLs for this address,
  // e.g. the OAuth endpoints MCP clients sign in with.
  const backend = { target: env.BACKEND_URL || 'http://127.0.0.1:8000', changeOrigin: false };

  // Source maps go to Sentry only when a build has an auth token and a release (the server's
  // image build: frontend/Dockerfile). Without them the build makes no source maps at all, and
  // local builds and CI need no token. The maps are hidden (no sourceMappingURL comment in
  // the bundles) and are not left in dist after the upload, so the site never serves them.
  const release = env.VITE_SENTRY_RELEASE;
  const uploadSourceMaps = Boolean(env.SENTRY_AUTH_TOKEN && release);

  return {
    plugins: [
      react(),
      // The static landing pages at / and /de/; the app itself is app.html (landing/plugin.ts).
      landingPages(),
      uploadSourceMaps &&
        sentryVitePlugin({
          ...SENTRY,
          authToken: env.SENTRY_AUTH_TOKEN,
          // The release the app reports its errors with (src/monitoring.ts).
          release: { name: release, inject: false, setCommits: false },
          sourcemaps: {
            // The maps never reach dist (deleteSourcemapsAfterUpload), and whatever .map file
            // might be there is deleted after the upload.
            filesToDeleteAfterUpload: ['./dist/**/*.map'],
            deleteSourcemapsAfterUpload: true,
          },
          // No data about this build to Sentry's own telemetry project.
          telemetry: false,
          // A failed upload costs readable stack traces, not the deploy.
          errorHandler: (error) => {
            console.warn(`Sentry: source maps not uploaded: ${error.message}`);
          },
        }),
    ],
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
      target: BROWSERS,
      sourcemap: uploadSourceMaps ? 'hidden' : false,
      chunkSizeWarningLimit: 1500,
      rolldownOptions: {
        output: {
          codeSplitting: {
            // The polyfills in a chunk of their own, which main.tsx's chunk imports first. Inside
            // it they would run only after the chunks it imports (Mantine, React, ...).
            groups: [{ name: 'polyfills', test: /[\\/]src[\\/]polyfills\.ts$/, priority: 100 }],
          },
        },
      },
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
