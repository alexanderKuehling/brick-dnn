/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Base URL of the Brick DNN API. Empty = same origin (default). */
  readonly VITE_API_URL?: string;
  /** Error monitoring (src/monitoring.ts): the DSN of the Sentry project; empty = off. */
  readonly VITE_SENTRY_DSN?: string;
  readonly VITE_SENTRY_ENVIRONMENT?: string;
  readonly VITE_SENTRY_RELEASE?: string;
  readonly VITE_SENTRY_TRACES_SAMPLE_RATE?: string;
}
