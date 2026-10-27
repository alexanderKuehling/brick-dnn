/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Base URL of the Brick DNN API. Empty = same origin (default). */
  readonly VITE_API_URL?: string;
}
