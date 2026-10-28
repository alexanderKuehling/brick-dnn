/**
 * The online demo is a static build (`npm run build:demo`) that needs no server: the code
 * generator runs in the browser with Pyodide, and accounts and training are switched off.
 */
export const DEMO = import.meta.env.MODE === 'demo';
