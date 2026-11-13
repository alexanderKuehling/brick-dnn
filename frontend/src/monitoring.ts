/**
 * Error monitoring with Sentry (docs/monitoring.md). Off unless the app was built with
 * VITE_SENTRY_DSN (on a server: BRICKDNN_SENTRY_FRONTEND_DSN): the online demo and local
 * builds send nothing.
 *
 * Privacy: no user, no IP address, cookies, headers or bodies (dataCollection; the Sentry
 * project also drops IP addresses), no Session Replay. Traces follow requests to this server's API only.
 */
import { useEffect } from 'react';
import * as Sentry from '@sentry/react';
import {
  createRoutesFromChildren,
  matchRoutes,
  Routes,
  useLocation,
  useNavigationType,
} from 'react-router-dom';

const env = import.meta.env;

function rate(value: string | undefined, fallback: number): number {
  const parsed = value ? Number(value) : NaN;
  return Number.isFinite(parsed) ? Math.min(Math.max(parsed, 0), 1) : fallback;
}

/** Whether Sentry was started (a DSN was built in). */
export const MONITORING = Boolean(env.VITE_SENTRY_DSN);

if (MONITORING) {
  Sentry.init({
    dsn: env.VITE_SENTRY_DSN,
    environment: env.VITE_SENTRY_ENVIRONMENT || (env.DEV ? 'development' : 'production'),
    release: env.VITE_SENTRY_RELEASE || undefined,
    // Nothing about the person: no user, IP address, cookies, headers, bodies or query strings.
    dataCollection: {
      userInfo: false,
      cookies: false,
      httpHeaders: false,
      httpBodies: [],
      urlQueryParams: false,
    },
    integrations: [
      Sentry.reactRouterV7BrowserTracingIntegration({
        useEffect,
        useLocation,
        useNavigationType,
        createRoutesFromChildren,
        matchRoutes,
      }),
    ],
    // Every trace locally, a tenth of the page loads in production.
    tracesSampleRate: rate(env.VITE_SENTRY_TRACES_SAMPLE_RATE, env.DEV ? 1 : 0.1),
    // Trace headers only to this server's API (same origin), never to other sites.
    tracePropagationTargets: [/^\/api\//],
  });
}

/** <Routes> that name page loads and navigations after the route (e.g. /data/:prepId?). */
export const MonitoredRoutes = MONITORING ? Sentry.withSentryReactRouterV7Routing(Routes) : Routes;

/** Errors React reports while rendering, for createRoot (React 19). */
export const rootErrorHandlers = MONITORING
  ? {
      onUncaughtError: Sentry.reactErrorHandler(),
      onCaughtError: Sentry.reactErrorHandler(),
      onRecoverableError: Sentry.reactErrorHandler(),
    }
  : {};
