import { Navigate, useLocation } from 'react-router-dom';

/** Where the editor lives. `/` is the static landing page (frontend/landing/). */
export const EDITOR_PATH = '/app';

/**
 * The app's route for `/`: a navigation to `/` inside the app (a link, `navigate('/')`, the
 * address after logging in) goes on to the editor, with its query, hash and state. Loading
 * `/` in the browser shows the landing page instead, which sends old links on to /app itself.
 */
export function RedirectToEditor() {
  const { search, hash, state } = useLocation();
  return <Navigate to={{ pathname: EDITOR_PATH, search, hash }} state={state} replace />;
}
