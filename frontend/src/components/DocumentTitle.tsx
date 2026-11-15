import { useEffect, useRef, useState } from 'react';
import { matchPath, useLocation } from 'react-router-dom';
import { VisuallyHidden } from '@mantine/core';
import { i18n, useLanguage } from '@/i18n';

/** The page each address shows, for the window title ("Data · Brick DNN"). */
const PAGES: [pattern: string, title: () => string][] = [
  ['/app', () => i18n.t('nav.builder')],
  ['/data/:prepId?', () => i18n.t('nav.data')],
  ['/projects', () => i18n.t('nav.myNetworks')],
  ['/tokens', () => i18n.t('nav.apiTokens')],
  ['/activity', () => i18n.t('nav.agentActivity')],
  ['/account', () => i18n.t('nav.account')],
  ['/pricing', () => i18n.t('nav.pricing')],
  ['/imprint', () => i18n.t('legal.labels.imprint')],
  ['/privacy', () => i18n.t('legal.labels.privacy')],
  ['/terms', () => i18n.t('legal.labels.terms')],
  ['/refund', () => i18n.t('legal.labels.refund')],
  ['/accessibility', () => i18n.t('legal.labels.accessibility')],
  ['/oauth/authorize', () => i18n.t('a11y.titles.authorize')],
  ['/share/:token', () => i18n.t('a11y.titles.shared')],
  ['/login', () => i18n.t('nav.logIn')],
  ['/register', () => i18n.t('nav.signUp')],
  ['/forgot-password', () => i18n.t('auth.forgot.title')],
  ['/reset-password', () => i18n.t('auth.reset.title')],
  ['/verify-email', () => i18n.t('auth.verify.title')],
  ['/', () => i18n.t('nav.builder')],
];

/** The window title of a page of the app. */
export function pageTitle(pathname: string): string {
  const found = PAGES.find(([pattern]) => matchPath(pattern, pathname));
  const page = found ? found[1]() : i18n.t('auth.notFound.title');
  return i18n.t('a11y.titles.format', { page });
}

/**
 * Names the window after the page (WCAG 2.4.2) and, after a navigation inside the app, says
 * the new page's name politely: screen readers announce a page load, not a route change.
 */
export function DocumentTitle() {
  const { pathname } = useLocation();
  const language = useLanguage();
  const [announced, setAnnounced] = useState('');
  const first = useRef(true);
  useEffect(() => {
    const title = pageTitle(pathname);
    document.title = title;
    if (first.current) {
      first.current = false;
    } else {
      setAnnounced(title);
    }
  }, [pathname, language]);
  return (
    <VisuallyHidden aria-live="polite" aria-atomic="true">
      {announced}
    </VisuallyHidden>
  );
}
