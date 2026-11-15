import { Link } from 'react-router-dom';
import { Anchor } from '@mantine/core';
import { useTranslation } from '@/i18n';
import { legalPages, useLegalInfo } from './useLegalInfo';
import classes from './Legal.module.css';

/** "Imprint · Privacy · Terms · Refunds · Accessibility · Contact": the legal pages this server
 * has (nothing if it has none). Contact leads to the imprint, which has the operator's address.
 * The links say their short name (the full one is their title): an accessible name that
 * differs from the visible text confuses speech input (WCAG 2.5.3). */
export function LegalLinks({ mt }: { mt?: string | number }) {
  const { t } = useTranslation();
  const pages = legalPages(useLegalInfo());
  if (pages.length === 0) {
    return null;
  }
  return (
    <nav aria-label={t('legal.nav')} style={{ marginTop: mt }}>
      <ul className={classes.links}>
        {pages.map((page) => (
          <li key={page}>
            <Anchor
              component={Link}
              to={`/${page}`}
              c="dimmed"
              size="xs"
              underline="hover"
              title={t(`legal.labels.${page}`)}
            >
              {t(`legal.short.${page}`)}
            </Anchor>
          </li>
        ))}
        {pages.includes('imprint') && (
          <li>
            <Anchor component={Link} to="/imprint" c="dimmed" size="xs" underline="hover">
              {t('legal.contact')}
            </Anchor>
          </li>
        )}
      </ul>
    </nav>
  );
}
