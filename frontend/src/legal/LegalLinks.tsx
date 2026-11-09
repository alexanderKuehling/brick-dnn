import { Link } from 'react-router-dom';
import { Anchor } from '@mantine/core';
import { LEGAL_LABELS, legalPages, useLegalInfo } from './useLegalInfo';
import classes from './Legal.module.css';

const SHORT_LABELS = { imprint: 'Imprint', privacy: 'Privacy', terms: 'Terms' } as const;

/** "Imprint · Privacy · Terms": the legal pages this server has (nothing if it has none). */
export function LegalLinks({ mt }: { mt?: string | number }) {
  const pages = legalPages(useLegalInfo());
  if (pages.length === 0) {
    return null;
  }
  return (
    <nav aria-label="Legal" style={{ marginTop: mt }}>
      <ul className={classes.links}>
        {pages.map((page) => (
          <li key={page}>
            <Anchor
              component={Link}
              to={`/${page}`}
              c="dimmed"
              size="xs"
              aria-label={LEGAL_LABELS[page]}
            >
              {SHORT_LABELS[page]}
            </Anchor>
          </li>
        ))}
      </ul>
    </nav>
  );
}
