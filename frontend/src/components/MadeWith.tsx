import type { ReactNode } from 'react';
import { Text } from '@mantine/core';
import { useTranslation } from '@/i18n';
import classes from './MadeWith.module.css';

/**
 * "Made with Brick DNN" on the pages people reach through a shared link, most of whom have
 * never seen the app: the logo, one sentence on what Brick DNN is and one action (`children`,
 * a button or link into the editor). `bar` is a strip across the page (under the canvas of a
 * shared network), `card` a quiet box at the end of a page's content.
 */
export function MadeWith({
  variant = 'card',
  children,
}: {
  variant?: 'bar' | 'card';
  children: ReactNode;
}) {
  const { t } = useTranslation();
  return (
    <section
      className={classes.root}
      data-variant={variant}
      aria-label={t('brand.madeWith.region')}
    >
      <p className={classes.mark}>
        <img src={`${import.meta.env.BASE_URL}favicon.svg`} width={22} height={22} alt="" />
        {t('brand.madeWith.label')}
      </p>
      <Text size="sm" c="dimmed" className={classes.what}>
        {t('brand.madeWith.what')}
      </Text>
      <div className={classes.action}>{children}</div>
    </section>
  );
}
