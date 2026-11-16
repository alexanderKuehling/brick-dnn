/** Parts of the pages' JSON-LD (schema.org) that several pages share. */
import type { Assets } from './html.ts';
import type { Crumb } from './layout.ts';
import {
  absolute,
  AUTHOR,
  CONTACT_EMAIL,
  CURRENCY,
  PLANS,
  REPOSITORY_URL,
  SITE_URL,
} from './site.ts';

export const ORGANIZATION_ID = `${SITE_URL}/#organization`;
export const WEBSITE_ID = `${SITE_URL}/#website`;
export const SOFTWARE_ID = `${SITE_URL}/#software`;

export function organization(assets: Assets): object {
  return {
    '@type': 'Organization',
    '@id': ORGANIZATION_ID,
    name: 'Brick DNN',
    url: `${SITE_URL}/`,
    logo: {
      '@type': 'ImageObject',
      url: absolute(assets.logo.src),
      width: assets.logo.width,
      height: assets.logo.height,
    },
    email: CONTACT_EMAIL,
    sameAs: [REPOSITORY_URL],
  };
}

export function author(): object {
  return { '@type': 'Person', name: AUTHOR.name, url: AUTHOR.url };
}

export function breadcrumbList(crumbs: Crumb[]): object {
  return {
    '@type': 'BreadcrumbList',
    itemListElement: crumbs.map((crumb, index) => ({
      '@type': 'ListItem',
      position: index + 1,
      name: crumb.name,
      item: absolute(crumb.path),
    })),
  };
}

/** brickdnn.com's plans as schema.org offers (bought on /pricing). */
export function offers(): object[] {
  return PLANS.map((plan) => ({
    '@type': 'Offer',
    name: plan.name,
    price: String(plan.price),
    priceCurrency: CURRENCY,
    url: absolute('/pricing'),
    priceSpecification: {
      '@type': 'UnitPriceSpecification',
      price: String(plan.price),
      priceCurrency: CURRENCY,
      billingDuration: 'P1M',
      unitText: 'month',
      valueAddedTaxIncluded: true,
    },
  }));
}
