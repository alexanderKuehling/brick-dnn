/** Parts of the pages' JSON-LD (schema.org) that several pages share. */
import type { Assets } from './html.ts';
import type { Crumb } from './layout.ts';
import {
  absolute,
  AUTHOR,
  CONTACT_EMAIL,
  CURRENCY,
  EDITOR_PATH,
  PLANS,
  REPOSITORY_URL,
  SITE_URL,
} from './site.ts';

export const ORGANIZATION_ID = `${SITE_URL}/#organization`;
export const WEBSITE_ID = `${SITE_URL}/#website`;
export const SOFTWARE_ID = `${SITE_URL}/#software`;

/** Where brickdnn.com is aimed at: Europe (the server is in Switzerland, see site.ts). */
export const AREA_SERVED = { '@type': 'Continent', name: 'Europe' };
/** The languages of the app and of the site. */
export const LANGUAGES = ['en', 'de'];

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
    areaServed: AREA_SERVED,
    knowsAbout: ['Deep learning', 'Neural networks', 'PyTorch', 'Machine learning'],
  };
}

/**
 * The application (the editor at /app), as the pages that describe it share it: the landing
 * pages add their description, screenshots and features.
 */
export function softwareApplication(extra: object = {}): object {
  return {
    '@type': 'SoftwareApplication',
    '@id': SOFTWARE_ID,
    name: 'Brick DNN',
    url: absolute(EDITOR_PATH),
    applicationCategory: 'DeveloperApplication',
    applicationSubCategory: 'Deep learning',
    operatingSystem: 'Web browser',
    inLanguage: LANGUAGES,
    areaServed: AREA_SERVED,
    isAccessibleForFree: true,
    ...extra,
    offers: offers(),
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
