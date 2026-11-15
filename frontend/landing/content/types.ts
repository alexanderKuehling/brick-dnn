/**
 * The text of a landing page. Strings named `html` (and the `text` of items) may contain
 * inline markup (<code>, <a>, <strong>); they are written by hand and inserted as they are.
 * Everything else is plain text and is escaped.
 */
import type { Lang, PlanFacts } from '../site.ts';

/** A screenshot of the app (images/<id>-<width>.webp, made by screenshots/shoot.mjs). */
export interface Shot {
  id: string;
  alt: string;
}

export interface Item {
  title: string;
  html: string;
  image?: Shot;
}

export interface Question {
  question: string;
  /** The answer; the FAQPage JSON-LD gets it without markup. */
  html: string;
}

export interface LandingContent {
  lang: Lang;
  /** Open Graph locale, e.g. en_US. */
  locale: string;
  title: string;
  description: string;
  /** Short name of the language in its own language, for the switch ("English"). */
  languageName: string;
  skipLink: string;
  nav: {
    label: string;
    useCases: string;
    features: string;
    pricing: string;
    faq: string;
    editor: string;
  };
  hero: {
    title: string;
    lead: string;
    primary: string;
    secondary: string;
    /** Short facts under the buttons. */
    facts: string[];
    image: Shot;
    caption: string;
  };
  glance: { title: string; items: { term: string; html: string }[] };
  useCases: { title: string; intro: string; items: Item[] };
  steps: { title: string; items: Item[] };
  features: { title: string; intro: string; items: Item[] };
  privacy: { title: string; intro: string; items: Item[]; more: string };
  teaching: { title: string; items: string[]; discount: string };
  pricing: {
    title: string;
    intro: string;
    caption: string;
    columns: [string, string, string, string, string, string];
    row: (plan: PlanFacts) => [string, string, string, string, string, string];
    notes: string[];
    more: string;
  };
  faq: { title: string; items: Question[] };
  footer: {
    tagline: string;
    /** The name of the legal links' navigation. */
    legalLabel: string;
    legal: {
      imprint: string;
      privacy: string;
      terms: string;
      refund: string;
      accessibility: string;
    };
    /** A line under the legal links if they are in another language (empty: none). */
    legalNote: string;
    source: string;
    contact: string;
    language: string;
  };
  /** The application's description in the JSON-LD. */
  appDescription: string;
  /** Features in the JSON-LD (plain text). */
  featureList: string[];
  ogImageAlt: string;
}
