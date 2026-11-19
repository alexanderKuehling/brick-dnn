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

/** The copy of one example page (examples.<lang>.ts); the facts come from examples.json. */
export interface ExampleCopy {
  /** The id of the example in the app and in examples.json. */
  id: string;
  /** A short name for lists and breadcrumbs, e.g. “U-Net”. */
  name: string;
  /** <title>, aimed at what people search for. */
  title: string;
  description: string;
  h1: string;
  /** The task shown in the facts (the analysis does not know segmentation or text). */
  task: 'classification' | 'regression' | 'segmentation' | 'sequence';
  /** One sentence for the cards of the gallery and of related examples (HTML). */
  summary: string;
  /** The opening paragraph (HTML). */
  lead: string;
  /** What the network does, paragraph by paragraph (HTML). */
  what: string[];
  /** When to use it, as a list (HTML). */
  when: string[];
  /** How to train it in Brick DNN (HTML). */
  train: string;
  /** Ids of related examples. */
  related: string[];
}

export interface ExampleGallery {
  title: string;
  description: string;
  h1: string;
  /** HTML. */
  intro: string;
  groups: { title: string; intro: string; ids: string[] }[];
  /** HTML: how the pages are made. */
  note: string;
}

export interface GuideStep {
  title: string;
  /** HTML. */
  html: string;
}

export interface GuideSection {
  id: string;
  title: string;
  /** Paragraphs, lists or tables (HTML, written by hand). */
  html: string;
  /** Numbered steps (they become the HowTo steps of the JSON-LD). */
  steps?: GuideStep[];
  /** A screenshot of images/ and its caption. */
  image?: Shot & { caption: string };
  /** More HTML after the steps or the screenshot. */
  after?: string;
}

/** A guide (guides.<lang>.ts). The same `id` in both languages makes the pair. */
export interface Guide {
  id: string;
  slug: string;
  title: string;
  description: string;
  h1: string;
  /** One sentence for the list of guides (HTML). */
  summary: string;
  /** The opening paragraph (HTML). */
  lead: string;
  published: string;
  modified: string;
  /** Article, or HowTo for a guide of steps (with `steps` in its sections). */
  type: 'Article' | 'HowTo';
  /** ISO 8601 duration of a HowTo, e.g. PT20M. */
  totalTime?: string;
  sections: GuideSection[];
  examples: string[];
}

export interface GuideIndex {
  title: string;
  description: string;
  h1: string;
  /** HTML. */
  intro: string;
}

/**
 * A page about Brick DNN itself (info.<lang>.ts): for lecturers, who is behind it, security.
 * The same `id` in both languages makes the pair.
 */
export interface InfoPage {
  id: 'teaching' | 'about' | 'security' | 'dataProtection';
  title: string;
  description: string;
  /** A short line above the heading. */
  kicker: string;
  h1: string;
  /** The opening paragraph (HTML). */
  lead: string;
  /** The one button under the opening paragraph. */
  action?: { label: string; href: string };
  /** HTML written by hand: paragraphs in `.prose`, lists, `dl.glance`, snippets. */
  sections: { id: string; title: string; html: string }[];
}

/** The German pricing page (pricing.de.ts). */
export interface PricingPageContent {
  title: string;
  description: string;
  h1: string;
  /** HTML. */
  lead: string;
  caption: string;
  columns: [string, string, string, string, string, string];
  row: (plan: PlanFacts) => [string, string, string, string, string, string];
  buy: { title: string; html: string; button: string };
  sections: { id: string; title: string; html: string }[];
  faq: { title: string; items: Question[] };
}
