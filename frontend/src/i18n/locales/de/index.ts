/** Die deutschen Texte: dieselben Schlüssel wie `../en` (Sie-Form, übliche ML-Fachbegriffe). */
import type { en } from '../en';
import type { Messages } from '../types';
import { auth, legal } from './auth';
import { builder } from './builder';
import { common } from './common';
import { nav } from './nav';
import { projects } from './projects';
import { training } from './training';

export const de: Messages<typeof en> = {
  common,
  nav,
  auth,
  legal,
  builder,
  training,
  projects,
};
