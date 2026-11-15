/** Die deutschen Texte: dieselben Schlüssel wie `../en` (Sie-Form, übliche ML-Fachbegriffe). */
import type { en } from '../en';
import type { Messages } from '../types';
import { auth, legal } from './auth';
import { builder } from './builder';
import { common } from './common';
import { data } from './data';
import { nav } from './nav';
import { projects } from './projects';
import { training } from './training';
import { ui } from './ui';

export const de: Messages<typeof en> = {
  common,
  nav,
  auth,
  legal,
  builder,
  training,
  projects,
  ui,
  data,
};
