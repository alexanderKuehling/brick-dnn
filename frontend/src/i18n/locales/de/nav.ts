import type { nav as en } from '../en/nav';
import type { Messages } from '../types';

export const nav: Messages<typeof en> = {
  home: 'Brick DNN – Startseite',
  builder: 'Editor',
  data: 'Daten',
  myNetworks: 'Meine Netze',
  pricing: 'Preise',
  sourceCode: 'Quellcode auf GitHub',
  githubRepository: 'GitHub-Repository',
  coffee: 'Spendieren Sie mir einen Kaffee',
  legalTooltip: 'Impressum und Datenschutz',
  legalInfo: 'Rechtliches',
  logIn: 'Anmelden',
  signUp: 'Registrieren',
  pages: 'Seiten',
  userWaiting: '{{user}}, {{count}} warten auf Ihre Freigabe',
  dataPreparation: 'Datenaufbereitung',
  apiTokens: 'API-Token',
  waiting: '{{count}} warten',
  agentActivity: 'Agenten-Aktivität',
  account: 'Konto',
  logOut: 'Abmelden',
};
