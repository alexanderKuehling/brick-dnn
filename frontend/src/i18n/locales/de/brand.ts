import type { brand as en } from '../en/brand';
import type { Messages } from '../types';

export const brand: Messages<typeof en> = {
  madeWith: {
    region: 'Über Brick DNN',
    label: 'Erstellt mit Brick DNN',
    what: 'Brick DNN ist ein Editor für neuronale Netze im Browser: aus Schichten bauen, trainieren und lesbaren PyTorch-Code exportieren.',
    openCopy: 'Kopie im Editor öffnen',
    buildYourOwn: 'Eigenes Netz bauen',
  },
  badge: {
    title: 'Badge für ein README oder eine Kursseite',
    hint: 'Eine Schaltfläche, die dieses Netz öffnet. Fügen Sie sie in ein README, ein Notebook oder Ihr Kursmaterial ein.',
    alt: 'In Brick DNN öffnen',
    copyMarkdown: 'Markdown kopieren',
    copyHtml: 'HTML kopieren',
    copied: 'Kopiert',
  },
};
