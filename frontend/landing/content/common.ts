/**
 * Texts of the content pages that are not tied to one page: the navigation, and the labels of
 * the example pages, the guides and the pricing page. The pages' own copy is in
 * examples.<lang>.ts, guides.<lang>.ts and pricing.de.ts.
 */
import type { DiagramLabels } from '../examples/diagram.ts';
import type { Lang } from '../site.ts';

export interface CommonTexts {
  nav: {
    label: string;
    siteLabel: string;
    breadcrumbs: string;
    home: string;
    examples: string;
    guides: string;
    pricing: string;
    editor: string;
  };
  skipLink: string;
  /** "Updated 16 November 2026". */
  date: (iso: string) => string;
  by: string;
  published: string;
  updated: string;
  example: {
    openEditor: string;
    toCode: string;
    facts: {
      framework: string;
      task: string;
      params: string;
      model: string;
      layers: string;
      input: string;
    };
    frameworks: Record<'pytorch' | 'scikit-learn' | 'pytorch+scikit-learn', string>;
    tasks: Record<'classification' | 'regression' | 'segmentation' | 'sequence', string>;
    whatTitle: string;
    whenTitle: string;
    networkTitle: string;
    trainTitle: string;
    codeTitle: string;
    relatedTitle: string;
    guidesTitle: string;
    diagramCaption: string;
    diagramScroll: string;
    diagramTitle: (name: string) => string;
    tableCaption: (name: string) => string;
    columns: [string, string, string, string, string];
    total: string;
    /** The parameter total of a classical model, which has no trainable parameters. */
    noParams: string;
    defaults: string;
    codeIntro: (framework: ExampleFramework, filename: string, csvLoop: boolean) => string;
    codeLabel: string;
    diagram: DiagramLabels;
  };
  guide: {
    relatedExamples: string;
    moreGuides: string;
    contents: string;
    tryIt: string;
  };
}

export type ExampleFramework = 'pytorch' | 'scikit-learn' | 'pytorch+scikit-learn';

const MONTHS_DE = [
  'Januar',
  'Februar',
  'März',
  'April',
  'Mai',
  'Juni',
  'Juli',
  'August',
  'September',
  'Oktober',
  'November',
  'Dezember',
];
const MONTHS_EN = [
  'January',
  'February',
  'March',
  'April',
  'May',
  'June',
  'July',
  'August',
  'September',
  'October',
  'November',
  'December',
];

function parts(iso: string): [number, number, number] {
  const [year, month, day] = iso.split('-').map(Number);
  return [year, month, day];
}

const count = (value: number, lang: Lang) =>
  value.toLocaleString(lang === 'de' ? 'de-DE' : 'en-US');

export const COMMON: Record<Lang, CommonTexts> = {
  en: {
    nav: {
      label: 'Site',
      siteLabel: 'Brick DNN',
      breadcrumbs: 'Breadcrumbs',
      home: 'Brick DNN',
      examples: 'Examples',
      guides: 'Guides',
      pricing: 'Pricing',
      editor: 'Editor',
    },
    skipLink: 'Skip to content',
    date: (iso) => {
      const [year, month, day] = parts(iso);
      return `${day} ${MONTHS_EN[month - 1]} ${year}`;
    },
    by: 'By',
    published: 'Published',
    updated: 'Updated',
    example: {
      openEditor: 'Open in the editor',
      toCode: 'See the code',
      facts: {
        framework: 'Code',
        task: 'Task',
        params: 'Trainable parameters',
        model: 'Model',
        layers: 'Bricks',
        input: 'Input',
      },
      frameworks: {
        pytorch: 'PyTorch',
        'scikit-learn': 'scikit-learn',
        'pytorch+scikit-learn': 'PyTorch + scikit-learn',
      },
      tasks: {
        classification: 'Classification',
        regression: 'Regression',
        segmentation: 'Segmentation',
        sequence: 'Text classification',
      },
      whatTitle: 'What the network does',
      whenTitle: 'When to use it',
      networkTitle: 'Layers, shapes and parameters',
      trainTitle: 'Training it in Brick DNN',
      codeTitle: 'The generated code',
      relatedTitle: 'Related examples',
      guidesTitle: 'Guides',
      diagramCaption:
        'The network as it is laid out in the editor. Each box shows the output shape (without the batch dimension) and the number of parameters.',
      diagramScroll: 'The diagram is wider than the screen: scroll it sideways.',
      diagramTitle: (name) => `Diagram of the example ${name}`,
      tableCaption: (name) =>
        `The bricks of ${name}: settings, output shape and parameters, computed by Brick DNN's shape inference`,
      columns: ['Brick', 'Type', 'Settings', 'Output', 'Parameters'],
      total: 'Total',
      noParams: 'none: fitted by scikit-learn',
      defaults: 'defaults',
      codeIntro: (framework, filename, csvLoop) =>
        framework === 'scikit-learn'
          ? `This is the scikit-learn script that the editor’s code dialog generates from the graph above (<code>${filename}</code>). It runs on its own: <code>python ${filename}</code> fits the model to random data, <code>--csv data.csv</code> fits it to your table and saves it, <code>--predict</code> predicts new rows.`
          : framework === 'pytorch'
            ? `This is the PyTorch script that the editor’s code dialog generates from the graph above (<code>${filename}</code>): an <code>nn.Module</code> with one attribute per layer and a <code>forward</code> method, plus a check that runs a random batch through it${csvLoop ? ', and, because the network takes one flat vector and returns one, a training loop for CSV files (<code>--csv data.csv</code>)' : ''}. It needs nothing but PyTorch.`
            : `This is the script that the editor’s code dialog generates from the graph above (<code>${filename}</code>): the neural layers as a PyTorch module, the classical model as a scikit-learn pipeline, and the code that trains both phases on a CSV file.`,
      codeLabel: 'Generated Python code',
      diagram: {
        kernel: 'kernel',
        model: 'model',
        classes: 'classes',
        params: (value) => `${count(value, 'en')} ${value === 1 ? 'parameter' : 'parameters'}`,
        hyperparameters: (value) =>
          `${count(value, 'en')} ${value === 1 ? 'hyperparameter' : 'hyperparameters'}`,
      },
    },
    guide: {
      relatedExamples: 'Examples to start from',
      moreGuides: 'More guides',
      contents: 'Contents',
      tryIt: 'Open the editor',
    },
  },
  de: {
    nav: {
      label: 'Website',
      siteLabel: 'Brick DNN',
      breadcrumbs: 'Brotkrumen',
      home: 'Brick DNN',
      examples: 'Beispiele',
      guides: 'Anleitungen',
      pricing: 'Preise',
      editor: 'Editor',
    },
    skipLink: 'Zum Inhalt springen',
    date: (iso) => {
      const [year, month, day] = parts(iso);
      return `${day}. ${MONTHS_DE[month - 1]} ${year}`;
    },
    by: 'Von',
    published: 'Veröffentlicht',
    updated: 'Aktualisiert',
    example: {
      openEditor: 'Im Editor öffnen',
      toCode: 'Zum Code',
      facts: {
        framework: 'Code',
        task: 'Aufgabe',
        params: 'Trainierbare Parameter',
        model: 'Modell',
        layers: 'Bausteine',
        input: 'Eingabe',
      },
      frameworks: {
        pytorch: 'PyTorch',
        'scikit-learn': 'scikit-learn',
        'pytorch+scikit-learn': 'PyTorch + scikit-learn',
      },
      tasks: {
        classification: 'Klassifikation',
        regression: 'Regression',
        segmentation: 'Segmentierung',
        sequence: 'Textklassifikation',
      },
      whatTitle: 'Was das Netz macht',
      whenTitle: 'Wann es passt',
      networkTitle: 'Schichten, Formen und Parameter',
      trainTitle: 'Training in Brick DNN',
      codeTitle: 'Der erzeugte Code',
      relatedTitle: 'Verwandte Beispiele',
      guidesTitle: 'Anleitungen',
      diagramCaption:
        'Das Netz so, wie es im Editor angeordnet ist. Jeder Kasten zeigt die Ausgabeform (ohne Batch-Dimension) und die Zahl der Parameter.',
      diagramScroll:
        'Das Diagramm ist breiter als der Bildschirm: Sie können es seitlich verschieben.',
      diagramTitle: (name) => `Diagramm des Beispiels ${name}`,
      tableCaption: (name) =>
        `Die Bausteine von ${name}: Einstellungen, Ausgabeform und Parameter, berechnet von der Formprüfung von Brick DNN`,
      columns: ['Baustein', 'Typ', 'Einstellungen', 'Ausgabe', 'Parameter'],
      total: 'Gesamt',
      noParams: 'keine: scikit-learn passt das Modell an',
      defaults: 'Standardwerte',
      codeIntro: (framework, filename, csvLoop) =>
        framework === 'scikit-learn'
          ? `Das ist das scikit-learn-Skript, das der Code-Dialog des Editors aus dem Graphen oben erzeugt (<code>${filename}</code>). Es läuft für sich: <code>python ${filename}</code> passt das Modell an Zufallsdaten an, <code>--csv daten.csv</code> an Ihre Tabelle und speichert es, <code>--predict</code> sagt neue Zeilen vorher.`
          : framework === 'pytorch'
            ? `Das ist das PyTorch-Skript, das der Code-Dialog des Editors aus dem Graphen oben erzeugt (<code>${filename}</code>): ein <code>nn.Module</code> mit einem Attribut je Schicht und einer <code>forward</code>-Methode, dazu ein Test, der einen zufälligen Batch durch das Netz schickt${csvLoop ? ', und, weil das Netz einen flachen Vektor nimmt und einen ausgibt, eine Trainingsschleife für CSV-Dateien (<code>--csv daten.csv</code>)' : ''}. Es braucht nur PyTorch.`
            : `Das ist das Skript, das der Code-Dialog des Editors aus dem Graphen oben erzeugt (<code>${filename}</code>): die neuronalen Schichten als PyTorch-Modul, das klassische Modell als scikit-learn-Pipeline und der Code, der beide Phasen auf einer CSV-Datei trainiert.`,
      codeLabel: 'Erzeugter Python-Code',
      diagram: {
        kernel: 'Kernel',
        model: 'Modell',
        classes: 'Klassen',
        params: (value) => `${count(value, 'de')} Parameter`,
        hyperparameters: (value) => `${count(value, 'de')} Hyperparameter`,
      },
    },
    guide: {
      relatedExamples: 'Beispiele zum Ausprobieren',
      moreGuides: 'Weitere Anleitungen',
      contents: 'Inhalt',
      tryIt: 'Editor öffnen',
    },
  },
};
