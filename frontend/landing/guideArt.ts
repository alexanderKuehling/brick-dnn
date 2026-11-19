/**
 * The small drawings on the cards of the guides' overview (/guides/, /de/anleitungen/). A guide
 * gets the drawing of its id, else the first whose topic its id names (`activation-functions`
 * → the curves), else a general one: a new guide needs nothing here to get a proper card.
 *
 * They are inline SVG without any text, so they need no translation, and decorative (the
 * card's title says what the guide is about): hidden from assistive technology. The colours
 * come from landing.css (classes, no style attributes: the pages' Content-Security-Policy
 * allows no inline styles), so they follow light and dark.
 */

/** A layer as the editor draws it: a box with the colour of its category and two lines. */
function brick(x: number, y: number, colour: string): string {
  return `<rect class="box" x="${x}" y="${y}" width="72" height="40" rx="6"/><rect class="${colour}" x="${x + 9}" y="${y + 10}" width="4" height="20" rx="2"/><rect class="ink" x="${x + 21}" y="${y + 12}" width="34" height="5" rx="2.5"/><rect class="faint" x="${x + 21}" y="${y + 23}" width="22" height="5" rx="2.5"/>`;
}

/** Layers on the canvas, and the code they become. */
function noCode(): string {
  const code = [
    [36, 104, 30, 'accent'],
    [72, 104, 84, 'faint'],
    [48, 116, 22, 'accent'],
    [76, 116, 118, 'faint'],
    [48, 128, 64, 'faint'],
    [118, 128, 40, 'teal'],
  ] as const;
  return `<path class="line" d="M96 46h28M196 46h28"/>
${brick(24, 26, 'teal')}${brick(124, 26, 'violet')}${brick(224, 26, 'accent')}
<path class="line" d="M160 66v18"/><path class="line" d="M155 80l5 5 5-5"/>
<rect class="box" x="24" y="92" width="272" height="50" rx="6"/>
${code.map(([x, y, width, colour]) => `<rect class="${colour}" x="${x}" y="${y}" width="${width}" height="5" rx="2.5"/>`).join('')}`;
}

/** A decision tree next to a small multi-layer perceptron. */
function tabular(): string {
  const layers = [
    [200, [50, 80, 110]],
    [240, [35, 65, 95, 125]],
    [280, [65, 95]],
  ] as const;
  const links: string[] = [];
  for (let index = 0; index < layers.length - 1; index++) {
    const [x1, from] = layers[index];
    const [x2, to] = layers[index + 1];
    for (const y1 of from) {
      for (const y2 of to) {
        links.push(`M${x1} ${y1}L${x2} ${y2}`);
      }
    }
  }
  const neurons = layers
    .map(([x, ys], index) =>
      ys
        .map(
          (y) =>
            `<circle class="${index === layers.length - 1 ? 'accent' : 'box'}" cx="${x}" cy="${y}" r="7"/>`
        )
        .join('')
    )
    .join('');
  return `<path class="line" d="M80 40L48 78M80 40l32 38M48 84l-18 34M48 84l18 34M112 84L94 118M112 84l18 34"/>
<rect class="box" x="66" y="28" width="28" height="16" rx="4"/><rect class="box" x="34" y="72" width="28" height="16" rx="4"/><rect class="box" x="98" y="72" width="28" height="16" rx="4"/>
<circle class="accent" cx="30" cy="122" r="7"/><circle class="muted" cx="66" cy="122" r="7"/><circle class="muted" cx="94" cy="122" r="7"/><circle class="accent" cx="130" cy="122" r="7"/>
<path class="rule" d="M160 26v108"/>
<path class="thin" d="${links.join('')}"/>
${neurons}`;
}

/** An image of shapes and the mask a U-Net predicts for it: one colour per class. */
function segmentation(): string {
  const shapes = (dx: number, classes: [string, string, string]) =>
    `<circle class="${classes[0]}" cx="${60 + dx}" cy="62" r="17"/><path class="${classes[1]}" d="M${106 + dx} 44l17 30h-34z"/><rect class="${classes[2]}" x="${70 + dx}" y="90" width="44" height="26" rx="3"/>`;
  return `<rect class="box" x="28" y="28" width="104" height="104" rx="8"/>
${shapes(0, ['outline', 'outline', 'outline'])}
<path class="line" d="M144 80h30"/><path class="line" d="M169 75l5 5-5 5"/>
<rect class="panel" x="188" y="28" width="104" height="104" rx="8"/>
${shapes(160, ['accent', 'teal', 'violet'])}`;
}

/** Many samples of one class, a few of the other, and the threshold that decides. */
function imbalanced(): string {
  const rare = new Set(['2,1', '7,0', '9,3', '4,4', '11,2']);
  const dots: string[] = [];
  for (let row = 0; row < 5; row++) {
    for (let column = 0; column < 12; column++) {
      const cls = rare.has(`${column},${row}`) ? 'accent' : 'muted';
      dots.push(`<circle class="${cls}" cx="${50 + column * 20}" cy="${26 + row * 19}" r="5.5"/>`);
    }
  }
  return `${dots.join('')}
<path class="line" d="M50 138h220"/><path class="accent-line" d="M50 138h72"/><circle class="knob" cx="122" cy="138" r="7"/>`;
}

/** Data on a server, behind a shield with a lock. */
function gdpr(): string {
  const rack = [50, 72, 94]
    .map(
      (y) =>
        `<rect class="box" x="34" y="${y}" width="66" height="16" rx="4"/><circle class="teal" cx="45" cy="${y + 8}" r="2.5"/><rect class="faint" x="54" y="${y + 6}" width="34" height="4" rx="2"/>`
    )
    .join('');
  return `${rack}
<path class="rule" d="M106 80h14M200 80h14"/>
<path class="shield" d="M160 24l40 15v35c0 29-17 50-40 62-23-12-40-33-40-62V39z"/>
<path class="accent-line" d="M151 76v-9a9 9 0 0118 0v9"/><rect class="accent" x="145" y="76" width="30" height="24" rx="4"/><circle class="hole" cx="160" cy="88" r="3.5"/>
<rect class="box" x="220" y="50" width="66" height="60" rx="5"/><path class="thin" d="M220 70h66M220 90h66M242 50v60M264 50v60"/>`;
}

/** Two activation functions on their axes: ReLU and a sigmoid. */
function activation(): string {
  return `<path class="line" d="M40 112h240M160 24v112"/>
<path class="teal-line" d="M48 106c70 0 92-8 112-34s42-36 112-36"/>
<path class="accent-line" d="M48 112h112l70-80"/>`;
}

/** A table (a feature is a column, a data point a row) and the weights of a small network. */
function parameters(): string {
  const layers = [
    [200, [50, 80, 110]],
    [244, [62, 98]],
    [288, [80]],
  ] as const;
  const links: string[] = [];
  for (let index = 0; index < layers.length - 1; index++) {
    const [x1, from] = layers[index];
    const [x2, to] = layers[index + 1];
    for (const y1 of from) {
      for (const y2 of to) {
        links.push(`M${x1} ${y1}L${x2} ${y2}`);
      }
    }
  }
  return `<rect class="box" x="28" y="34" width="124" height="92" rx="5"/>
<rect class="shade" x="90" y="34" width="31" height="92"/><rect class="shade-teal" x="28" y="80" width="124" height="23"/>
<path class="thin" d="M28 57h124M28 80h124M28 103h124M59 34v92M90 34v92M121 34v92"/>
<path class="accent-thin" d="${links.join('')}"/>
${layers.map(([x, ys]) => ys.map((y) => `<circle class="box" cx="${x}" cy="${y}" r="7"/>`).join('')).join('')}`;
}

/** A guide the drawings do not know: a page of text with a small network on it. */
function general(): string {
  return `<rect class="box" x="88" y="20" width="144" height="120" rx="8"/>
<rect class="ink" x="106" y="38" width="72" height="6" rx="3"/>
<rect class="faint" x="106" y="56" width="108" height="5" rx="2.5"/><rect class="faint" x="106" y="68" width="96" height="5" rx="2.5"/><rect class="faint" x="106" y="80" width="64" height="5" rx="2.5"/>
<path class="line" d="M116 114h34M170 114h34"/>
<circle class="teal" cx="112" cy="114" r="7"/><circle class="accent" cx="160" cy="114" r="7"/><circle class="violet" cx="208" cy="114" r="7"/>`;
}

/** By the id of the guide (guides.<lang>.ts). */
const BY_ID: Record<string, () => string> = {
  'no-code': noCode,
  tabular,
  segmentation,
  imbalanced,
  gdpr,
};

/** By a word in the id, for guides written later. */
const BY_TOPIC: [RegExp, () => string][] = [
  [/activation/, activation],
  [/param|feature|data-?point/, parameters],
  [/segment|u-?net|mask/, segmentation],
  [/tabular|table|boosting/, tabular],
  [/imbalanc|rare|threshold/, imbalanced],
  [/gdpr|dsgvo|privacy/, gdpr],
];

function pick(id: string): () => string {
  return BY_ID[id] ?? BY_TOPIC.find(([pattern]) => pattern.test(id))?.[1] ?? general;
}

/** Whether a guide has a drawing of its topic (false: the general one). */
export function hasOwnGuideArt(id: string): boolean {
  return pick(id) !== general;
}

/** The drawing on a guide's card. */
export function guideArt(id: string): string {
  return `<svg class="art" viewBox="0 0 320 160" width="320" height="160" aria-hidden="true" focusable="false">${pick(id)()}</svg>`;
}
