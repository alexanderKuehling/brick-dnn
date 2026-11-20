/**
 * What each screenshot shows. A shot opens `path` in the app (as `user`, with the `storage`
 * entries and the navigation `state` it names), and `take` prepares the page and returns the
 * part of it to keep, in CSS pixels. `display` is the width the landing page shows it at; the
 * script writes it in a phone width, that width and twice that (or in `widths`).
 *
 * Selectors use roles and the app's visible texts, taken from TEXTS in the language of the
 * screenshots (`--lang`); a text missing there is looked for in English.
 */

/**
 * The texts of the app the shots look for (src/i18n/locales, and backend/locale for the names
 * of the preparation's steps). The activity page shows what an assistant did in English.
 */
export const TEXTS = {
  en: {
    filters64: '64 filters',
    howDataFlows: 'How data flows',
    next: 'Next',
    code: 'Code',
    fillMissing: 'Fill missing values',
    goToColumn: 'Go to column',
    predictions: 'Predictions',
    generatedCode: 'Generated code',
    yourDatasets: 'Your datasets',
    readyToTrain: 'Ready to train',
    weights: 'Weights',
    howToUseWeights: 'How to use the weights',
    loss: 'Loss',
    valuesPerEpoch: 'Values per epoch',
    confusionMatrix: 'Confusion matrix',
    trials: 'Trials',
    metrics: 'Metrics',
    showWhatMatters: 'Show what matters',
    whyTheModelDecides: 'Why the model decides',
    comments: 'Comments',
  },
  de: {
    filters64: '64 Filter',
    howDataFlows: 'Wie die Daten fließen',
    next: 'Weiter',
    code: 'Code',
    fillMissing: 'Fehlende Werte auffüllen',
    goToColumn: 'Zur Spalte',
    predictions: 'Vorhersagen',
    yourDatasets: 'Ihre Datensätze',
    readyToTrain: 'Bereit zum Training',
    weights: 'Gewichte',
    howToUseWeights: 'So verwenden Sie die Gewichte',
    loss: 'Verlust',
    valuesPerEpoch: 'Werte pro Epoche',
    confusionMatrix: 'Konfusionsmatrix',
    trials: 'Versuche',
    metrics: 'Metriken',
    showWhatMatters: 'Zeigen, was zählt',
    whyTheModelDecides: 'Warum das Modell so entscheidet',
    comments: 'Kommentare',
  },
};

const t = (ctx, key) => TEXTS[ctx.lang]?.[key] ?? TEXTS.en[key];

/** The builder's working copy, so that the editor opens with this network. */
function draft(graph, name) {
  return { 'brickdnn:draft': JSON.stringify({ doc: graph, name, projectId: null }) };
}

/** The bounding box of an element, grown by `pad` pixels and kept inside the viewport. */
async function box(page, locator, pad = 0) {
  const b = await locator.boundingBox();
  const { width, height } = page.viewportSize();
  const x = Math.max(0, b.x - pad);
  const y = Math.max(0, b.y - pad);
  return {
    x,
    y,
    width: Math.min(width, b.x + b.width + pad) - x,
    height: Math.min(height, b.y + b.height + pad) - y,
  };
}

/** Scrolls the element that scrolls `locator`'s content so that it starts at the top. */
async function scrollIntoTop(locator, offset = 0) {
  await locator.evaluate((element, offset) => {
    let host = element.parentElement;
    while (
      host &&
      !(
        host.scrollHeight > host.clientHeight + 4 &&
        /auto|scroll/.test(getComputedStyle(host).overflowY)
      )
    ) {
      host = host.parentElement;
    }
    if (host) {
      host.scrollTop +=
        element.getBoundingClientRect().top - host.getBoundingClientRect().top - offset;
    }
  }, offset);
}

/** A saved network as the builder's working copy (it then edits it live with others). */
function projectDraft(project, role) {
  return {
    'brickdnn:draft': JSON.stringify({
      doc: project.graph,
      name: project.name,
      projectId: project.id,
      version: project.version ?? null,
      role,
      saved: true,
    }),
  };
}

/** The training panel at full width: results on the left, explanations on the right. */
const WIDE = { 'brickdnn:training-panel-wide': '1' };

/** Opens a finished run in the training panel (as the app's own links do). */
const openRun = (key) => (ctx) => ({ openTraining: true, run: String(ctx.runs[key].id) });

export const SHOTS = [
  {
    // The hero: a CNN with every shape, a Conv2d selected (its parameters and receptive field).
    id: 'editor',
    display: 1088,
    widths: [720, 1280, 2560],
    viewport: [1440, 860],
    path: () => '/app',
    storage: (ctx) => draft(ctx.examples.cnn_mnist.graph, ctx.examples.cnn_mnist.name),
    async take(page, ctx) {
      await page
        .locator('.react-flow__node')
        .filter({ hasText: t(ctx, 'filters64') })
        .first()
        .click();
      await page.mouse.move(5, 500);
      return { x: 0, y: 0, width: 1440, height: 860 };
    },
  },
  {
    // Teaching: "How data flows" steps a sample through the network.
    id: 'data-flow',
    display: 600,
    viewport: [1440, 900],
    path: () => '/app',
    storage: (ctx) => draft(ctx.examples.cnn_mnist.graph, ctx.examples.cnn_mnist.name),
    async take(page, ctx) {
      await page.getByRole('button', { name: t(ctx, 'howDataFlows') }).click();
      await page.waitForTimeout(800);
      for (let i = 0; i < 3; i++) {
        await page.getByRole('button', { name: t(ctx, 'next') }).click();
        await page.waitForTimeout(600);
      }
      await page.mouse.move(5, 500);
      return { x: 276, y: 462, width: 626, height: 422 };
    },
  },
  {
    // How it works, step 1: an ensemble of three classical models in the builder.
    id: 'build',
    display: 360,
    viewport: [1440, 860],
    path: () => '/app',
    storage: (ctx) =>
      draft(
        ctx.examples.voting_ensemble_classifier.graph,
        ctx.examples.voting_ensemble_classifier.name
      ),
    async take(page, ctx) {
      // The minimap would cover the corner of this small crop.
      await page.addStyleTag({ content: '.react-flow__minimap { display: none; }' });
      await page.mouse.move(5, 500);
      return { x: 270, y: 270, width: 810, height: 506 };
    },
  },
  {
    // Feature: a mistake marked before anything runs (a kernel larger than its input).
    id: 'checks',
    display: 528,
    viewport: [1440, 860],
    path: () => '/app',
    storage: (ctx) => {
      const graph = structuredClone(ctx.examples.cnn_mnist.graph);
      const conv = graph.nodes.find((node) => node.id === 'conv2');
      conv.params = { ...conv.params, kernel_size: [16, 16], padding: 'valid' };
      return draft(graph, ctx.examples.cnn_mnist.name);
    },
    async take(page, ctx) {
      const node = page
        .locator('.react-flow__node')
        .filter({ hasText: t(ctx, 'filters64') })
        .first();
      await node.click();
      await page.waitForTimeout(800);
      await page.mouse.move(5, 500);
      return { x: 240, y: 60, width: 864, height: 540 };
    },
  },
  {
    // Reproducible code: the PyTorch code of the CNN.
    id: 'code-export',
    display: 600,
    viewport: [1440, 960],
    path: () => '/app',
    storage: (ctx) => draft(ctx.examples.cnn_mnist.graph, ctx.examples.cnn_mnist.name),
    async take(page, ctx) {
      await page.getByRole('button', { name: t(ctx, 'code'), exact: true }).click();
      await page.waitForTimeout(1200);
      await page.mouse.move(700, 500);
      await page.mouse.wheel(0, 300);
      await page.waitForTimeout(500);
      await page.mouse.move(1300, 900);
      return box(page, page.locator('.mantine-Modal-content'));
    },
  },
  {
    // Tabular research data: the preparation's steps and the prepared table.
    id: 'data-prep',
    display: 600,
    viewport: [1440, 940],
    path: (ctx) => `/data/${ctx.prep}`,
    async take(page, ctx) {
      await page.waitForTimeout(1500);
      await scrollIntoTop(page.getByText(t(ctx, 'fillMissing')).first(), 10);
      // The table from the measurements on (after the date parts and categories).
      await page.getByPlaceholder(t(ctx, 'goToColumn')).click();
      await page.getByRole('option', { name: 'rainfall_mm' }).click();
      await page.evaluate(() => document.activeElement?.blur());
      await page.waitForTimeout(800);
      await page.mouse.move(5, 900);
      return { x: 260, y: 56, width: 1010, height: 720 };
    },
  },
  {
    // Segmentation: predictions of the U-Net on validation images next to the true masks.
    id: 'segmentation',
    display: 600,
    viewport: [1440, 1000],
    path: () => '/app',
    storage: () => WIDE,
    state: openRun('unet'),
    async take(page, ctx) {
      const heading = page.getByText(t(ctx, 'predictions'), { exact: true }).first();
      await heading.waitFor();
      await page.waitForTimeout(1500);
      await scrollIntoTop(heading, 70);
      await page.waitForTimeout(800);
      const top = (await heading.boundingBox()).y - 12;
      return { x: 0, y: top, width: 860, height: 500 };
    },
  },
  {
    // The AI assistant: what it did in the account, on the agent activity page.
    id: 'agent-activity',
    display: 600,
    viewport: [1440, 900],
    path: () => '/activity',
    async take(page, ctx) {
      await page.getByText(t(ctx, 'generatedCode')).first().waitFor();
      return { x: 236, y: 76, width: 968, height: 694 };
    },
  },
  {
    // How it works, step 2: the training panel with data, settings and the fit check.
    id: 'training-setup',
    display: 336,
    viewport: [1440, 1000],
    path: () => '/app',
    storage: (ctx) => draft(ctx.mlpGraph, 'MLP: high yield'),
    state: (ctx) => ({ openTraining: true, dataset: ctx.dataset }),
    async take(page, ctx) {
      const heading = page.getByText(t(ctx, 'yourDatasets'), { exact: true }).first();
      await heading.waitFor();
      await page.getByText(t(ctx, 'readyToTrain')).first().waitFor();
      await page.waitForTimeout(1000);
      const drawer = await page.locator('.mantine-Drawer-content').boundingBox();
      const top = (await heading.boundingBox()).y - 14;
      return { x: drawer.x, y: top, width: 560, height: 350 };
    },
  },
  {
    // How it works, step 3: the downloads of a finished run.
    id: 'export',
    display: 336,
    viewport: [1440, 1000],
    path: () => '/app',
    storage: () => WIDE,
    state: openRun('cnn'),
    async take(page, ctx) {
      const button = page.getByRole('button', { name: new RegExp(t(ctx, 'weights')) }).first();
      await button.waitFor();
      // With the code that loads the weights opened below the buttons.
      await page.getByText(t(ctx, 'howToUseWeights')).click();
      await page.waitForTimeout(600);
      await scrollIntoTop(button, 200);
      await page.waitForTimeout(800);
      // From the heading above the buttons on.
      const top = (await button.boundingBox()).y - 38;
      return { x: 0, y: top, width: 544, height: 340 };
    },
  },
  {
    // Feature: loss and accuracy per epoch of the CNN on MNIST.
    id: 'training-curves',
    display: 528,
    viewport: [1440, 1000],
    path: () => '/app',
    storage: () => WIDE,
    state: openRun('cnn'),
    async take(page, ctx) {
      const heading = page.getByText(t(ctx, 'loss'), { exact: true }).first();
      await heading.waitFor();
      await page.waitForTimeout(1500);
      await scrollIntoTop(heading, 160);
      await page.waitForTimeout(800);
      const end = (await page.getByText(t(ctx, 'valuesPerEpoch')).first().boundingBox()).y - 10;
      return { x: 0, y: end - 538, width: 860, height: 538 };
    },
  },
  {
    // Feature: the confusion matrix of the CNN on the validation images.
    id: 'evaluation',
    display: 528,
    viewport: [1440, 1000],
    path: () => '/app',
    storage: () => WIDE,
    state: openRun('cnn'),
    async take(page, ctx) {
      const heading = page.getByText(t(ctx, 'confusionMatrix'), { exact: true }).first();
      await heading.waitFor();
      await page.waitForTimeout(1500);
      await scrollIntoTop(heading, 70);
      await page.waitForTimeout(800);
      const top = (await heading.boundingBox()).y - 12;
      return { x: 0, y: top, width: 680, height: 425 };
    },
  },
  {
    // Feature: the trials of a hyperparameter search of gradient boosting.
    id: 'search-results',
    display: 528,
    viewport: [1440, 1000],
    path: () => '/app',
    storage: () => WIDE,
    state: openRun('boosting'),
    async take(page, ctx) {
      const heading = page.getByText(t(ctx, 'trials'), { exact: true }).first();
      await heading.waitFor();
      await page.waitForTimeout(1500);
      await scrollIntoTop(heading, 300);
      await page.waitForTimeout(800);
      const metrics = page.getByText(t(ctx, 'metrics'), { exact: true }).first();
      const end = (await metrics.boundingBox()).y - 10;
      return { x: 0, y: end - 538, width: 860, height: 538 };
    },
  },
  {
    // Feature: which features the gradient boosting model relies on.
    id: 'explain',
    display: 528,
    viewport: [1440, 1000],
    path: () => '/app',
    storage: () => WIDE,
    state: openRun('boosting'),
    async take(page, ctx) {
      await page.getByRole('button', { name: t(ctx, 'showWhatMatters') }).click();
      await page.waitForTimeout(15000);
      const panel = page.getByText(t(ctx, 'whyTheModelDecides')).first();
      const top = (await panel.boundingBox()).y - 22;
      return { x: 862, y: top, width: 576, height: 360 };
    },
  },
  {
    // Feature: working together, a second person in the network and a comment on a layer.
    id: 'collaboration',
    display: 528,
    viewport: [1440, 900],
    path: () => '/app',
    storage: (ctx) => projectDraft(ctx.shared, 'owner'),
    async take(page, ctx, open) {
      const other = await open({
        user: 'other',
        storage: () => projectDraft(ctx.shared, 'editor'),
      });
      await other.goto(new URL('/app', page.url()).href, { waitUntil: 'networkidle' });
      await other.waitForTimeout(2000);
      await other.locator('.react-flow__node').filter({ hasText: 'Conv2d' }).first().click();
      await page.waitForTimeout(2500);
      // The comments of the network next to the canvas.
      await page.getByRole('button', { name: new RegExp(t(ctx, 'comments')) }).click();
      await page.waitForTimeout(1500);
      await page.mouse.move(5, 800);
      return { x: 340, y: 56, width: 1100, height: 688 };
    },
  },
];

/** Adds the examples to the context (the shots need their graphs). */
export async function prepareContext(ctx, base) {
  const response = await fetch(`${base}/api/examples/`);
  ctx.examples = Object.fromEntries((await response.json()).map((item) => [item.id, item]));
  return ctx;
}
