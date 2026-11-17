# Accessibility

Brick DNN aims at **WCAG 2.2, level AA** for the web app (`frontend/src`), and with it at the
European standard **EN 301 549**, which refers to WCAG 2.1 AA. This page says what was checked,
how to test it, how the editor works without a mouse, and what does not meet the aim yet. The
public accessibility statement of brickdnn.com is
[`deploy/sites/brickdnn.com/legal/accessibility.md`](../deploy/sites/brickdnn.com/legal/accessibility.md),
shown at `/accessibility`.

## The legal frame (Germany), in short

This is a summary for orientation, not legal advice.

- **Barrierefreiheitsstärkungsgesetz (BFSG)**, in force since 28 June 2025, implements the
  European Accessibility Act (Directive (EU) 2019/882). Among other things it covers
  *services in electronic commerce* offered to consumers, which a web app with paid plans is.
- **Micro-enterprises that provide services are exempt** (§ 3 (3) BFSG): fewer than 10
  employees and an annual turnover or balance sheet total of at most 2 million euros.
  brickdnn.com is run by a sole proprietor, so the exemption most likely applies; the
  statement says that this is assumed. Brick DNN follows the requirements voluntarily.
- **Technical standard:** the BFSG's regulation (BFSGV) is met by the harmonised standard
  EN 301 549, whose web chapter is WCAG 2.1 AA. WCAG 2.2 AA adds a few criteria (focus not
  obscured, dragging movements, target size, accessible authentication), which Brick DNN
  targets too.
- **Statement:** a service under the BFSG must say how it meets the requirements (Anlage 3
  BFSG), in an accessible form, and name the market surveillance authority (for all German
  states: the MLBF in Magdeburg). brickdnn.com publishes such a statement voluntarily, with
  status "teilweise barrierefrei" (partially accessible), known limitations, a feedback
  address and a date.
- **BITV 2.0** (Barrierefreie-Informationstechnik-Verordnung) binds only public bodies of the
  federal government (the states have their own laws). It does not apply to Brick DNN; its
  statement format was not used.

## How to test

```bash
cd frontend
npx vitest run src/a11y src/builder/BuilderKeyboard.test.tsx   # axe and keyboard tests
```

- **axe in the component tests** (`test-utils/axe.ts`, `axeViolations()`): runs axe-core with
  the WCAG 2.0/2.1/2.2 A and AA rules on the rendered document and returns the serious and
  critical violations. jsdom has no layout, so colour contrast, link distinction and target
  size are left to the browser check. Used in `src/a11y/pages.test.tsx` (login, sign-up with
  terms, forgotten password, accessibility statement, pricing, 404, charts, skip link, form
  errors; English and German), `src/builder/BuilderKeyboard.test.tsx` (editor with and without
  a selected layer) and `src/training/TrainingDrawer.test.tsx` (training setup and a run with
  its table).
- **Keyboard tests** for the editor (`BuilderKeyboard.test.tsx`): adding from the palette with
  Enter, arrow keys in the palette, Enter in the search, connecting and disconnecting in the
  properties panel, Delete on the canvas and in the panel with the focus kept, the list of
  layers, the names in German.
- **Browser check** against a running app (`scripts/axe-smoke.mjs`, not in CI): opens the main
  pages in Chromium in English and German, light and dark, logged out and (with
  `A11Y_USER`/`A11Y_PASSWORD`) logged in: login, sign-up, forgotten password, editor, training
  panel, a finished run, Data tab, my networks, account, API tokens, agent activity, pricing,
  the legal pages and 404. It needs Playwright with Chromium, which is not a dependency:

  ```bash
  npm install -g playwright && npx playwright install chromium   # once
  make backend & npm run dev                                     # the app
  NODE_PATH=$(npm root -g) A11Y_USER=alice A11Y_PASSWORD=… node scripts/axe-smoke.mjs
  ```

  It writes screenshots and `axe-report.json` into a temporary folder (`A11Y_OUT`) and exits
  with 1 on a serious or critical violation.
- **By hand**, before a release that changes the UI: Tab through the editor from the skip link
  to the properties panel; build Input → Linear → Linear with the keyboard alone; zoom the
  browser to 200 % and narrow the window to 320 px; switch the system to "reduce motion" and
  to the dark theme; read a page with a screen reader (NVDA, VoiceOver or Orca).

## What was checked and fixed (November 2026)

The first browser check found serious violations on every page (76 of 76 page/language/theme
combinations), almost all of them contrast. Fixed:

- **Contrast** (`src/theme.ts`): dimmed text (Mantine's gray-6, 3.3:1) is darker in the light
  theme and lighter in the dark one; coloured text, outline buttons and light badges use a
  darker shade per colour (e.g. green "Valid" badge 3.8:1 → 5.3:1); red buttons use red-8;
  filled badges pick black or white text (`autoContrast`); links in the dark theme 4.5:1.
- **Links in text are underlined** (Anchor default), not told apart by colour alone.
- **Focus** is visible everywhere with at least 3:1 in both themes (Mantine's ring had 1.6:1
  in the dark theme), including layers, connections and the zoom buttons of the canvas, which
  React Flow leaves without an outline.
- **Skip link** "Skip to content" as the first Tab stop; the main region is focusable; the page
  links in the header are a `nav` landmark with `aria-current`.
- **Page titles**: every route names the window ("Data · Brick DNN"), and a navigation inside
  the app is announced politely.
- **Label in name** (WCAG 2.5.3): the legal links, the account button, the budget buttons and
  the readiness tab had accessible names that did not contain their visible text.
- **Names**: the close button of the training panel had none; the guide's root had an
  `aria-label` on an element without a role.
- **The status badge** of the editor (valid / problems) opened the summary on click only; it
  is a button now, and no longer faded while the network is checked (its text fell below 4.5:1).
- **Notifications** are `role="status"` (polite) instead of `role="alert"`.
- **Form errors** of sign-up are shown at their field (`aria-invalid`, linked with
  `aria-describedby`), and the first one gets the focus.
- **Focus return**: the code preview and the share dialog are removed when they close, and the
  training panel is mounted open the first time, so Mantine did not return the focus; it now
  goes back to the button that opened them.
- **Reduced motion**: Mantine's transitions and the app's animations are off with
  `prefers-reduced-motion: reduce`, and the canvas zooms without animation.
- **Line charts** (training curves): the x axis name and "no value" are translated.
- **Target size** of the footer's legal links (24 px).
- **The runs page** ([experiments.md](experiments.md)): sortable column headers are buttons
  with `aria-sort` on the header cell; the overlay of curves, the scatter plot and the
  parallel coordinates each have a text summary, a table view and one Tab stop whose arrow
  keys read the values (of every run at an epoch, or of one run); runs keep their colour in
  every chart and are told apart by dashes after the eighth colour, not by colour alone in
  the legend and the table. Checked with axe in English and German (`RunsPage.test.tsx`,
  `SharedComparisonPage.test.tsx`, `overlay.test.tsx`).

After the fixes the browser check reports no serious or critical violation (76 checks). The
pages reflow at 320 px and at 200 % zoom without horizontal scrolling (the editor's canvas is
a two-dimensional area, which WCAG 1.4.10 exempts). Modals and drawers (Mantine) trap the focus
and close with Escape.

## The editor without a mouse

React Flow (`@xyflow/react`) already makes layers and connections focusable (`nodesFocusable`,
`edgesFocusable`), selectable with Enter or Space, movable with the arrow keys (Shift: faster;
it announces the new position) and deletable with Delete. `src/builder/a11y.tsx` adds:

- names for layers (`"hidden, Linear"`, `"…, has problems"`) and connections
  (`"Connection from Input to hidden"`), and React Flow's descriptions, control and minimap
  labels and its move announcement in the app's language (`ariaLabelConfig`);
- a polite live region that says what an action changed: added (and after which layer),
  connected, disconnected, deleted;
- the focus stays on the canvas after a delete (on a neighbouring layer) and moves to a layer
  chosen in a list.

And in the UI:

- **Palette:** one Tab stop; arrow keys, Home and End move between the layers; Enter or Space
  adds the layer after the selected one (connected if it fits); Enter in the search adds the
  first match, the down arrow goes to the list.
- **Properties panel, "Connections":** the inputs and outputs of the selected layer, each with
  a button to remove the connection, and the lists "Add input from …" and "Add output to …",
  which offer only layers that fit (right kind, no cycle).
- **Model summary, "Layers":** every layer with where its inputs come from, the canvas in
  words; choosing one selects it and moves the focus to it.
- **Delete** removes the selected layer or connection; Escape clears the selection.

## Known limitations

- The canvas is visual: positions carry no meaning, but the drawing (lines, minimap) is only
  available to screen readers through the names of layers and connections and the two lists.
- Box selection and moving several layers at once need a pointer. Moving a layer with the
  arrow keys works for the selected layer.
- Charts of the Data tab use categorical colours of which some are below 3:1 on the
  background; every chart has a legend and a table view.
- Generated code, logs and wide tables scroll horizontally inside their box.
- Some texts are still English in the German app ([translations.md](translations.md#known-gaps));
  the explanation of a layer's parameter count (`src/explain/learning.ts`) is one of them,
  and screen readers then read it with German pronunciation.
- The checkout of Paddle and other sites linked from the app are outside the app.
- No test with real users of assistive technology and no external audit yet.
