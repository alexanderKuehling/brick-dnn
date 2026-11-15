import axe from 'axe-core';

/** WCAG 2.0, 2.1 and 2.2, levels A and AA (what EN 301 549 and the BFSG refer to). */
const TAGS = ['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa', 'wcag22aa'];

/**
 * Rules jsdom cannot decide: it has no layout, so colours, sizes and what is visible on the
 * screen are unknown. The browser check (scripts/axe-smoke.mjs) covers them.
 */
const NEEDS_A_BROWSER = ['color-contrast', 'link-in-text-block', 'target-size'];
/** The test document has no <title>; the app's has one and DocumentTitle names each page. */
const NOT_IN_TESTS = ['document-title'];

/**
 * The serious and critical axe-core violations in `context` (default: the whole document,
 * portals of modals and menus included), one line each: "rule: target (message)".
 */
export async function axeViolations(context: Element | Document = document): Promise<string[]> {
  const results = await axe.run(context, {
    runOnly: { type: 'tag', values: TAGS },
    rules: Object.fromEntries(
      [...NEEDS_A_BROWSER, ...NOT_IN_TESTS].map((rule) => [rule, { enabled: false }])
    ),
    resultTypes: ['violations'],
  });
  return results.violations
    .filter((violation) => violation.impact === 'serious' || violation.impact === 'critical')
    .flatMap((violation) =>
      violation.nodes.map((node) => `${violation.id}: ${node.target.join(' ')} (${violation.help})`)
    );
}
