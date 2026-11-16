/**
 * Syntax highlighting of the example code at build time, with the highlight.js of the app's
 * code dialog (src/builder/CodeView.tsx): the pages get plain <span class="hljs-…"> markup
 * that landing.css colours, and load no script.
 */
import hljs from 'highlight.js/lib/core';
import python from 'highlight.js/lib/languages/python';

hljs.registerLanguage('python', python);

/** The code as HTML (highlight.js escapes it, so the result is safe to insert). */
export function highlightPython(code: string): string {
  return hljs.highlight(code, { language: 'python', ignoreIllegals: true }).value;
}
