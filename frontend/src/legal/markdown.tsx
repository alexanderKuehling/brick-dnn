import { Fragment, type ReactNode } from 'react';
import { Link } from 'react-router-dom';
import { Anchor, Title } from '@mantine/core';
import classes from './Legal.module.css';

/**
 * A small, safe subset of Markdown for the legal pages the operator writes (imprint, privacy
 * policy, terms of use). Everything becomes React elements: raw HTML is shown as text, and
 * links only go to http(s), mailto: and tel: addresses or to paths of this site.
 *
 * Supported: headings (`#` to `######`), paragraphs, where a single line break stays a line
 * break (addresses in an imprint), bulleted and numbered lists (also nested), block quotes,
 * horizontal rules, **bold**, *italic*, `code`, [links](https://…), <https://…> and plain web
 * and e-mail addresses. Not supported (shown as text): tables, images, HTML.
 */

export type Inline =
  | { type: 'text'; text: string }
  | { type: 'strong' | 'em'; children: Inline[] }
  | { type: 'code'; text: string }
  | { type: 'link'; href: string; children: Inline[] }
  | { type: 'break' };

export type Block =
  | { type: 'heading'; level: 1 | 2 | 3 | 4 | 5 | 6; children: Inline[] }
  | { type: 'paragraph'; children: Inline[] }
  | { type: 'list'; ordered: boolean; start: number; items: Block[][] }
  | { type: 'quote'; children: Block[] }
  | { type: 'rule' };

// ---------------------------------------------------------------------------------------------
// Links
// ---------------------------------------------------------------------------------------------

/** `raw` if it is a safe link target, otherwise null (the link is shown as text). */
export function safeHref(raw: string): string | null {
  const href = raw.trim();
  // Control characters (and spaces) can hide a scheme from simple checks.
  // eslint-disable-next-line no-control-regex
  if (href === '' || /[\u0000- \u007f]/.test(href)) {
    return null;
  }
  if (/^(https?:\/\/|mailto:|tel:)/i.test(href)) {
    return href;
  }
  // A path of this site; not "//host" or "/\host", which browsers treat as another site.
  if (/^\/(?![/\\])/.test(href) || href.startsWith('#')) {
    return href;
  }
  return null;
}

// ---------------------------------------------------------------------------------------------
// Inline elements
// ---------------------------------------------------------------------------------------------

const ESCAPE = /\\([!-/:-@[-`{-~])/y;
const CODE = /(`+)([^`]|[^`][\s\S]*?[^`])\1(?!`)/y;
const AUTOLINK = /<((?:https?:\/\/|mailto:|tel:)[^\s<>]+)>/iy;
const AUTOLINK_EMAIL = /<([^\s<>@]+@[^\s<>@]+\.[^\s<>@]+)>/y;
const LINK =
  /\[((?:[^[\]\\]|\\.)*)\]\(\s*<?([^\s()<>]*(?:\([^\s()<>]*\)[^\s()<>]*)*)>?(?:\s+(?:"[^"]*"|'[^']*'))?\s*\)/y;
const STRONG = /(\*\*|__)(?=\S)([\s\S]*?\S)\1/y;
const EMPHASIS = /([*_])(?=\S)([\s\S]*?\S)\1/y;
// Trailing punctuation belongs to the sentence, not to the address.
const URL = /https?:\/\/[^\s<>]*[^\s<>.,:;"'!?)\]*_]/iy;
// Bounded, so that long runs of word characters cannot make the search slow.
const EMAIL = /[\w.+-]{1,64}@[\w-]{1,63}(?:\.[\w-]{1,63}){0,8}\.[a-z]{2,24}(?![\w-])/iy;

function isWordChar(char: string | undefined): boolean {
  return char !== undefined && /[\p{L}\p{N}_]/u.test(char);
}

function match(pattern: RegExp, text: string, position: number): RegExpExecArray | null {
  pattern.lastIndex = position;
  return pattern.exec(text);
}

/** Inline elements of `text`; `links: false` inside a link's text. */
export function parseInline(text: string, links = true): Inline[] {
  const result: Inline[] = [];
  let plain = '';
  const flush = () => {
    if (plain) {
      result.push({ type: 'text', text: plain });
      plain = '';
    }
  };
  const push = (node: Inline) => {
    flush();
    result.push(node);
  };
  const link = (href: string, children: Inline[], raw: string) => {
    const safe = links ? safeHref(href) : null;
    if (safe) {
      push({ type: 'link', href: safe, children });
    } else {
      plain += raw;
    }
  };

  let i = 0;
  while (i < text.length) {
    const char = text[i];
    const before = text[i - 1];
    let m: RegExpExecArray | null = null;

    if (char === '\n') {
      plain = plain.replace(/[ \t]+$/, '');
      push({ type: 'break' });
      i += 1;
      continue;
    }
    if (char === '\\') {
      if (text[i + 1] === '\n') {
        i += 1; // a backslash at the end of a line is a line break, as in CommonMark
        continue;
      }
      if ((m = match(ESCAPE, text, i))) {
        plain += m[1];
        i += m[0].length;
        continue;
      }
    }
    if (char === '`' && (m = match(CODE, text, i))) {
      push({ type: 'code', text: m[2].replace(/^ (.+) $/, '$1') });
      i += m[0].length;
      continue;
    }
    if (char === '<') {
      if ((m = match(AUTOLINK, text, i))) {
        link(m[1], [{ type: 'text', text: m[1] }], m[0]);
        i += m[0].length;
        continue;
      }
      if ((m = match(AUTOLINK_EMAIL, text, i))) {
        link(`mailto:${m[1]}`, [{ type: 'text', text: m[1] }], m[0]);
        i += m[0].length;
        continue;
      }
    }
    if (char === '[' && links && (m = match(LINK, text, i))) {
      const label = parseInline(m[1], false);
      const href = m[2].replace(/\\([()])/g, '$1');
      if (safeHref(href)) {
        push({ type: 'link', href: safeHref(href)!, children: label });
      } else {
        // An unsafe target: keep the text, drop the link.
        flush();
        result.push(...label);
      }
      i += m[0].length;
      continue;
    }
    if (char === '*' || char === '_') {
      // Underscores only at word boundaries: snake_case stays as it is.
      const boundary = char === '*' || !isWordChar(before);
      if (boundary && (m = match(STRONG, text, i))) {
        if (char === '*' || !isWordChar(text[i + m[0].length])) {
          push({ type: 'strong', children: parseInline(m[2], links) });
          i += m[0].length;
          continue;
        }
      }
      if (boundary && (m = match(EMPHASIS, text, i)) && m[1] === char) {
        if (char === '*' || !isWordChar(text[i + m[0].length])) {
          push({ type: 'em', children: parseInline(m[2], links) });
          i += m[0].length;
          continue;
        }
      }
    }
    if (links && (char === 'h' || char === 'H') && !isWordChar(before)) {
      if ((m = match(URL, text, i))) {
        link(m[0], [{ type: 'text', text: m[0] }], m[0]);
        i += m[0].length;
        continue;
      }
    }
    if (links && isWordChar(char) && !isWordChar(before) && before !== '.' && before !== '@') {
      if ((m = match(EMAIL, text, i))) {
        link(`mailto:${m[0]}`, [{ type: 'text', text: m[0] }], m[0]);
        i += m[0].length;
        continue;
      }
    }
    plain += char;
    i += 1;
  }
  flush();
  return result;
}

// ---------------------------------------------------------------------------------------------
// Blocks
// ---------------------------------------------------------------------------------------------

const HEADING = /^ {0,3}(#{1,6})(?:[ \t]+(.*?))?(?:[ \t]+#+)?[ \t]*$/;
const RULE = /^ {0,3}([-*_])(?:[ \t]*\1){2,}[ \t]*$/;
const QUOTE = /^ {0,3}> ?(.*)$/;
const BULLET = /^( {0,3})([-*+])([ \t]+)(.*)$/;
const ORDERED = /^( {0,3})(\d{1,9})([.)])([ \t]+)(.*)$/;

interface Marker {
  ordered: boolean;
  start: number;
  /** Columns before the item's text: lines indented this far belong to the item. */
  width: number;
  text: string;
}

function listMarker(line: string): Marker | null {
  let m = BULLET.exec(line);
  if (m && !RULE.test(line)) {
    return { ordered: false, start: 1, width: m[1].length + 1 + m[3].length, text: m[4] };
  }
  m = ORDERED.exec(line);
  if (m) {
    const width = m[1].length + m[2].length + 1 + m[4].length;
    return { ordered: true, start: Number(m[2]), width, text: m[5] };
  }
  return null;
}

const isBlank = (line: string) => line.trim() === '';
const indentOf = (line: string) => line.length - line.trimStart().length;

/** Lines that end a paragraph (besides a blank line). */
function startsBlock(line: string): boolean {
  return HEADING.test(line) || RULE.test(line) || QUOTE.test(line) || listMarker(line) !== null;
}

function parseList(lines: string[], start: number): [Block, number] {
  const first = listMarker(lines[start])!;
  const items: Block[][] = [];
  let i = start;
  for (;;) {
    const marker = listMarker(lines[i])!;
    const content = [marker.text];
    i += 1;
    while (i < lines.length) {
      const line = lines[i];
      if (isBlank(line)) {
        let next = i + 1;
        while (next < lines.length && isBlank(lines[next])) {
          next += 1;
        }
        if (next < lines.length && indentOf(lines[next]) >= marker.width) {
          content.push(...lines.slice(i, next).map(() => ''));
          i = next;
          continue;
        }
        break;
      }
      if (indentOf(line) >= marker.width) {
        content.push(line.slice(marker.width));
      } else if (startsBlock(line)) {
        break;
      } else {
        content.push(line.trim()); // a paragraph that goes on without indentation
      }
      i += 1;
    }
    items.push(parseBlocks(content));

    let next = i;
    while (next < lines.length && isBlank(lines[next])) {
      next += 1;
    }
    const sibling = next < lines.length ? listMarker(lines[next]) : null;
    if (sibling && sibling.ordered === first.ordered) {
      i = next;
      continue;
    }
    return [{ type: 'list', ordered: first.ordered, start: first.start, items }, i];
  }
}

function parseBlocks(lines: string[]): Block[] {
  const blocks: Block[] = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    let m: RegExpExecArray | null;
    if (isBlank(line)) {
      i += 1;
    } else if (RULE.test(line)) {
      blocks.push({ type: 'rule' });
      i += 1;
    } else if ((m = HEADING.exec(line))) {
      const level = m[1].length as 1 | 2 | 3 | 4 | 5 | 6;
      blocks.push({ type: 'heading', level, children: parseInline(m[2] ?? '') });
      i += 1;
    } else if (QUOTE.test(line)) {
      const inner: string[] = [];
      while (i < lines.length && !isBlank(lines[i])) {
        const quoted = QUOTE.exec(lines[i]);
        inner.push(quoted ? quoted[1] : lines[i]);
        i += 1;
      }
      blocks.push({ type: 'quote', children: parseBlocks(inner) });
    } else if (listMarker(line)) {
      const [list, next] = parseList(lines, i);
      blocks.push(list);
      i = next;
    } else {
      const paragraph: string[] = [line.trim()];
      i += 1;
      while (i < lines.length && !isBlank(lines[i]) && !startsBlock(lines[i])) {
        paragraph.push(lines[i].trim());
        i += 1;
      }
      blocks.push({ type: 'paragraph', children: parseInline(paragraph.join('\n')) });
    }
  }
  return blocks;
}

/** The blocks of a Markdown document. */
export function parseMarkdown(text: string): Block[] {
  const lines = text.replace(/\r\n?/g, '\n').replace(/\t/g, '    ').split('\n');
  return parseBlocks(lines);
}

// ---------------------------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------------------------

function renderInline(nodes: Inline[]): ReactNode {
  return nodes.map((node, index) => {
    switch (node.type) {
      case 'text':
        return <Fragment key={index}>{node.text}</Fragment>;
      case 'break':
        return <br key={index} />;
      case 'strong':
        return <strong key={index}>{renderInline(node.children)}</strong>;
      case 'em':
        return <em key={index}>{renderInline(node.children)}</em>;
      case 'code':
        return <code key={index}>{node.text}</code>;
      case 'link':
        if (node.href.startsWith('/')) {
          return (
            <Anchor key={index} component={Link} to={node.href}>
              {renderInline(node.children)}
            </Anchor>
          );
        }
        if (/^https?:/i.test(node.href)) {
          return (
            <Anchor key={index} href={node.href} target="_blank" rel="noopener noreferrer">
              {renderInline(node.children)}
            </Anchor>
          );
        }
        return (
          <Anchor key={index} href={node.href}>
            {renderInline(node.children)}
          </Anchor>
        );
    }
    return null;
  });
}

function renderBlocks(blocks: Block[], tight = false): ReactNode {
  return blocks.map((block, index) => {
    switch (block.type) {
      case 'heading':
        return (
          <Title key={index} order={block.level} className={classes.heading}>
            {renderInline(block.children)}
          </Title>
        );
      case 'paragraph':
        // Items of a list hold their text without a paragraph around it.
        return tight ? (
          <Fragment key={index}>{renderInline(block.children)}</Fragment>
        ) : (
          <p key={index}>{renderInline(block.children)}</p>
        );
      case 'list': {
        const items = block.items.map((item, itemIndex) => (
          <li key={itemIndex}>
            {renderBlocks(item, item.length === 1 && item[0].type === 'paragraph')}
          </li>
        ));
        return block.ordered ? (
          <ol key={index} start={block.start === 1 ? undefined : block.start}>
            {items}
          </ol>
        ) : (
          <ul key={index}>{items}</ul>
        );
      }
      case 'quote':
        return <blockquote key={index}>{renderBlocks(block.children)}</blockquote>;
      case 'rule':
        return <hr key={index} />;
    }
    return null;
  });
}

/** Markdown `text` as React elements (see the top of this file for what is supported). */
export function Markdown({ text }: { text: string }) {
  return <div className={classes.markdown}>{renderBlocks(parseMarkdown(text))}</div>;
}
