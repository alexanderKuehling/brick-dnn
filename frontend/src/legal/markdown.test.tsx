/* eslint-disable no-script-url -- these tests make sure such links are never created */
import { render, screen } from '@test-utils';
import { MemoryRouter } from 'react-router-dom';
import { Markdown, parseInline, parseMarkdown, safeHref } from './markdown';

function renderMarkdown(text: string) {
  return render(
    <MemoryRouter>
      <Markdown text={text} />
    </MemoryRouter>
  );
}

describe('safeHref', () => {
  it('allows web, e-mail and phone addresses and paths of this site', () => {
    for (const href of [
      'https://example.com/a?b=c#d',
      'HTTP://example.com',
      'mailto:max@example.com',
      'tel:+49301234567',
      '/privacy',
      '#contact',
    ]) {
      expect(safeHref(href)).toBe(href);
    }
  });

  it('refuses everything else', () => {
    for (const href of [
      'javascript:alert(1)',
      'JaVaScRiPt:alert(1)',
      ' javascript:alert(1)',
      'java\tscript:alert(1)',
      'java\nscript:alert(1)',
      'data:text/html,<script>alert(1)</script>',
      'vbscript:msgbox(1)',
      '//evil.example.com',
      '/\\evil.example.com',
      'privacy',
      'ftp://example.com',
      '',
    ]) {
      expect(safeHref(href)).toBeNull();
    }
  });
});

describe('parseMarkdown', () => {
  it('reads headings, paragraphs with line breaks, lists, quotes and rules', () => {
    const blocks = parseMarkdown(
      [
        '# Impressum',
        '',
        'Max Mustermann',
        'Musterstraße 1  ',
        '12345 Musterstadt',
        '',
        '## Kontakt',
        '- E-Mail: max@example.com',
        '- Telefon: 030 123',
        '  - nested',
        '',
        '3. third',
        '4. fourth',
        '',
        '> quoted',
        '',
        '---',
      ].join('\n')
    );
    expect(blocks.map((block) => block.type)).toEqual([
      'heading',
      'paragraph',
      'heading',
      'list',
      'list',
      'quote',
      'rule',
    ]);
    expect(blocks[0]).toEqual({
      type: 'heading',
      level: 1,
      children: [{ type: 'text', text: 'Impressum' }],
    });
    expect(blocks[1]).toEqual({
      type: 'paragraph',
      children: [
        { type: 'text', text: 'Max Mustermann' },
        { type: 'break' },
        { type: 'text', text: 'Musterstraße 1' },
        { type: 'break' },
        { type: 'text', text: '12345 Musterstadt' },
      ],
    });
    const bullets = blocks[3];
    expect(bullets.type === 'list' && bullets.items).toHaveLength(2);
    expect(bullets.type === 'list' && bullets.items[1][1].type).toBe('list');
    const numbers = blocks[4];
    expect(numbers.type === 'list' && [numbers.ordered, numbers.start]).toEqual([true, 3]);
  });

  it('keeps a list item going over several lines', () => {
    const [list] = parseMarkdown('1. first line\n   second line\nlazy line\n2. next');
    expect(list.type === 'list' && list.items.length).toBe(2);
    expect(list.type === 'list' && list.items[0]).toEqual([
      {
        type: 'paragraph',
        children: [
          { type: 'text', text: 'first line' },
          { type: 'break' },
          { type: 'text', text: 'second line' },
          { type: 'break' },
          { type: 'text', text: 'lazy line' },
        ],
      },
    ]);
  });
});

describe('parseInline', () => {
  it('reads emphasis, code, links and plain addresses', () => {
    expect(parseInline('**bold** and *it* and `code`')).toEqual([
      { type: 'strong', children: [{ type: 'text', text: 'bold' }] },
      { type: 'text', text: ' and ' },
      { type: 'em', children: [{ type: 'text', text: 'it' }] },
      { type: 'text', text: ' and ' },
      { type: 'code', text: 'code' },
    ]);
    expect(parseInline('See [the terms](/terms).')).toEqual([
      { type: 'text', text: 'See ' },
      { type: 'link', href: '/terms', children: [{ type: 'text', text: 'the terms' }] },
      { type: 'text', text: '.' },
    ]);
    expect(parseInline('Mail max@example.com, or https://example.com/x.')).toEqual([
      { type: 'text', text: 'Mail ' },
      {
        type: 'link',
        href: 'mailto:max@example.com',
        children: [{ type: 'text', text: 'max@example.com' }],
      },
      { type: 'text', text: ', or ' },
      {
        type: 'link',
        href: 'https://example.com/x',
        children: [{ type: 'text', text: 'https://example.com/x' }],
      },
      { type: 'text', text: '.' },
    ]);
  });

  it('leaves snake_case, escaped characters and lone stars alone', () => {
    expect(parseInline('a_file_name and 2 * 3 * 4 and \\*no\\*')).toEqual([
      { type: 'text', text: 'a_file_name and 2 * 3 * 4 and *no*' },
    ]);
  });

  it('drops links to unsafe targets but keeps their text', () => {
    expect(parseInline('[click](javascript:alert(1))')).toEqual([{ type: 'text', text: 'click' }]);
    expect(parseInline('<javascript:alert(1)>')).toEqual([
      { type: 'text', text: '<javascript:alert(1)>' },
    ]);
  });
});

describe('Markdown', () => {
  it('renders the document', () => {
    renderMarkdown('# Privacy\n\nWe store **little**. Write to [us](mailto:max@example.com).');
    expect(screen.getByRole('heading', { level: 1, name: 'Privacy' })).toBeInTheDocument();
    expect(screen.getByText('little').tagName).toBe('STRONG');
    expect(screen.getByRole('link', { name: 'us' })).toHaveAttribute(
      'href',
      'mailto:max@example.com'
    );
  });

  it('opens other sites in a new tab and keeps paths of this site in the app', () => {
    renderMarkdown('[Hetzner](https://www.hetzner.com) and [terms](/terms)');
    const external = screen.getByRole('link', { name: 'Hetzner' });
    expect(external).toHaveAttribute('target', '_blank');
    expect(external).toHaveAttribute('rel', 'noopener noreferrer');
    const internal = screen.getByRole('link', { name: 'terms' });
    expect(internal).toHaveAttribute('href', '/terms');
    expect(internal).not.toHaveAttribute('target');
  });

  it('shows HTML as text and never creates scripts, handlers or unsafe links', () => {
    const { container } = renderMarkdown(
      [
        '<script>alert("xss")</script>',
        '',
        '<img src="x" onerror="alert(1)">',
        '',
        '[a](javascript:alert(1)) [b](data:text/html;base64,PHNjcmlwdD4=) [c]( javascript:alert(1))',
        '',
        '<a href="javascript:alert(1)">d</a> <iframe src="https://evil.example"></iframe>',
        '',
        '# <b onclick="alert(1)">title</b>',
      ].join('\n')
    );
    expect(container.querySelector('script, img, iframe, b')).toBeNull();
    for (const element of container.querySelectorAll('*')) {
      for (const attribute of element.getAttributeNames()) {
        expect(attribute.startsWith('on')).toBe(false);
      }
    }
    for (const link of container.querySelectorAll('a')) {
      expect(link.getAttribute('href')).not.toMatch(/^\s*(javascript|data|vbscript):/i);
    }
    expect(container).toHaveTextContent('<script>alert("xss")</script>');
    expect(container).toHaveTextContent('<img src="x" onerror="alert(1)">');
    // Only the plain web address in the iframe's text became a link.
    expect(screen.getAllByRole('link').map((link) => link.getAttribute('href'))).toEqual([
      'https://evil.example',
    ]);
  });
});
