import { render, screen, userEvent } from '@test-utils';
import { vi } from 'vitest';
import { Toolbar } from './Toolbar';

function renderToolbar(over: Partial<React.ComponentProps<typeof Toolbar>> = {}) {
  const props: React.ComponentProps<typeof Toolbar> = {
    name: 'MLP',
    onNameChange: vi.fn(),
    analysis: null,
    analyzing: false,
    analysisError: null,
    empty: false,
    examples: [],
    saving: false,
    isSaved: true,
    onLoadExample: vi.fn(),
    onNew: vi.fn(),
    onImport: vi.fn(),
    onExportJson: vi.fn(),
    onTidy: vi.fn(),
    onSave: vi.fn(),
    onShowCode: vi.fn(),
    onExportPython: vi.fn(),
    onTrain: vi.fn(),
    trainingRun: null,
    onShowSummary: vi.fn(),
    onOpenPalette: vi.fn(),
    onOpenInspector: vi.fn(),
    ...over,
  };
  render(<Toolbar {...props} />);
  return props;
}

describe('Toolbar', () => {
  it('exports the script with the labelled button and, on the narrowest phones, the icon', async () => {
    const props = renderToolbar();
    await userEvent.click(screen.getByRole('button', { name: 'Export .py' }));
    await userEvent.click(screen.getByRole('button', { name: 'Download the PyTorch script' }));
    expect(props.onExportPython).toHaveBeenCalledTimes(2);
  });

  it('scrolls the file menu when the examples are longer than the screen', async () => {
    const examples = Array.from({ length: 20 }, (_, i) => ({
      id: `example_${i}`,
      name: `Example ${i}`,
      description: 'A network to start from.',
      order: i,
      graph: { version: 1 as const, nodes: [], edges: [] },
    }));
    const props = renderToolbar({ examples });
    await userEvent.click(screen.getByRole('button', { name: 'File' }));
    const menu = await screen.findByRole('menu');
    // jsdom cannot compute dvh units, so read the style as written.
    expect(menu.getAttribute('style')).toContain('overflow-y: auto');
    expect(menu.getAttribute('style')).toContain('max-height: calc(100dvh - 5rem)');
    await userEvent.click(screen.getByRole('menuitem', { name: /Example 19/ }));
    expect(props.onLoadExample).toHaveBeenCalledWith(examples[19]);
  });

  it('names the live state also where only its dot is shown', () => {
    renderToolbar({ live: 'offline' });
    expect(screen.getByRole('status', { name: 'Offline – changes will sync' })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Save to account' })).not.toBeInTheDocument();
  });
});
