import { render, screen, userEvent } from '@test-utils';
import { vi } from 'vitest';
import type { LayerCatalog } from '@/api/types';
import { CLASSICAL_CATALOG } from './classicalFixtures';
import { Palette } from './Palette';

const layer = (type: string, category: string) => ({
  type,
  label: type,
  category,
  description: `${type} layer`,
  params: [],
  min_inputs: 1,
  max_inputs: 1,
});

const catalog: LayerCatalog = {
  categories: ['Input', 'Basic', 'Sequence'],
  layers: [layer('Input', 'Input'), layer('Linear', 'Basic'), layer('LSTM', 'Sequence')],
};

describe('Palette', () => {
  it('groups layers by category and filters them', async () => {
    render(<Palette catalog={catalog} onAdd={() => {}} />);
    expect(screen.getByText('Sequence')).toBeInTheDocument();
    await userEvent.type(screen.getByLabelText('Search layers'), 'lstm');
    expect(screen.queryByText('Linear')).not.toBeInTheDocument();
    expect(screen.getByText('LSTM')).toBeInTheDocument();
  });

  it('has no family headings without classical bricks', () => {
    render(<Palette catalog={catalog} onAdd={() => {}} />);
    expect(screen.queryByText('Neural network · PyTorch')).not.toBeInTheDocument();
  });

  it('separates neural layers from the classical bricks', async () => {
    render(<Palette catalog={CLASSICAL_CATALOG} onAdd={() => {}} />);
    const headings = screen.getAllByText(/ · (PyTorch|scikit-learn)$/).map((h) => h.textContent);
    expect(headings).toEqual(['Neural network · PyTorch', 'Classical · scikit-learn']);
    expect(screen.getByText('Kernels')).toBeInTheDocument();
    expect(screen.getByText(/Combine them with Kernel sum and Kernel product/)).toBeInTheDocument();
    expect(screen.getByText('Classical models')).toBeInTheDocument();
    // Kernels have a diamond swatch like their handles.
    const rbf = screen.getByRole('button', { name: 'RBF kernel' });
    expect(rbf.querySelector('[data-kernel]')).not.toBeNull();
    const svc = screen.getByRole('button', { name: 'Support vector classification' });
    expect(svc.querySelector('[data-kernel]')).toBeNull();
    const boosting = screen.getByRole('button', { name: 'Gradient boosting classification' });
    expect(boosting.querySelector('[data-kernel]')).toBeNull();

    // The search finds gradient boosting by its description, too.
    await userEvent.type(screen.getByLabelText('Search layers'), 'decision trees');
    expect(
      screen.getByRole('button', { name: 'Gradient boosting classification' })
    ).toBeInTheDocument();
    expect(
      screen.queryByRole('button', { name: 'Support vector classification' })
    ).not.toBeInTheDocument();
    await userEvent.clear(screen.getByLabelText('Search layers'));

    // Searching keeps the heading of what is found.
    await userEvent.type(screen.getByLabelText('Search layers'), 'white');
    expect(screen.getByText('Classical · scikit-learn')).toBeInTheDocument();
    expect(screen.queryByText('Neural network · PyTorch')).not.toBeInTheDocument();
  });

  it('lists the ensembles after the classical models', async () => {
    render(<Palette catalog={CLASSICAL_CATALOG} onAdd={() => {}} />);
    const categories = screen
      .getAllByText(/^(Kernels|Classical models|Ensembles)$/)
      .map((h) => h.textContent);
    expect(categories).toEqual(['Kernels', 'Classical models', 'Ensembles']);
    expect(screen.getByText(/two or more models of the same task/)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Voting classifier' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Stacking regression' })).toBeInTheDocument();
    await userEvent.type(screen.getByLabelText('Search layers'), 'ensemble');
    expect(screen.getByRole('button', { name: 'Voting classifier' })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'RBF kernel' })).not.toBeInTheDocument();
  });

  it('adds a layer on click', async () => {
    const onAdd = vi.fn();
    render(<Palette catalog={catalog} onAdd={onAdd} />);
    await userEvent.click(screen.getByText('Linear'));
    expect(onAdd).toHaveBeenCalledWith(catalog.layers[1]);
  });
});

describe('Palette dragging', () => {
  const pointer = (
    type: string,
    target: EventTarget,
    x: number,
    y: number,
    pointerType = 'mouse'
  ) => {
    const event = new MouseEvent(type, { bubbles: true, clientX: x, clientY: y, button: 0 });
    Object.assign(event, { pointerId: 1, pointerType });
    target.dispatchEvent(event);
  };

  it('drags a layer out with the mouse without also adding it on click', () => {
    const onAdd = vi.fn();
    const onDrop = vi.fn();
    const onDragStart = vi.fn();
    render(<Palette catalog={catalog} onAdd={onAdd} onDrop={onDrop} onDragStart={onDragStart} />);
    const item = screen.getByText('Linear');
    pointer('pointerdown', item, 10, 10);
    pointer('pointermove', window, 200, 120);
    expect(onDragStart).toHaveBeenCalled();
    expect(document.body.textContent).toContain('Linear');
    pointer('pointerup', window, 300, 150);
    item.click(); // the browser's click that follows the release
    expect(onDrop).toHaveBeenCalledWith(catalog.layers[1], { x: 300, y: 150 });
    expect(onAdd).not.toHaveBeenCalled();
  });

  it('keeps scrolling when a touch moves vertically', () => {
    const onDrop = vi.fn();
    const onDragStart = vi.fn();
    render(
      <Palette catalog={catalog} onAdd={() => {}} onDrop={onDrop} onDragStart={onDragStart} />
    );
    const item = screen.getByText('LSTM');
    pointer('pointerdown', item, 10, 10, 'touch');
    pointer('pointermove', window, 12, 60, 'touch');
    pointer('pointerup', window, 12, 80, 'touch');
    expect(onDragStart).not.toHaveBeenCalled();
    expect(onDrop).not.toHaveBeenCalled();
  });

  it('pulls a layer out sideways on touch', () => {
    const onDrop = vi.fn();
    render(<Palette catalog={catalog} onAdd={() => {}} onDrop={onDrop} />);
    pointer('pointerdown', screen.getByText('LSTM'), 10, 10, 'touch');
    pointer('pointermove', window, 80, 20, 'touch');
    pointer('pointerup', window, 400, 300, 'touch');
    expect(onDrop).toHaveBeenCalledWith(catalog.layers[2], { x: 400, y: 300 });
  });
});
