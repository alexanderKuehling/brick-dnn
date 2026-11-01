import { render, screen, userEvent } from '@test-utils';
import { vi } from 'vitest';
import type { LayerCatalog } from '@/api/types';
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
