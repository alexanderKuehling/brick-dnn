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
