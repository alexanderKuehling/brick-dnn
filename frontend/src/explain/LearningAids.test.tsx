/** The learning aids of the builder: the data-flow tour and the inspector's extras. */
import { render, screen, userEvent } from '@test-utils';
import { ReactFlowProvider } from '@xyflow/react';
import { vi } from 'vitest';
import type { LayerSpec } from '@/api/types';
import { DataFlowTour } from './DataFlowTour';
import { CNN, LINKS } from './fixtures';
import { LayerLearning } from './LayerLearning';

const SPECS = {
  Conv2d: {
    type: 'Conv2d',
    label: 'Conv2d',
    description: '2-D convolution over (channels, height, width).',
  },
  MaxPool2d: {
    type: 'MaxPool2d',
    label: 'MaxPool2d',
    description: 'Downsamples by taking the maximum.',
  },
} as unknown as Record<string, LayerSpec>;

describe('How data flows', () => {
  it('steps a sample through the bricks and centres each one', async () => {
    const focusNode = vi.fn();
    render(
      <ReactFlowProvider>
        <DataFlowTour analysis={CNN} specs={SPECS} links={LINKS} focusNode={focusNode} />
      </ReactFlowProvider>
    );
    await userEvent.click(screen.getByRole('button', { name: 'How data flows' }));
    const tour = screen.getByRole('region', { name: 'How data flows' });
    expect(tour).toHaveTextContent('Step 1 of 8');
    expect(focusNode).toHaveBeenLastCalledWith('in');

    await userEvent.click(screen.getByRole('button', { name: 'Next' }));
    expect(tour).toHaveTextContent('Step 2 of 8');
    expect(tour).toHaveTextContent('1 → 32 channels');
    expect(tour).toHaveTextContent('2-D convolution over (channels, height, width).');
    expect(tour).toHaveTextContent('3×3×1×32 + 32 = 320 parameters');
    expect(focusNode).toHaveBeenLastCalledWith('c1');

    await userEvent.click(screen.getByRole('button', { name: 'Next' }));
    await userEvent.click(screen.getByRole('button', { name: 'Next' }));
    expect(tour).toHaveTextContent('28×28 → 14×14 pixels');
    await userEvent.click(screen.getByRole('button', { name: 'Back' }));
    expect(tour).toHaveTextContent('Step 3 of 8');
    await userEvent.click(screen.getByRole('button', { name: 'Close' }));
    expect(screen.getByRole('button', { name: 'How data flows' })).toBeInTheDocument();
  });

  it('is not offered for invalid networks', () => {
    render(
      <ReactFlowProvider>
        <DataFlowTour
          analysis={{ ...CNN, valid: false }}
          specs={SPECS}
          links={LINKS}
          focusNode={vi.fn()}
        />
      </ReactFlowProvider>
    );
    expect(screen.queryByRole('button', { name: 'How data flows' })).not.toBeInTheDocument();
  });
});

describe('The inspector for learners', () => {
  it('shows the parameter formula and the receptive field', () => {
    render(<LayerLearning nodeId="c2" analysis={CNN} links={LINKS} />);
    expect(screen.getByLabelText('Parameter formula')).toHaveTextContent('3×3×32×64 + 64 = 18,496');
    expect(screen.getByText('8×8 pixels')).toBeInTheDocument();
    expect(screen.getByText(/neighbouring outputs are 2×2 input pixels apart/)).toBeInTheDocument();
  });

  it('shows nothing for bricks without either', () => {
    render(<LayerLearning nodeId="fl" analysis={CNN} links={LINKS} />);
    expect(screen.queryByText('Receptive field')).not.toBeInTheDocument();
    expect(screen.queryByText('How the parameters add up')).not.toBeInTheDocument();
  });
});
