import { axeViolations, render, screen, userEvent } from '@test-utils';
import { OverlayChart, overlayStyle } from './OverlayChart';
import { ParallelCoordinates } from './ParallelCoordinates';

const SERIES = Array.from({ length: 10 }, (_, i) => ({
  key: `run${i}`,
  label: `Run ${i}`,
  x: i === 0 ? [1, 2, 3] : [1, 2],
  y: i === 0 ? [1, 0.5, 0.25] : [2 - i * 0.1, null],
}));

describe('OverlayChart', () => {
  it('draws a line per run with its own epochs, dashed after the eighth colour', () => {
    render(<OverlayChart title="Loss" series={SERIES} xLabel="Epoch" yLabel="Loss" width={500} />);
    const svg = screen.getByRole('img', { name: /^Loss\. Loss against Epoch, 10 runs\./ });
    expect(svg).toHaveAccessibleName(
      expect.stringContaining('Run 0: 1 at Epoch 1 to 0.25 at Epoch 3.')
    );
    const first = screen.getByTestId('overlay-run0').querySelector('path')!;
    expect(first.getAttribute('stroke')).toBe('var(--chart-series-1)');
    expect(first).not.toHaveAttribute('stroke-dasharray');
    const ninth = screen.getByTestId('overlay-run8').querySelector('path')!;
    expect(ninth.getAttribute('stroke')).toBe('var(--chart-series-1)');
    expect(ninth).toHaveAttribute('stroke-dasharray', '6 4');
    // A run with a single value gets a dot.
    expect(screen.getByTestId('overlay-run1').querySelectorAll('circle')).toHaveLength(1);
    expect(overlayStyle(9)).toEqual({ color: 'var(--chart-series-2)', dashed: true });
  });

  it('reads the values of every run at an x with the keyboard', async () => {
    render(
      <OverlayChart
        title="Loss"
        series={SERIES.slice(0, 2)}
        xLabel="Epoch"
        yLabel="Loss"
        width={500}
      />
    );
    const slider = screen.getByRole('slider', {
      name: 'Loss: use the arrow keys to read the values per Epoch',
    });
    expect(slider).toHaveAttribute('aria-valuetext', 'Epoch 3: Run 0 0.25, Run 1 no value');
    slider.focus();
    await userEvent.keyboard('{Home}');
    expect(slider).toHaveAttribute('aria-valuetext', 'Epoch 1: Run 0 1, Run 1 1.9');
    expect(await axeViolations()).toEqual([]);
  });

  it('shows the values as a table', async () => {
    render(
      <OverlayChart
        title="Loss"
        series={SERIES.slice(0, 2)}
        xLabel="Epoch"
        yLabel="Loss"
        width={400}
        table={() => ({ head: ['Epoch', 'Run 0'], rows: [[1, 1]] })}
      />
    );
    await userEvent.click(screen.getByRole('button', { name: 'Show Loss as table' }));
    expect(screen.getByRole('table', { name: /Loss/ })).toBeInTheDocument();
  });
});

describe('ParallelCoordinates', () => {
  it('draws a line per run over number and word axes, and walks them with the keys', async () => {
    render(
      <ParallelCoordinates
        title="Settings"
        width={600}
        items={[
          { key: 'a', label: 'A' },
          { key: 'b', label: 'B' },
        ]}
        axes={[
          { key: 'lr', label: 'Learning rate', values: [0.001, 0.01] },
          { key: 'opt', label: 'Optimizer', values: ['adam', 'sgd'] },
          { key: 'acc', label: 'Accuracy', values: [0.9, null] },
        ]}
      />
    );
    expect(
      screen.getByRole('img', {
        name: 'Settings: 2 runs on the axes Learning rate, Optimizer, Accuracy.',
      })
    ).toBeInTheDocument();
    expect(screen.getByTestId('parallel-a').getAttribute('points')!.split(' ')).toHaveLength(3);
    // B has no accuracy: its line ends at the optimizer.
    expect(screen.getByTestId('parallel-b').getAttribute('points')!.split(' ')).toHaveLength(2);
    const slider = screen.getByRole('slider');
    slider.focus();
    expect(slider).toHaveAttribute(
      'aria-valuetext',
      'A: Learning rate 0.001, Optimizer adam, Accuracy 0.9'
    );
    await userEvent.keyboard('{ArrowDown}');
    expect(slider).toHaveAttribute(
      'aria-valuetext',
      'B: Learning rate 0.01, Optimizer sgd, Accuracy –'
    );
    expect(screen.getByTestId('parallel-a')).toHaveAttribute('stroke-opacity', '0.25');
    expect(await axeViolations()).toEqual([]);
  });

  it('needs two axes', () => {
    render(
      <ParallelCoordinates
        title="Settings"
        width={400}
        items={[{ key: 'a', label: 'A' }]}
        axes={[]}
      />
    );
    expect(screen.getByText('Choose at least two axes.')).toBeInTheDocument();
  });
});
