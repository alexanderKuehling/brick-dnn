import { fireEvent, render, screen } from '@test-utils';
import { formatPercentTick } from './chartScale';
import { LineChart, SERIES_COLORS } from './LineChart';

const loss = {
  x: [1, 2, 3],
  series: [
    { id: 'train', label: 'Training', color: SERIES_COLORS[0], values: [0.9, 0.5, 0.3] },
    { id: 'val', label: 'Validation', color: SERIES_COLORS[1], values: [0.8, null, 0.35] },
  ],
};

describe('LineChart', () => {
  it('renders lines, a legend and a summary for screen readers', () => {
    render(<LineChart title="Loss" width={400} {...loss} />);
    const chart = screen.getByRole('img');
    expect(chart).toHaveAttribute(
      'aria-label',
      'Loss. Training: 0.9 at epoch 1, 0.3 at epoch 3. Validation: 0.8 at epoch 1, 0.35 at epoch 3.'
    );
    expect(chart.querySelectorAll('path')).toHaveLength(2);
    // The validation line is broken at the missing epoch.
    expect(chart.querySelectorAll('path')[1].getAttribute('d')?.match(/M/g)).toHaveLength(2);
    expect(screen.getByText('Training')).toBeInTheDocument();
    expect(screen.getByText('Validation')).toBeInTheDocument();
  });

  it('shows a single point as a dot without a legend for one series', () => {
    render(
      <LineChart
        title="Accuracy"
        width={300}
        x={[1]}
        series={[{ id: 'acc', label: 'Validation', color: SERIES_COLORS[1], values: [0.97] }]}
        yBounds={[0, 1]}
        formatYTick={formatPercentTick}
      />
    );
    const chart = screen.getByRole('img');
    expect(chart.querySelectorAll('circle')).toHaveLength(1);
    expect(screen.queryByRole('list')).not.toBeInTheDocument();
    expect(chart.textContent).toContain('%');
  });

  it('keeps many points readable', () => {
    const x = Array.from({ length: 500 }, (_, i) => i + 1);
    const values = x.map((epoch) => 1 / epoch);
    render(
      <LineChart
        title="Loss"
        width={480}
        x={x}
        series={[{ id: 'train', label: 'Training', color: SERIES_COLORS[0], values }]}
      />
    );
    const chart = screen.getByRole('img');
    // Only the end point gets a dot; the x axis shows a handful of round ticks.
    expect(chart.querySelectorAll('circle')).toHaveLength(1);
    const ticks = Array.from(chart.querySelectorAll('text')).map((t) => t.textContent);
    expect(ticks).toContain('500');
    expect(ticks.length).toBeLessThan(20);
  });

  it('reads values with the keyboard', () => {
    render(<LineChart title="Loss" width={400} {...loss} />);
    const slider = screen.getByRole('slider');
    fireEvent.focus(slider);
    expect(slider).toHaveAttribute('aria-valuenow', '3');
    fireEvent.keyDown(slider, { key: 'ArrowLeft' });
    expect(slider).toHaveAttribute('aria-valuenow', '2');
    expect(slider).toHaveAttribute('aria-valuetext', 'Epoch 2: Training 0.5, Validation no value');
    // The tooltip lists every series at that epoch.
    const tooltip = screen.getByText('Epoch 2').parentElement!;
    expect(tooltip).toHaveTextContent('0.5Training–Validation');
    fireEvent.keyDown(slider, { key: 'Home' });
    expect(slider).toHaveAttribute('aria-valuenow', '1');
  });

  it('says when there is nothing to show', () => {
    render(<LineChart title="Loss" width={400} x={[]} series={[]} />);
    expect(screen.getByText('No data yet')).toBeInTheDocument();
  });
});
