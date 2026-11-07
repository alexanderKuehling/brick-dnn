import { fireEvent, render, screen, within } from '@test-utils';
import { BarChart } from './BarChart';
import { BoxPlot } from './BoxPlot';
import { ChartTooltip, tooltipPosition } from './ChartFrame';
import { seriesColor } from './colors';
import { Heatmap, strongestPairs } from './Heatmap';
import { binLabels, Histogram } from './Histogram';
import { MissingMatrix } from './MissingMatrix';
import { CANVAS_THRESHOLD, ScatterPlot } from './ScatterPlot';
import { boxFromValues } from './stats';
import { SegmentedBar, StatTile } from './StatTile';

beforeAll(() => {
  // jsdom has no canvas; the charts must cope with a missing context.
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(null);
});

const edges = [0, 10, 20, 30, 40];

describe('Histogram', () => {
  it('draws one bar per non-empty bin with a text summary', () => {
    render(
      <Histogram
        title="Age"
        width={400}
        edges={edges}
        series={[{ key: 'all', label: 'Age', color: seriesColor(0), counts: [5, 20, 0, 3] }]}
        unit="rows"
      />
    );
    const chart = screen.getByRole('img', { name: /Age\. Histogram/ });
    expect(chart.querySelectorAll('path')).toHaveLength(3);
    expect(chart).toHaveAccessibleName(
      'Age. Histogram, 4 bins from 0 to 40. 28 rows, most in 10 – 20 (20, 71%).'
    );
  });

  it('reads bins with the keyboard and shows a tooltip', () => {
    render(
      <Histogram
        title="Age"
        width={400}
        edges={edges}
        series={[{ key: 'all', label: 'Age', color: seriesColor(0), counts: [5, 20, 0, 3] }]}
      />
    );
    const slider = screen.getByRole('slider');
    fireEvent.focus(slider);
    expect(slider).toHaveAttribute('aria-valuetext', '0 – 10: 5 (18%)');
    fireEvent.keyDown(slider, { key: 'ArrowRight' });
    expect(slider).toHaveAttribute('aria-valuetext', '10 – 20: 20 (71%)');
    expect(screen.getByText('10 – 20')).toBeInTheDocument();
    fireEvent.keyDown(slider, { key: 'End' });
    expect(slider).toHaveAttribute('aria-valuenow', '4');
  });

  it('overlays several series as outlines with a legend, as shares', () => {
    render(
      <Histogram
        title="Age by churn"
        width={400}
        edges={edges}
        normalize
        series={[
          { key: 'no', label: 'no', color: seriesColor(0), counts: [10, 10, 0, 0] },
          { key: 'yes', label: 'yes', color: seriesColor(1), counts: [0, 1, 1, 2] },
        ]}
      />
    );
    const chart = screen.getByRole('img', { name: /Age by churn/ });
    // An area and an outline per series.
    expect(chart.querySelectorAll('path')).toHaveLength(4);
    expect(
      within(screen.getByRole('list', { name: 'Legend' })).getAllByRole('listitem')
    ).toHaveLength(2);
    expect(chart.textContent).toContain('%');
  });

  it('uses a log axis and offers a table view', async () => {
    render(
      <Histogram
        title="Income"
        width={400}
        edges={edges}
        log
        series={[{ key: 'all', label: 'Income', color: seriesColor(0), counts: [1137, 1, 0, 2] }]}
      />
    );
    const chart = screen.getByRole('img', { name: /Income/ });
    const ticks = Array.from(chart.querySelectorAll('text')).map((t) => t.textContent);
    expect(ticks).toEqual(expect.arrayContaining(['1', '10', '100', '1K']));
    fireEvent.click(screen.getByRole('button', { name: 'Show Income as table' }));
    const table = screen.getByRole('table', { name: 'Income (table)' });
    expect(within(table).getAllByRole('row')).toHaveLength(5);
    expect(within(table).getByText('1,137')).toBeInTheDocument();
  });

  it('labels whole-number and date bins', () => {
    expect(binLabels([17.5, 18.5, 19.5], { integer: true })).toEqual(['18', '19']);
    expect(binLabels([18, 24, 30], { integer: true })).toEqual(['18–23', '24–30']);
    expect(binLabels([Date.UTC(2021, 0, 1), Date.UTC(2022, 0, 1)], { time: true })).toEqual([
      '2021-01-01 – 2022-01-01',
    ]);
  });

  it('says when there is nothing to show', () => {
    render(<Histogram title="Empty" width={300} edges={[]} series={[]} />);
    expect(screen.getByText('No values to show')).toBeInTheDocument();
  });
});

describe('BarChart', () => {
  const rows = [
    { key: 'berlin', label: 'Berlin', values: [260] },
    { key: 'hamburg', label: 'Hamburg', values: [231] },
    { key: 'other', label: 'Other', values: [12], muted: true },
  ];

  it('draws labelled bars with values at their ends', () => {
    render(<BarChart title="City" width={400} rows={rows} unit="rows" />);
    const chart = screen.getByRole('img', { name: /City\. 3 bars/ });
    expect(within(chart).getByText('Berlin')).toBeInTheDocument();
    expect(within(chart).getByText('260')).toBeInTheDocument();
    expect(chart.querySelectorAll('path')).toHaveLength(3);
  });

  it('selects a bar with Enter', () => {
    const onSelect = vi.fn();
    render(
      <BarChart title="City" width={400} rows={rows} onSelect={onSelect} selectHint="open it" />
    );
    const slider = screen.getByRole('slider', { name: /Enter to open it/ });
    fireEvent.focus(slider);
    fireEvent.keyDown(slider, { key: 'ArrowDown' });
    expect(slider).toHaveAttribute('aria-valuetext', 'Hamburg: 231');
    fireEvent.keyDown(slider, { key: 'Enter' });
    expect(onSelect).toHaveBeenCalledWith('hamburg');
  });

  it('stacks classes per category as shares', () => {
    render(
      <BarChart
        title="Plan by churn"
        width={400}
        rows={[
          { key: 'basic', label: 'basic', values: [30, 10] },
          { key: 'pro', label: 'pro', values: [5, 0] },
        ]}
        series={[
          { key: 'no', label: 'no', color: seriesColor(0) },
          { key: 'yes', label: 'yes', color: seriesColor(1) },
        ]}
        layout="stacked"
        normalize
      />
    );
    const chart = screen.getByRole('img', { name: /Plan by churn/ });
    expect(chart).toHaveAccessibleName(/basic: no 30 \(75%\), yes 10 \(25%\)/);
    // basic has two segments, pro one (zero draws nothing).
    expect(chart.querySelectorAll('path')).toHaveLength(3);
    expect(chart.textContent).toContain('100%');
  });
});

describe('BoxPlot', () => {
  it('describes the five numbers per group', () => {
    render(
      <BoxPlot
        title="Income by class"
        width={400}
        groups={[
          { key: 'a', label: 'no', box: boxFromValues([1, 2, 3, 4, 5, 6, 7, 8, 100])! },
          { key: 'b', label: 'yes', box: boxFromValues([2, 3, 4])! },
        ]}
      />
    );
    const chart = screen.getByRole('img', { name: /Income by class/ });
    expect(chart).toHaveAccessibleName(/no: median 5, q1 – q3 3 – 7, whiskers 1 – 8/);
    expect(chart).toHaveAccessibleName(/outliers \(1\.5 × iqr\) 1/);
    // The far outlier cuts the axis and is named at the edge.
    expect(chart.textContent).toContain('▸ 100');
  });

  it('names dates in full, also at the cut edge', () => {
    const day = 86_400_000;
    const start = Date.UTC(2021, 0, 4);
    const values = [0, 1, 2, 3, 4, 5, 6, 7].map((d) => start + d * day);
    const late = Date.UTC(2023, 5, 1, 13, 5);
    render(
      <BoxPlot
        title="Signup"
        width={400}
        time
        groups={[{ key: 'all', label: 'signup', box: boxFromValues([...values, late])! }]}
      />
    );
    const chart = screen.getByRole('img', { name: /Signup/ });
    expect(chart).toHaveAccessibleName(/median 2021-01-08, q1 – q3 2021-01-06 – 2021-01-10/);
    expect(chart.textContent).toContain('▸ 2023-06-01 13:05:00');
    expect(chart.textContent).not.toMatch(/\d{12}/);
  });
});

describe('ScatterPlot', () => {
  it('draws SVG dots for small samples and reports the trend', () => {
    render(
      <ScatterPlot
        title="y vs x"
        width={400}
        x={[1, 2, 3, 4]}
        y={[2, 4, 6, 8.5]}
        xLabel="x"
        yLabel="y"
        trend
        color={{
          type: 'category',
          label: 'group',
          values: [0, 0, 1, 2],
          categories: [
            { key: 'a', label: 'a', slot: 0 },
            { key: 'b', label: 'b', slot: 1 },
            { key: 'other', label: 'Other', slot: -1 },
          ],
        }}
      />
    );
    const chart = screen.getByRole('img', { name: /y vs x\. 4 points/ });
    expect(chart.querySelectorAll('circle')).toHaveLength(4);
    expect(chart.querySelectorAll('line[class*="trend"]')).toHaveLength(1);
    expect(screen.getByText(/Trend \(least squares\): r = 1\.00/)).toBeInTheDocument();
    expect(
      within(screen.getByRole('list', { name: 'Legend' })).getAllByRole('listitem')
    ).toHaveLength(3);
    const slider = screen.getByRole('slider');
    fireEvent.focus(slider);
    expect(slider).toHaveAttribute('aria-valuetext', 'x 1, y 2, group a');
  });

  it('switches to a canvas for many points', () => {
    const n = CANVAS_THRESHOLD + 10;
    const x = Float64Array.from({ length: n }, (_, i) => i);
    const y = Float64Array.from({ length: n }, (_, i) => (i * 7) % 13);
    const { container } = render(
      <ScatterPlot
        title="Many"
        width={400}
        x={x}
        y={y}
        xLabel="x"
        yLabel="y"
        color={{ type: 'numeric', label: 'value', values: y }}
      />
    );
    expect(container.querySelector('canvas')).toBeInTheDocument();
    expect(
      screen.getByRole('img', { name: /Many\. 1,510 points/ }).querySelectorAll('circle')
    ).toHaveLength(0);
    expect(screen.getByRole('group', { name: /Colour: value from 0 to 12/ })).toBeInTheDocument();
  });

  it('draws the trend line and the focused point above the canvas', () => {
    const n = CANVAS_THRESHOLD + 10;
    const x = Float64Array.from({ length: n }, (_, i) => i);
    const y = Float64Array.from({ length: n }, (_, i) => i + ((i * 7) % 13));
    const { container } = render(
      <ScatterPlot title="Many" width={400} x={x} y={y} xLabel="x" yLabel="y" trend />
    );
    fireEvent.focus(screen.getByRole('slider'));
    const canvas = container.querySelector('canvas')!;
    const trend = container.querySelector('line[class*="trend"]')!;
    const marker = container.querySelector('circle')!;
    expect(trend).toBeInTheDocument();
    expect(marker).toBeInTheDocument();
    // Later in the document = painted on top of the canvas.
    expect(canvas.compareDocumentPosition(trend) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(canvas.compareDocumentPosition(marker) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  });

  it('names dates in full, not at the precision of the axis', () => {
    const x = [Date.UTC(2010, 2, 4), Date.UTC(2014, 5, 1, 13, 5), Date.UTC(2020, 0, 1)];
    render(
      <ScatterPlot title="When" width={400} x={x} y={[1, 2, 3]} xLabel="day" yLabel="y" xTime />
    );
    const slider = screen.getByRole('slider');
    fireEvent.focus(slider);
    expect(slider).toHaveAttribute('aria-valuetext', 'day 2010-03-04, y 1');
    fireEvent.keyDown(slider, { key: 'ArrowRight' });
    expect(slider).toHaveAttribute('aria-valuetext', 'day 2014-06-01 13:05:00, y 2');
  });

  it('says when no row has both values', () => {
    render(<ScatterPlot title="None" width={300} x={[]} y={[]} xLabel="x" yLabel="y" />);
    expect(screen.getByText('No rows with both values')).toBeInTheDocument();
  });
});

describe('Heatmap', () => {
  const labels = ['age', 'income', 'tenure'];
  const values = [
    [1, 0.42, -0.8],
    [0.42, 1, null],
    [-0.8, null, 1],
  ];

  it('annotates cells and names the strongest pairs', () => {
    render(
      <Heatmap
        title="Correlation"
        width={400}
        rows={labels}
        columns={labels}
        values={values}
        measure="Pearson r"
      />
    );
    const chart = screen.getByRole('img', { name: /Correlation\. 3 × 3 matrix/ });
    expect(chart).toHaveAccessibleName(
      /Strongest pairs: age and tenure \(−0\.80\), age and income \(0\.42\)/
    );
    expect(within(chart).getAllByText('−0.80')).toHaveLength(2);
    expect(within(chart).getAllByText('–')).toHaveLength(2);
    expect(strongestPairs(labels, values, 1)).toEqual([{ a: 0, b: 2, value: -0.8 }]);
  });

  it('opens a cell with the keyboard', () => {
    const onCellClick = vi.fn();
    render(
      <Heatmap
        title="Correlation"
        width={400}
        rows={labels}
        columns={labels}
        values={values}
        measure="Pearson r"
        onCellClick={onCellClick}
        clickHint="open the scatter plot"
      />
    );
    const slider = screen.getByRole('slider', { name: /Enter to open the scatter plot/ });
    fireEvent.focus(slider);
    fireEvent.keyDown(slider, { key: 'ArrowRight' });
    fireEvent.keyDown(slider, { key: 'ArrowDown' });
    expect(slider).toHaveAttribute('aria-valuetext', 'income × income: Pearson r 1.00');
    fireEvent.keyDown(slider, { key: 'ArrowLeft' });
    expect(slider).toHaveAttribute('aria-valuetext', 'income × age: Pearson r 0.42');
    fireEvent.keyDown(slider, { key: 'Enter' });
    expect(onCellClick).toHaveBeenCalledWith(1, 0);
  });
});

describe('MissingMatrix', () => {
  it('summarises the missing values per column', () => {
    render(
      <MissingMatrix
        title="Missing"
        width={400}
        columns={[
          { name: 'a', values: [1, null, 3, null] },
          { name: 'b', values: [1, 2, 3, 4] },
        ]}
      />
    );
    expect(screen.getByRole('img', { name: /Missing\. 2 columns, 4 rows/ })).toHaveAccessibleName(
      /Missing values in 1 column: a: 2 of 4 rows missing \(50%\)/
    );
    const slider = screen.getByRole('slider');
    fireEvent.focus(slider);
    fireEvent.keyDown(slider, { key: 'ArrowRight' });
    expect(slider).toHaveAttribute('aria-valuetext', 'b: 0 of 4 rows missing (0%)');
  });
});

describe('MissingMatrix at phone width', () => {
  const many = (m: number) =>
    Array.from({ length: m }, (_, i) => ({ name: `c${i}`, values: [1, null, 3] }));

  it('shrinks the strips of many columns to fit', () => {
    const { container } = render(<MissingMatrix title="Missing" width={328} columns={many(200)} />);
    const svg = screen.getByRole('img', { name: /Missing\. 200 columns/ });
    expect(Number(svg.getAttribute('width'))).toBeLessThanOrEqual(328);
    const canvas = container.querySelector('canvas')!;
    expect(44 + parseFloat(canvas.style.width)).toBeLessThanOrEqual(328);
  });

  it('scrolls inside its own box when even thin strips do not fit', () => {
    const { container } = render(
      <MissingMatrix title="Missing" width={328} columns={many(1000)} />
    );
    const scroller = container.querySelector('[data-scroll]') as HTMLElement;
    expect(scroller).toBeInTheDocument();
    const plot = scroller.firstElementChild as HTMLElement;
    expect(parseFloat(plot.style.width)).toBeGreaterThan(1000);
  });
});

describe('ChartTooltip', () => {
  it('stays inside the plot', () => {
    // Right of the point; flipped to the left in the right half; clamped if neither fits.
    expect(tooltipPosition(10, 5, { width: 120, height: 60 }, 300, 200)).toEqual({
      left: 22,
      top: 5,
    });
    expect(tooltipPosition(250, 5, { width: 120, height: 60 }, 300, 200)).toEqual({
      left: 118,
      top: 5,
    });
    expect(tooltipPosition(150, 190, { width: 200, height: 60 }, 300, 200)).toEqual({
      left: 100,
      top: 140,
    });
    expect(tooltipPosition(20, 5, { width: 400, height: 60 }, 300, 200).left).toBe(0);
  });

  it('measures itself and keeps inside the plot', () => {
    const width = vi.spyOn(HTMLElement.prototype, 'offsetWidth', 'get').mockReturnValue(200);
    const height = vi.spyOn(HTMLElement.prototype, 'offsetHeight', 'get').mockReturnValue(50);
    try {
      const { container } = render(
        <div style={{ position: 'relative' }}>
          <ChartTooltip x={150} y={10} width={300}>
            Tip
          </ChartTooltip>
        </div>
      );
      const tip = container.querySelector('[aria-hidden="true"]') as HTMLElement;
      expect(tip.style.left).toBe('100px');
      expect(tip.style.transform).toBe('');
    } finally {
      width.mockRestore();
      height.mockRestore();
    }
  });
});

describe('StatTile and SegmentedBar', () => {
  it('shows a figure with its label and parts', () => {
    render(
      <StatTile label="Split" value="840 / 180 / 180">
        <SegmentedBar
          label="Rows per split"
          segments={[
            { key: 't', label: 'Training', value: 840, color: seriesColor(0) },
            { key: 'v', label: 'Validation', value: 180, color: seriesColor(1) },
            { key: 'x', label: 'Test', value: 0, color: seriesColor(2) },
          ]}
        />
      </StatTile>
    );
    expect(screen.getByText('840 / 180 / 180')).toBeInTheDocument();
    const bar = screen.getByRole('img', { name: /Rows per split/ });
    expect(bar).toHaveAccessibleName(
      'Rows per split: Training 840 (82%), Validation 180 (18%), Test 0 (0%)'
    );
    expect(bar.children).toHaveLength(2);
  });
});
