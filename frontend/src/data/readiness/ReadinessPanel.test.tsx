import { render, screen, userEvent, within } from '@test-utils';
import { vi } from 'vitest';
import type { Readiness } from '@/api/types';
import { READINESS } from '../fixtures';
import { ReadinessPanel, type ReadinessPanelProps } from './ReadinessPanel';
import classes from './ReadinessPanel.module.css';

function setup(
  readiness: Readiness = READINESS,
  exporting = false,
  extra: Partial<ReadinessPanelProps> = {}
) {
  const onAddStep = vi.fn();
  const onExport = vi.fn();
  const view = render(
    <ReadinessPanel
      readiness={readiness}
      onAddStep={onAddStep}
      onExport={onExport}
      exporting={exporting}
      {...extra}
    />
  );
  return { onAddStep, onExport, view };
}

const READY: Readiness = {
  ...READINESS,
  ready: true,
  issues: READINESS.issues.filter((issue) => issue.severity === 'warning'),
};

describe('ReadinessPanel', () => {
  it('summarises the data and lists the problems', () => {
    setup();
    expect(screen.getByText('Not ready for training yet')).toBeInTheDocument();
    expect(screen.getByText(/4 problems must be fixed first/)).toBeInTheDocument();
    expect(screen.getByText('churn')).toBeInTheDocument();
    expect(screen.getByText('Classification')).toBeInTheDocument();
    expect(screen.getByText('840 / 180 / 180')).toBeInTheDocument();
    expect(screen.getByText('0: no')).toBeInTheDocument();
    expect(screen.getByText('1: yes')).toBeInTheDocument();

    const errors = screen.getByRole('heading', { name: 'Must be fixed' }).parentElement!;
    expect(within(errors).getAllByRole('listitem')).toHaveLength(4);
    const warnings = screen.getByRole('heading', { name: 'Worth a look' }).parentElement!;
    expect(within(warnings).getByText('The classes are imbalanced (2.7 : 1).')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Create training dataset/ })).toBeDisabled();
  });

  it('adds the fix of one problem as a step', async () => {
    const { onAddStep } = setup();
    await userEvent.click(screen.getByRole('button', { name: 'Impute 2 columns (median)' }));
    expect(onAddStep).toHaveBeenCalledTimes(1);
    expect(onAddStep).toHaveBeenCalledWith({
      type: 'impute',
      params: { columns: ['age', 'income'], strategy: 'median' },
    });
    // Not twice while the preview is updating.
    expect(screen.getAllByRole('button', { name: 'Step added' })[0]).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Fix all (3)' })).toBeInTheDocument();
  });

  it('fixes all errors at once, in order, and resets after a new check', async () => {
    const { onAddStep, view } = setup();
    await userEvent.click(screen.getByRole('button', { name: 'Fix all (4)' }));
    expect(onAddStep.mock.calls.map(([step]) => step.type)).toEqual([
      'impute',
      'encode',
      'drop_columns',
      'datetime_features',
    ]);
    expect(screen.queryByRole('button', { name: /Fix all/ })).not.toBeInTheDocument();
    expect(screen.getAllByRole('button', { name: 'Step added' })).toHaveLength(4);

    view.rerender(
      <ReadinessPanel
        readiness={{ ...READINESS }}
        onAddStep={onAddStep}
        onExport={() => {}}
        exporting={false}
      />
    );
    expect(screen.getByRole('button', { name: 'Fix all (4)' })).toBeInTheDocument();
  });

  it('exports ready data', async () => {
    const { onExport } = setup(READY);
    expect(screen.getByText('Ready for training')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /Fix all/ })).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: /Create training dataset/ }));
    expect(onExport).toHaveBeenCalled();
  });

  it('shows the export in progress', () => {
    setup(READY, true);
    expect(screen.getByRole('button', { name: /Create training dataset/ })).toHaveAttribute(
      'data-loading'
    );
  });

  it('lists a failed step, which has no fix and no column', () => {
    setup({
      ...READY,
      ready: false,
      issues: [
        {
          severity: 'error',
          column: null,
          message: "Step 2 (Clip values) failed: 'city' is a categorical column.",
          fix: null,
        },
        ...READY.issues,
      ],
    });
    expect(screen.getByText(/^1 problem must be fixed first\.$/)).toBeInTheDocument();
    const errors = screen.getByRole('heading', { name: 'Must be fixed' }).parentElement!;
    const item = within(errors).getByRole('listitem');
    expect(item).toHaveTextContent("Error: Step 2 (Clip values) failed: 'city' is a categorical");
    expect(within(item).queryByRole('button')).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /Fix all/ })).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Create training dataset/ })).toBeDisabled();
  });

  it('blocks the export while a step failed, even if an older check says ready', () => {
    setup(READY, false, { exportBlocked: 'Step 2 “Clip values” failed.' });
    expect(screen.queryByText('Ready for training')).not.toBeInTheDocument();
    expect(screen.getByText('Not ready for training yet')).toBeInTheDocument();
    expect(screen.getByText('Step 2 “Clip values” failed.')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Create training dataset/ })).toBeDisabled();
    expect(screen.getByText(/Available once it is fixed/)).toBeInTheDocument();
  });

  it('waits with fixes while a new preview is loading', () => {
    const { onAddStep } = setup(READINESS, false, { fixesDisabled: true });
    expect(screen.getByRole('button', { name: 'Impute 2 columns (median)' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Fix all (4)' })).toBeDisabled();
    expect(onAddStep).not.toHaveBeenCalled();
  });

  it('lets long fix labels wrap on narrow screens', () => {
    setup();
    const button = screen.getByRole('button', { name: 'Split signup into year, month, weekday' });
    expect(classes.fix).toBeTruthy();
    expect(button).toHaveClass(classes.fix);
    expect(button.querySelector('.mantine-Button-label')).toHaveClass(classes.fixLabel);
  });

  it('handles data without problems and without a target', () => {
    setup({
      ...READINESS,
      ready: false,
      target: null,
      task: null,
      num_classes: null,
      class_names: null,
      split_counts: null,
      issues: [],
    });
    expect(screen.getByText('Not set')).toBeInTheDocument();
    expect(screen.getByText('No problems found.')).toBeInTheDocument();
  });
});
