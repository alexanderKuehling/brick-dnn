import { useState } from 'react';
import { fireEvent, render, screen, userEvent, within } from '@test-utils';
import type { ColumnInfo, PipelineStep, PrepSettings, StepReport } from '@/api/types';
import { CATALOG, OTHER_SOURCE, PREP, SOURCE, STEP_REPORTS } from '../fixtures';
import { reportsById } from '../steps';
import { PipelineEditor, type PipelineEditorProps } from './PipelineEditor';

let steps: PipelineStep[] = [];
let settings: PrepSettings = PREP.settings;
let upto: string | null = null;

function Harness({
  initialSteps = PREP.steps,
  initialSettings = PREP.settings,
  ...props
}: Partial<PipelineEditorProps> & {
  initialSteps?: PipelineStep[];
  initialSettings?: PrepSettings;
}) {
  const [stepList, setSteps] = useState(initialSteps);
  const [prepSettings, setSettings] = useState(initialSettings);
  const [uptoId, setUpto] = useState<string | null>(null);
  steps = stepList;
  settings = prepSettings;
  upto = uptoId;
  return (
    <PipelineEditor
      catalog={CATALOG}
      steps={stepList}
      onStepsChange={setSteps}
      settings={prepSettings}
      onSettingsChange={setSettings}
      reports={reportsById(STEP_REPORTS)}
      sourceColumns={SOURCE.columns}
      sources={[SOURCE, OTHER_SOURCE]}
      currentSourceId={SOURCE.id}
      upto={uptoId}
      onUptoChange={setUpto}
      previewing={false}
      task="classification"
      splitCounts={[840, 180, 180]}
      {...props}
    />
  );
}

const card = (name: RegExp | string) => screen.getByRole('listitem', { name });

async function stepAction(index: number, action: string) {
  await userEvent.click(screen.getByRole('button', { name: `Actions for step ${index}` }));
  await userEvent.click(screen.getByRole('menuitem', { name: new RegExp(`^${action}`) }));
}

describe('PipelineEditor', () => {
  it('shows the target and split settings', () => {
    render(<Harness />);
    expect(screen.getByRole('combobox', { name: 'Target column' })).toHaveValue('churn');
    expect(screen.getByText('Detected: classification.')).toBeInTheDocument();
    expect(screen.getByText(/Train 70 % · 840 rows/)).toBeInTheDocument();
    expect(screen.getByText(/Validation 15 % · 180 rows/)).toBeInTheDocument();
    expect(screen.getByRole('switch', { name: /Stratify/ })).toBeChecked();
  });

  it('shows every step with its report', () => {
    render(<Harness />);
    expect(screen.getAllByRole('listitem', { name: /^Step/ })).toHaveLength(3);

    const impute = card('Step 1: Fill missing values');
    expect(within(impute).getByText('OK')).toBeInTheDocument();
    expect(
      within(impute).getByText('Filled 180 missing values in 2 columns with the median.')
    ).toBeInTheDocument();
    expect(within(impute).getByText(/median 38,400/)).toBeInTheDocument();
    expect(within(impute).getByText(/Learns from training rows/)).toBeInTheDocument();
    expect(
      within(impute).getByText('All columns with missing values · Median')
    ).toBeInTheDocument();

    const clip = card('Step 2: Clip values');
    expect(within(clip).getByText('Error')).toBeInTheDocument();
    expect(
      within(clip).getByText("Columns: 'city' is a categorical column; numeric columns needed.")
    ).toBeInTheDocument();

    const encode = card('Step 3: Encode categories');
    expect(within(encode).getByText('Skipped')).toBeInTheDocument();
    expect(within(encode).getByText('Not run because an earlier step failed.')).toBeInTheDocument();
  });

  it('adds a step from the searchable menu and opens its settings', async () => {
    render(<Harness />);
    await userEvent.click(screen.getAllByRole('button', { name: 'Add step' })[0]);
    expect(screen.getByText('Combine data')).toBeInTheDocument();
    await userEvent.type(screen.getByLabelText('Search steps'), 'year month');
    expect(screen.queryByRole('menuitem', { name: /Drop columns/ })).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole('menuitem', { name: /Date parts/ }));

    expect(steps).toHaveLength(4);
    expect(steps[3]).toEqual({
      id: 's4',
      type: 'datetime_features',
      enabled: true,
      params: { columns: [], parts: ['year', 'month', 'weekday'], drop_original: true },
    });
    const added = card('Step 4: Date parts');
    expect(within(added).getByRole('group', { name: 'Parts' })).toBeInTheDocument();
    expect(within(added).getByText('Waiting')).toBeInTheDocument();
  });

  it('says when nothing matches the search', async () => {
    render(<Harness />);
    await userEvent.click(screen.getAllByRole('button', { name: 'Add step' })[0]);
    await userEvent.type(screen.getByLabelText('Search steps'), 'xyz');
    expect(screen.getByText('No step matches “xyz”.')).toBeInTheDocument();
  });

  it('puts combine steps first and needs another table for them', async () => {
    const { unmount } = render(<Harness sources={[SOURCE]} />);
    await userEvent.click(screen.getAllByRole('button', { name: 'Add step' })[0]);
    expect(screen.getByRole('menuitem', { name: /Join another table/ })).toBeDisabled();
    unmount();

    render(<Harness />);
    await userEvent.click(screen.getAllByRole('button', { name: 'Add step' })[0]);
    await userEvent.click(screen.getByRole('menuitem', { name: /Join another table/ }));
    expect(steps.map((s) => s.type)).toEqual(['join', 'impute', 'clip', 'encode']);
    expect(
      within(card('Step 1: Join another table')).getByText('Runs before the split')
    ).toBeInTheDocument();

    // The join cannot move below normal steps, normal steps not above it.
    await userEvent.click(screen.getByRole('button', { name: 'Actions for step 1' }));
    expect(screen.getByRole('menuitem', { name: /Move down/ })).toBeDisabled();
    await userEvent.keyboard('{Escape}');
    await userEvent.click(screen.getByRole('button', { name: 'Actions for step 2' }));
    expect(screen.getByRole('menuitem', { name: /Move up/ })).toBeDisabled();
  });

  it('turns steps off and on', async () => {
    render(<Harness />);
    await userEvent.click(screen.getByRole('switch', { name: 'Use step 1 (Fill missing values)' }));
    expect(steps[0].enabled).toBe(false);
    expect(within(card('Step 1: Fill missing values')).getByText('Off')).toBeInTheDocument();
    await userEvent.click(screen.getByRole('switch', { name: 'Use step 1 (Fill missing values)' }));
    expect(steps[0].enabled).toBe(true);
  });

  it('moves steps with the menu and with Alt+arrow keys', async () => {
    render(<Harness />);
    await stepAction(1, 'Move down');
    expect(steps.map((s) => s.id)).toEqual(['s2', 's1', 's3']);

    const toggle = within(card('Step 3: Encode categories')).getAllByRole('button')[0];
    toggle.focus();
    fireEvent.keyDown(toggle, { key: 'ArrowUp', altKey: true });
    expect(steps.map((s) => s.id)).toEqual(['s2', 's3', 's1']);
    expect(toggle).toHaveFocus();
    expect(toggle).toHaveTextContent('2Encode categories');
  });

  it('duplicates a step', async () => {
    render(<Harness />);
    await stepAction(1, 'Duplicate');
    expect(steps.map((s) => s.id)).toEqual(['s1', 's4', 's2', 's3']);
    expect(steps[1].params).toEqual(steps[0].params);
  });

  it('deletes a step and can undo it', async () => {
    render(<Harness />);
    await stepAction(2, 'Delete');
    expect(steps.map((s) => s.id)).toEqual(['s1', 's3']);
    expect(screen.getByText('Deleted “Clip values”.')).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Undo' }));
    expect(steps.map((s) => s.id)).toEqual(['s1', 's2', 's3']);
    expect(screen.queryByText(/Deleted/)).not.toBeInTheDocument();
  });

  it('previews the data after a step', async () => {
    render(<Harness />);
    const button = within(card('Step 1: Fill missing values')).getByRole('button', {
      name: 'Preview after this step',
    });
    await userEvent.click(button);
    expect(upto).toBe('s1');
    expect(button).toHaveAttribute('aria-pressed', 'true');
    expect(
      screen.getByText('The preview shows the data after step 1; later steps do not run.')
    ).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Show all steps' }));
    expect(upto).toBeNull();

    await userEvent.click(button);
    await userEvent.click(button);
    expect(upto).toBeNull();
  });

  it('shows the steps after the previewed one as not run, with the columns of the last full run', async () => {
    const columns: ColumnInfo[] = SOURCE.columns.map(({ name, kind }) => ({ name, kind }));
    const renamed = columns.map((c) => (c.name === 'age' ? { ...c, name: 'years' } : c));
    const stepList: PipelineStep[] = [
      { ...PREP.steps[0] },
      { id: 's2', type: 'rename_columns', enabled: true, params: { mapping: { age: 'years' } } },
      {
        id: 's3',
        type: 'clip',
        enabled: true,
        params: { columns: ['years'], mode: 'quantiles', lower_q: 0.01, upper_q: 0.99 },
      },
    ];
    const ok = (step: PipelineStep, after: ColumnInfo[]): StepReport => ({
      ...STEP_REPORTS[0],
      id: step.id,
      type: step.type,
      columns: after,
      summary: `Ran ${step.type}.`,
      details: [],
    });
    // While step 1 is previewed, the later steps keep the reports of the last full preview.
    const reports = {
      s1: ok(stepList[0], columns),
      s2: ok(stepList[1], renamed),
      s3: ok(stepList[2], renamed),
    };
    render(
      <Harness
        initialSteps={stepList}
        initialSettings={{ ...PREP.settings, target: 'years' }}
        reports={reports}
        upto="s1"
      />
    );
    expect(within(card('Step 1: Fill missing values')).getByText('OK')).toBeInTheDocument();
    const rename = card('Step 2: Rename columns');
    expect(within(rename).getByText('Not run')).toBeInTheDocument();
    expect(within(rename).queryByText('Ran rename_columns.')).not.toBeInTheDocument();
    const clip = card('Step 3: Clip values');
    expect(within(clip).getByText('Not run')).toBeInTheDocument();

    await userEvent.click(within(clip).getAllByRole('button')[0]);
    expect(within(clip).getAllByText('years').length).toBeGreaterThan(0);
    expect(within(clip).queryByText(/does not exist/)).not.toBeInTheDocument();
    expect(screen.queryByText(/does not exist after the steps/)).not.toBeInTheDocument();
  });

  it('gives a new step a new id, not the one of a deleted step', async () => {
    render(<Harness />);
    await stepAction(3, 'Delete');
    await userEvent.click(screen.getByRole('button', { name: 'Add step' }));
    await userEvent.click(screen.getByRole('menuitem', { name: /Date parts/ }));
    expect(steps.map((s) => s.id)).toEqual(['s1', 's2', 's4']);
    // Not the old report of the deleted step.
    expect(within(card('Step 3: Date parts')).getByText('Waiting')).toBeInTheDocument();
  });

  it('clears the preview point when its step is deleted', async () => {
    render(<Harness />);
    await userEvent.click(
      within(card('Step 2: Clip values')).getByRole('button', { name: 'Preview after this step' })
    );
    await stepAction(2, 'Delete');
    expect(upto).toBeNull();
  });

  it('edits params at once and shows the fields of the current choice', async () => {
    render(<Harness />);
    const impute = card('Step 1: Fill missing values');
    await userEvent.click(within(impute).getAllByRole('button')[0]);
    expect(within(impute).queryByLabelText('Value')).not.toBeInTheDocument();
    await userEvent.click(within(impute).getByRole('combobox', { name: 'Fill with' }));
    await userEvent.click(screen.getByRole('option', { name: 'A fixed value' }));
    expect(steps[0].params.strategy).toBe('constant');
    await userEvent.type(within(impute).getByLabelText('Value'), '0');
    expect(steps[0].params.fill_value).toBe(0);
  });

  it('keeps hidden values when visible_if hides a field', async () => {
    const filter: PipelineStep = {
      id: 'f1',
      type: 'filter_rows',
      enabled: true,
      params: { column: 'age', operator: 'between', value: 18, value2: 65, values: [] },
    };
    render(<Harness initialSteps={[filter]} />);
    const item = card('Step 1: Filter rows');
    await userEvent.click(within(item).getAllByRole('button')[0]);
    expect(within(item).getByLabelText('and')).toHaveValue('65');

    await userEvent.click(
      within(item).getByRole('combobox', { name: 'Keep rows where the value' })
    );
    await userEvent.click(screen.getByRole('option', { name: 'is one of' }));
    expect(within(item).queryByLabelText('and')).not.toBeInTheDocument();
    expect(within(item).queryByLabelText('Value')).not.toBeInTheDocument();
    expect(within(item).getByRole('combobox', { name: 'Values' })).toBeInTheDocument();
    expect(steps[0].params).toMatchObject({ operator: 'in', value: 18, value2: 65 });

    await userEvent.click(
      within(item).getByRole('combobox', { name: 'Keep rows where the value' })
    );
    await userEvent.click(screen.getByRole('option', { name: 'is between' }));
    expect(within(item).getByLabelText('and')).toHaveValue('65');
  });

  it('marks the field a step error is about', async () => {
    render(<Harness />);
    const clip = card('Step 2: Clip values');
    await userEvent.click(within(clip).getAllByRole('button')[0]);
    expect(
      within(clip).getByText("'city' is a categorical column; numeric columns needed.")
    ).toBeInTheDocument();
  });

  it('shows unknown step types as errors that can be deleted', async () => {
    render(
      <Harness
        initialSteps={[{ id: 'x1', type: 'mystery', enabled: true, params: {} }, ...PREP.steps]}
      />
    );
    const item = card('Step 1: unknown step type mystery');
    expect(within(item).getByText(/Unknown step type “mystery”/)).toBeInTheDocument();
    await stepAction(1, 'Delete');
    expect(steps.map((s) => s.id)).toEqual(['s1', 's2', 's3']);
  });

  it('explains the empty pipeline', () => {
    render(<Harness initialSteps={[]} />);
    expect(screen.getByText('No steps yet')).toBeInTheDocument();
  });

  it('changes the target and the task', async () => {
    render(<Harness />);
    await userEvent.click(screen.getByRole('combobox', { name: 'Target column' }));
    await userEvent.click(screen.getByRole('option', { name: /income/ }));
    expect(settings.target).toBe('income');

    await userEvent.click(screen.getByRole('radio', { name: 'Regression' }));
    expect(settings.task).toBe('regression');
    expect(screen.getByRole('switch', { name: /Stratify/ })).toBeDisabled();
    expect(screen.getByText('Predicts a number for every row.')).toBeInTheDocument();
  });

  it('flags a target that the steps removed', () => {
    render(<Harness initialSettings={{ ...PREP.settings, target: 'gone' }} task={null} />);
    expect(
      screen.getByText('“gone” does not exist after the steps. Choose the target again.')
    ).toBeInTheDocument();
  });

  it('validates the split shares and updates the row counts', async () => {
    render(<Harness />);
    const val = screen.getByLabelText('Validation');
    await userEvent.clear(val);
    await userEvent.type(val, '0');
    expect(screen.getByText('Use more than 0 % for validation.')).toBeInTheDocument();
    expect(settings.split.val).toBe(0.15);

    await userEvent.clear(val);
    await userEvent.type(val, '80');
    expect(
      screen.getByText('Validation and test together can be at most 90 %.')
    ).toBeInTheDocument();
    expect(settings.split.val).not.toBe(0.8);

    await userEvent.clear(val);
    await userEvent.type(val, '20');
    expect(settings.split.val).toBe(0.2);
    expect(screen.getByText(/Validation 20 % · ≈ 240 rows/)).toBeInTheDocument();

    const seed = screen.getByLabelText('Seed');
    await userEvent.clear(seed);
    await userEvent.type(seed, '7');
    expect(settings.split.seed).toBe(7);
  });
});
