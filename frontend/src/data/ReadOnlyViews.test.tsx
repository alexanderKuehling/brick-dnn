/** The editor and the result views for a preparation shared with the user to look at. */
import { render, screen, userEvent, within } from '@test-utils';
import { vi } from 'vitest';
import type { PipelineStep } from '@/api/types';
import { WarningList } from './analysis/WarningList';
import { CATALOG, PREP, PROFILE, READINESS, SAMPLE, SOURCE, STEP_REPORTS } from './fixtures';
import { PipelineEditor } from './pipeline/PipelineEditor';
import { PrepAccessContext, type PrepAccess } from './prepAccess';
import { ReadinessPanel } from './readiness/ReadinessPanel';
import { reportsById } from './steps';
import { DataTable } from './table/DataTable';

const VIEWER: PrepAccess = { readOnly: true, sharedBy: 'bob' };
const EDITOR: PrepAccess = { readOnly: false, sharedBy: 'bob' };

function withAccess(access: PrepAccess, ui: React.ReactNode) {
  return render(<PrepAccessContext.Provider value={access}>{ui}</PrepAccessContext.Provider>);
}

function editor(access: PrepAccess, steps: PipelineStep[] = PREP.steps) {
  const props = {
    onStepsChange: vi.fn(),
    onSettingsChange: vi.fn(),
    onUptoChange: vi.fn(),
  };
  withAccess(
    access,
    <PipelineEditor
      catalog={CATALOG}
      steps={steps}
      settings={PREP.settings}
      reports={reportsById(STEP_REPORTS)}
      sourceColumns={SOURCE.columns}
      sources={[SOURCE]}
      currentSourceId={SOURCE.id}
      upto={null}
      previewing={false}
      task="classification"
      splitCounts={[840, 180, 180]}
      {...props}
    />
  );
  return props;
}

describe('read-only views', () => {
  it('shows the steps and settings without ways to change them', async () => {
    const props = editor(VIEWER);
    expect(screen.queryByRole('button', { name: /Add step/ })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /^Actions for step/ })).not.toBeInTheDocument();
    expect(screen.getByRole('combobox', { name: 'Target column' })).toHaveAttribute('readonly');
    expect(screen.getByRole('textbox', { name: 'Validation' })).toHaveAttribute('readonly');
    expect(screen.getByRole('switch', { name: /Stratify/ })).toBeDisabled();
    expect(screen.getByRole('switch', { name: 'Use step 1 (Fill missing values)' })).toBeDisabled();

    // The settings of a step can be looked at, not changed.
    const impute = screen.getByRole('listitem', { name: 'Step 1: Fill missing values' });
    await userEvent.click(within(impute).getByRole('button', { name: /Fill missing values/ }));
    const form = impute.querySelector('fieldset')!;
    expect(form).toBeDisabled();
    expect(within(form).getByRole('combobox', { name: 'Fill with' })).toBeDisabled();

    // Previewing up to a step still works.
    await userEvent.click(within(impute).getByRole('button', { name: 'Preview after this step' }));
    expect(props.onUptoChange).toHaveBeenCalledWith('s1');
    expect(props.onStepsChange).not.toHaveBeenCalled();
    expect(props.onSettingsChange).not.toHaveBeenCalled();
  });

  it('lets editors of a shared preparation edit, but not combine tables', async () => {
    const props = editor(EDITOR);
    await userEvent.click(screen.getAllByRole('button', { name: /Add step/ })[0]);
    expect(await screen.findByRole('menuitem', { name: /Join/ })).toHaveAttribute('data-disabled');
    await userEvent.click(screen.getByRole('menuitem', { name: /Fill missing values/ }));
    expect(props.onStepsChange).toHaveBeenCalled();
  });

  it('names the tables of the owner in combine steps', async () => {
    const join: PipelineStep = {
      id: 'j1',
      type: 'join',
      enabled: true,
      params: { source: 77, how: 'left', left_on: [], right_on: [] },
    };
    editor(VIEWER, [join]);
    const card = screen.getByRole('listitem', { name: /^Step 1/ });
    await userEvent.click(within(card).getByRole('button', { name: /Join/ }));
    expect(within(card).getByRole('combobox', { name: /table/i })).toHaveValue(
      'A table of bob (#77)'
    );
    expect(within(card).queryByText(/does not exist/)).not.toBeInTheDocument();
  });

  it('lists problems without fixes or export', () => {
    withAccess(
      VIEWER,
      <ReadinessPanel
        readiness={READINESS}
        onAddStep={vi.fn()}
        onExport={vi.fn()}
        exporting={false}
      />
    );
    expect(screen.getByText('Not ready for training yet')).toBeInTheDocument();
    expect(
      screen.getByText('2 feature columns have missing values (age, income).')
    ).toBeInTheDocument();
    expect(
      screen.queryByRole('button', { name: 'Impute 2 columns (median)' })
    ).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /Fix all/ })).not.toBeInTheDocument();
    expect(
      screen.queryByRole('button', { name: /Create training dataset/ })
    ).not.toBeInTheDocument();
  });

  it('keeps only “Show in Analysis” in the column menus', async () => {
    const onShowColumn = vi.fn();
    withAccess(
      VIEWER,
      <DataTable
        sample={SAMPLE}
        profile={PROFILE}
        target="churn"
        onAddStep={vi.fn()}
        onSetTarget={vi.fn()}
        onShowColumn={onShowColumn}
      />
    );
    await userEvent.click(screen.getByRole('button', { name: 'Actions for column age' }));
    const menu = await screen.findByRole('menu');
    expect(
      within(menu)
        .getAllByRole('menuitem')
        .map((item) => item.textContent)
    ).toEqual(['Show in Analysis']);
    await userEvent.click(within(menu).getByRole('menuitem', { name: 'Show in Analysis' }));
    expect(onShowColumn).toHaveBeenCalledWith('age');
  });

  it('shows the warnings of the analysis without their fixes', () => {
    withAccess(
      VIEWER,
      <WarningList warnings={PROFILE.warnings} resetKey={1} onAddStep={vi.fn()} />
    );
    expect(screen.getByText(/it looks like an ID/)).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Drop customer_id' })).not.toBeInTheDocument();
  });
});
