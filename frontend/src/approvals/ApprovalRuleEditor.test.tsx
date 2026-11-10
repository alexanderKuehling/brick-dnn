import { render, screen, userEvent, waitFor } from '@test-utils';
import { vi } from 'vitest';
import { ApprovalRuleEditor } from './ApprovalRuleEditor';

const DEFAULT = { training_cpu_minutes: 15, publishing: true, deleting: false };

describe('ApprovalRuleEditor', () => {
  it('shows the rule as a sentence', () => {
    render(<ApprovalRuleEditor rule={DEFAULT} label="“laptop”" onSave={vi.fn()} />);
    expect(
      screen.getByText('Asks before trainings above 15 CPU min and publishing')
    ).toBeInTheDocument();
  });

  it('saves a new limit and switches', async () => {
    const onSave = vi.fn().mockResolvedValue(undefined);
    render(<ApprovalRuleEditor rule={DEFAULT} label="“laptop”" onSave={onSave} />);
    await userEvent.click(
      screen.getByRole('button', { name: 'Change what “laptop” may do without asking' })
    );
    const minutes = screen.getByLabelText('CPU minutes a training may use without asking');
    await userEvent.clear(minutes);
    await userEvent.type(minutes, '1');
    await userEvent.click(screen.getByLabelText('Publishing'));
    await userEvent.click(screen.getByLabelText('Deleting'));
    await userEvent.click(screen.getByRole('button', { name: 'Save' }));
    await waitFor(() =>
      expect(onSave).toHaveBeenCalledWith({
        training_cpu_minutes: 1,
        publishing: false,
        deleting: true,
      })
    );
  });

  it('never asks about trainings without the box', async () => {
    const onSave = vi.fn().mockResolvedValue(undefined);
    render(<ApprovalRuleEditor rule={DEFAULT} label="“laptop”" onSave={onSave} />);
    await userEvent.click(screen.getByRole('button', { name: /Change what/ }));
    await userEvent.click(screen.getByLabelText('Trainings above'));
    expect(screen.getByLabelText('CPU minutes a training may use without asking')).toBeDisabled();
    await userEvent.click(screen.getByRole('button', { name: 'Save' }));
    await waitFor(() =>
      expect(onSave).toHaveBeenCalledWith({
        training_cpu_minutes: null,
        publishing: true,
        deleting: false,
      })
    );
  });

  it('shows why saving failed', async () => {
    const onSave = vi.fn().mockRejectedValue(new Error('From 0 to 10,000,000 CPU minutes.'));
    render(<ApprovalRuleEditor rule={DEFAULT} label="“laptop”" onSave={onSave} />);
    await userEvent.click(screen.getByRole('button', { name: /Change what/ }));
    await userEvent.click(screen.getByRole('button', { name: 'Save' }));
    expect(await screen.findByText('From 0 to 10,000,000 CPU minutes.')).toBeInTheDocument();
  });
});
