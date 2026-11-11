import { useState } from 'react';
import { render, screen, userEvent, waitFor } from '@test-utils';
import {
  DEFAULT_SETTINGS,
  toConfig,
  toTuning,
  validateSettings,
  type SettingsValues,
} from './settings';
import { SettingsForm } from './SettingsForm';

let latest: SettingsValues = DEFAULT_SETTINGS;

function Harness({
  initial = DEFAULT_SETTINGS,
  builtin = false,
}: {
  initial?: SettingsValues;
  builtin?: boolean;
}) {
  const [values, setValues] = useState<SettingsValues>(initial);
  latest = values;
  return (
    <SettingsForm
      values={values}
      onChange={setValues}
      errors={validateSettings(values)}
      name=""
      namePlaceholder="Tiny MLP"
      onNameChange={() => {}}
      devices={[]}
      defaultDevice={null}
      builtin={builtin}
    />
  );
}

describe('SettingsForm', () => {
  it('keeps unfinished decimals while the learning rate is edited', async () => {
    render(<Harness />);
    const input = screen.getByLabelText('Learning rate') as HTMLInputElement;
    expect(input.value).toBe('0.001');
    await userEvent.click(input);
    input.setSelectionRange(input.value.length, input.value.length);
    await userEvent.keyboard('{Backspace}');
    expect(input.value).toBe('0.00');
    await userEvent.keyboard('5');
    expect(input.value).toBe('0.005');
    expect(toConfig(latest).learning_rate).toBe(0.005);
    expect(validateSettings(latest)).toEqual({});
  });

  it('accepts small learning rates typed from scratch', async () => {
    render(<Harness />);
    const input = screen.getByLabelText('Learning rate');
    await userEvent.clear(input);
    await userEvent.type(input, '0.0005');
    expect(input).toHaveValue('0.0005');
    expect(toConfig(latest).learning_rate).toBe(0.0005);
  });

  it('accepts decimal validation splits', async () => {
    render(<Harness />);
    const input = screen.getByLabelText('Validation split');
    await userEvent.clear(input);
    await userEvent.type(input, '12.5');
    expect(toConfig(latest).val_split).toBe(0.125);
  });

  it('holds out a test set on request', async () => {
    render(<Harness />);
    expect(screen.queryByLabelText('Test split')).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole('switch', { name: /Hold out a test set/ }));
    const input = screen.getByLabelText('Test split');
    expect(input).toHaveValue('20%');
    await userEvent.clear(input);
    await userEvent.type(input, '70');
    expect(await screen.findByText('Between 5% and 50%')).toBeInTheDocument();
    await userEvent.clear(input);
    await userEvent.type(input, '10');
    expect(toConfig(latest).test_split).toBe(0.1);
  });

  it('always tests built-in datasets on their official test images', () => {
    render(<Harness builtin />);
    // Validation data is a share of the training images, the official test images are the
    // test set: the switch is on and cannot be changed, and there is no test share.
    expect(screen.getByLabelText('Validation split')).toBeInTheDocument();
    const test = screen.getByRole('switch', { name: /official test images/ });
    expect(test).toHaveAccessibleDescription(/validation split is taken from the training images/);
    expect(test).toBeChecked();
    expect(test).toBeDisabled();
    expect(screen.queryByLabelText('Test split')).not.toBeInTheDocument();
    expect(toConfig(latest, { builtin: true }).test_split).toBe(0);
  });

  it('offers early stopping and the choice of the kept weights', async () => {
    render(<Harness />);
    await userEvent.click(screen.getByRole('button', { name: 'Advanced settings' }));
    // Wait until the section has opened: before that its fields are hidden (display: none)
    // and typing into them can get lost while it opens.
    await waitFor(() => expect(screen.getByLabelText('Early stopping')).toBeVisible());
    const patience = screen.getByLabelText('Early stopping');
    expect(patience).toHaveAttribute('placeholder', 'Off');
    expect(toConfig(latest)).toMatchObject({ patience: null, restore_best: true });
    await userEvent.type(patience, '5');
    expect(patience).toHaveValue('5');
    const keepBest = screen.getByRole('checkbox', { name: 'Keep the weights of the best epoch' });
    expect(keepBest).toBeChecked();
    await userEvent.click(keepBest);
    expect(keepBest).not.toBeChecked();
    expect(toConfig(latest)).toMatchObject({ patience: 5, restore_best: false });
  });

  it('sets up a hyperparameter search', async () => {
    render(<Harness />);
    expect(screen.queryByText('Method')).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole('switch', { name: /Hyperparameter search/ }));

    // The searched settings replace their fixed fields.
    const learningRate = screen.getByRole('textbox', { name: 'Learning rate' });
    expect(learningRate).toBeDisabled();
    expect(learningRate).toHaveAttribute('placeholder', 'Set by the search');
    expect(screen.getByLabelText('Epochs')).toHaveAccessibleDescription('Per trial of the search');
    expect(screen.getByText('10 trials of 10 epochs each.', { exact: false })).toBeInTheDocument();
    expect(screen.getByText('Fixed: Adam')).toBeInTheDocument();

    await userEvent.click(screen.getByRole('checkbox', { name: 'Search optimizer' }));
    expect(screen.getByRole('combobox', { name: 'Optimizers to try' })).toBeInTheDocument();
    expect(toTuning(latest)?.space.optimizer).toEqual(['adam', 'sgd']);

    const smallest = screen.getByLabelText('Smallest learning rate');
    await userEvent.clear(smallest);
    await userEvent.type(smallest, '0.05');
    expect(await screen.findByText('At most the maximum')).toBeInTheDocument();
    await userEvent.clear(smallest);
    await userEvent.type(smallest, '0.0005');
    expect(toTuning(latest)?.space.learning_rate).toEqual({ min: 0.0005, max: 0.01 });

    // A grid lists the learning rates and needs no number of trials.
    await userEvent.click(screen.getByRole('radio', { name: 'Grid search' }));
    expect(screen.queryByLabelText('Trials')).not.toBeInTheDocument();
    expect(screen.getByRole('combobox', { name: 'Learning rates to try' })).toBeInTheDocument();
    expect(screen.getByText(/18 trials of 10 epochs each/)).toBeInTheDocument();
    expect(toTuning(latest)).toEqual({
      method: 'grid',
      space: {
        learning_rate: [0.0001, 0.001, 0.01],
        batch_size: [32, 64, 128],
        optimizer: ['adam', 'sgd'],
      },
    });

    await userEvent.click(screen.getByRole('checkbox', { name: 'Search learning rate' }));
    await userEvent.click(screen.getByRole('checkbox', { name: 'Search batch size' }));
    await userEvent.click(screen.getByRole('checkbox', { name: 'Search optimizer' }));
    expect(screen.getByRole('alert')).toHaveTextContent('Choose at least one setting to search.');
    expect(screen.getByRole('textbox', { name: 'Learning rate' })).toBeEnabled();
  });
});
