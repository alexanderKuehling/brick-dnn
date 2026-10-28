import { useState } from 'react';
import { render, screen, userEvent } from '@test-utils';
import { DEFAULT_SETTINGS, toConfig, validateSettings, type SettingsValues } from './settings';
import { SettingsForm } from './SettingsForm';

let latest: SettingsValues = DEFAULT_SETTINGS;

function Harness() {
  const [values, setValues] = useState<SettingsValues>(DEFAULT_SETTINGS);
  latest = values;
  return (
    <SettingsForm
      values={values}
      onChange={setValues}
      errors={{}}
      name=""
      namePlaceholder="Tiny MLP"
      onNameChange={() => {}}
      devices={[]}
      defaultDevice={null}
      showValSplit
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
});
