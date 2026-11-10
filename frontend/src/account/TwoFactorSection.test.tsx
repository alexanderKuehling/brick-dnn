import { render, screen, userEvent, waitFor, within } from '@test-utils';
import { afterEach, beforeEach, vi } from 'vitest';
import { notifications } from '@mantine/notifications';
import { api, ApiError, getTokens, setTokens, type TwoFactorState } from '@/api';
import { downloadText } from '@/builder/download';
import { qrPath } from './QrCode';
import { groupSecret, TwoFactorSection } from './TwoFactorSection';

vi.mock('@/api', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/api')>();
  return {
    ...actual,
    api: {
      twoFactor: {
        state: vi.fn(),
        setup: vi.fn(),
        enable: vi.fn(),
        disable: vi.fn(),
        recoveryCodes: vi.fn(),
      },
    },
  };
});

vi.mock('@/builder/download', () => ({ downloadText: vi.fn(), downloadBlob: vi.fn() }));

const OFF: TwoFactorState = { enabled: false, enabled_at: null, recovery_codes_left: 0 };
const ON: TwoFactorState = {
  enabled: true,
  enabled_at: '2026-11-10T10:00:00Z',
  recovery_codes_left: 10,
};
const CODES = Array.from({ length: 10 }, (_, i) => `code-${i}aaa-bbbb-cccc`);
const SECRET = 'JBSWY3DPEHPK3PXPJBSWY3DPEHPK3PXP';
const URI = `otpauth://totp/Brick%20DNN%3Aada?secret=${SECRET}&issuer=Brick+DNN`;

function renderSection(state: TwoFactorState) {
  const onChange = vi.fn();
  render(<TwoFactorSection username="ada" state={state} onChange={onChange} />);
  return onChange;
}

beforeEach(() => {
  vi.clearAllMocks();
  setTokens({ access: 'old-access', refresh: 'old-refresh' });
  vi.spyOn(notifications, 'show').mockImplementation(() => '');
});

afterEach(() => setTokens(null));

test('groupSecret', () => {
  expect(groupSecret('ABCDEFGHIJ')).toBe('ABCD EFGH IJ');
});

test('qrPath draws one square per dark module', () => {
  const path = qrPath((row, col) => row === col, 3, 1);
  expect(path).toBe('M1,1h1v1h-1zM2,2h1v1h-1zM3,3h1v1h-1z');
});

test('turns it on: password, QR code and code, recovery codes', async () => {
  vi.mocked(api.twoFactor.setup)
    .mockRejectedValueOnce(new ApiError(400, { password: ['The password is wrong.'] }, 'x'))
    .mockResolvedValueOnce({ secret: SECRET, otpauth_uri: URI });
  vi.mocked(api.twoFactor.enable)
    .mockRejectedValueOnce(new ApiError(400, { code: ['The code is wrong. Check the app.'] }, 'x'))
    .mockResolvedValueOnce({ recovery_codes: CODES, state: ON, access: 'new-a', refresh: 'new-r' });
  const onChange = renderSection(OFF);
  expect(screen.getByText(/asks for a code from an authenticator app/)).toBeInTheDocument();
  await userEvent.click(screen.getByRole('button', { name: 'Turn on…' }));
  const dialog = await screen.findByRole('dialog', { name: 'Turn on two-factor authentication' });

  await userEvent.type(within(dialog).getByLabelText(/Your password/), 'nope');
  await userEvent.click(within(dialog).getByRole('button', { name: 'Continue' }));
  expect(await within(dialog).findByText('The password is wrong.')).toBeInTheDocument();
  await userEvent.type(within(dialog).getByLabelText(/Your password/), 'right');
  await userEvent.click(within(dialog).getByRole('button', { name: 'Continue' }));

  // The QR code (rendered here from the otpauth URI) and the key to type in.
  expect(
    await within(dialog).findByRole('img', { name: 'QR code for your authenticator app' })
  ).toBeInTheDocument();
  expect(within(dialog).getByTestId('two-factor-secret')).toHaveTextContent(groupSecret(SECRET));
  const turnOn = within(dialog).getByRole('button', { name: 'Turn on' });
  expect(turnOn).toBeDisabled();
  await userEvent.type(within(dialog).getByLabelText(/Code from the app/), '000000');
  await userEvent.click(turnOn);
  expect(await within(dialog).findByText('The code is wrong. Check the app.')).toBeInTheDocument();
  await userEvent.type(within(dialog).getByLabelText(/Code from the app/), '123456');
  await userEvent.click(within(dialog).getByRole('button', { name: 'Turn on' }));

  // Recovery codes, shown once; this session goes on with new tokens.
  const list = await within(dialog).findByLabelText('Recovery codes');
  expect(within(list).getAllByText(/^code-/)).toHaveLength(10);
  expect(api.twoFactor.enable).toHaveBeenLastCalledWith('123456');
  expect(getTokens()).toEqual({ access: 'new-a', refresh: 'new-r' });
  expect(onChange).toHaveBeenCalledWith(ON);
  await userEvent.click(within(dialog).getByRole('button', { name: 'Download' }));
  expect(downloadText).toHaveBeenCalledWith(
    'brick-dnn-recovery-codes-ada.txt',
    expect.stringContaining(CODES.join('\n'))
  );
  const done = within(dialog).getByRole('button', { name: 'Done' });
  expect(done).toBeDisabled();
  await userEvent.click(within(dialog).getByLabelText('I have saved my recovery codes'));
  await userEvent.click(done);
  await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());
}, 40000); // many steps: slow when all test files run at once

test('shows the state when it is on', () => {
  renderSection({ ...ON, recovery_codes_left: 2 });
  expect(screen.getByText('On')).toBeInTheDocument();
  expect(screen.getByText(/2 recovery codes left/)).toBeInTheDocument();
  expect(screen.getByText(/Few recovery codes are left/)).toBeInTheDocument();
});

test('turning it off needs the password and a code', async () => {
  vi.mocked(api.twoFactor.disable)
    .mockRejectedValueOnce(new ApiError(400, { code: ['The code is wrong.'] }, 'x'))
    .mockResolvedValueOnce(OFF);
  const onChange = renderSection(ON);
  await userEvent.click(screen.getByRole('button', { name: 'Turn off…' }));
  const dialog = await screen.findByRole('dialog');
  const submit = within(dialog).getByRole('button', { name: 'Turn off' });
  expect(submit).toBeDisabled();
  await userEvent.type(within(dialog).getByLabelText(/Your password/), 'pw');
  await userEvent.type(within(dialog).getByLabelText(/Code from the app/), '111111');
  await userEvent.click(submit);
  expect(await within(dialog).findByText('The code is wrong.')).toBeInTheDocument();
  await userEvent.click(submit);
  await waitFor(() => expect(onChange).toHaveBeenCalledWith(OFF));
  expect(api.twoFactor.disable).toHaveBeenLastCalledWith('pw', '111111');
});

test('makes new recovery codes', async () => {
  vi.mocked(api.twoFactor.recoveryCodes).mockResolvedValue({ recovery_codes: CODES, state: ON });
  const onChange = renderSection({ ...ON, recovery_codes_left: 1 });
  await userEvent.click(screen.getByRole('button', { name: 'New recovery codes…' }));
  const dialog = await screen.findByRole('dialog', { name: 'New recovery codes' });
  await userEvent.type(within(dialog).getByLabelText(/Your password/), 'pw');
  await userEvent.type(within(dialog).getByLabelText(/Code from the app/), '222222');
  await userEvent.click(within(dialog).getByRole('button', { name: 'Make new codes' }));
  expect(await within(dialog).findByLabelText('Recovery codes')).toBeInTheDocument();
  expect(onChange).toHaveBeenCalledWith(ON);
});
