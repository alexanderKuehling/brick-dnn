import { errorMessage } from './client';

describe('errorMessage', () => {
  it('uses DRF detail messages', () => {
    expect(errorMessage({ detail: 'Nope.' }, 400)).toBe('Nope.');
  });

  it('joins field errors', () => {
    expect(
      errorMessage(
        { username: ['A user with that username already exists.'], non_field_errors: ['Bad.'] },
        400
      )
    ).toBe('Username: A user with that username already exists. Bad.');
  });

  it('falls back to the status', () => {
    expect(errorMessage(null, 500)).toBe('Request failed (HTTP 500).');
    expect(errorMessage(null, 429)).toContain('Too many requests');
  });
});
