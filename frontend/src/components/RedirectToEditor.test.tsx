import { render, screen } from '@testing-library/react';
import { MemoryRouter, Route, Routes, useLocation } from 'react-router-dom';
import { EDITOR_PATH, RedirectToEditor } from './RedirectToEditor';

function Editor() {
  const location = useLocation();
  return (
    <output>
      {JSON.stringify({
        path: location.pathname,
        search: location.search,
        hash: location.hash,
        state: location.state,
      })}
    </output>
  );
}

function renderAt(entry: { pathname: string; search?: string; hash?: string; state?: unknown }) {
  render(
    <MemoryRouter initialEntries={[entry]}>
      <Routes>
        <Route path="/" element={<RedirectToEditor />} />
        <Route path={EDITOR_PATH} element={<Editor />} />
      </Routes>
    </MemoryRouter>
  );
  return JSON.parse(screen.getByRole('status').textContent ?? '{}');
}

test('the editor is at /app', () => {
  expect(EDITOR_PATH).toBe('/app');
});

test('/ leads to the editor', () => {
  expect(renderAt({ pathname: '/' })).toEqual({
    path: '/app',
    search: '',
    hash: '',
    state: null,
  });
});

test('query, hash and state go along', () => {
  expect(
    renderAt({
      pathname: '/',
      search: '?project=5',
      hash: '#layer',
      state: { openTraining: true, run: '7' },
    })
  ).toEqual({
    path: '/app',
    search: '?project=5',
    hash: '#layer',
    state: { openTraining: true, run: '7' },
  });
});
