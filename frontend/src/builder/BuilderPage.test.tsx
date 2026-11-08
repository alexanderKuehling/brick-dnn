/** Opening the training panel from the Data tab ("Train with this data"). */
import { act, render, screen } from '@test-utils';
import { MemoryRouter, useLocation, useNavigate, type NavigateFunction } from 'react-router-dom';
import { beforeEach, vi } from 'vitest';
import { useAuth } from '@/auth/AuthContext';
import type { SourceChoice } from '@/training/settings';
import { useTraining, type TrainingController } from '@/training/useTraining';
import { BuilderPage } from './BuilderPage';

vi.mock('@/auth/AuthContext', () => ({ useAuth: vi.fn() }));
vi.mock('@/training/useTraining', () => ({ useTraining: vi.fn() }));
vi.mock('./useCatalog', () => ({
  useCatalog: () => ({ catalog: null, examples: [], error: null, reload: vi.fn() }),
}));
vi.mock('./useAnalysis', () => ({
  useAnalysis: () => ({ analysis: null, loading: false, error: null }),
}));

let navigate: NavigateFunction = () => undefined;

function Probe() {
  const location = useLocation();
  navigate = useNavigate();
  return (
    <output data-testid="location">
      {location.pathname}
      {location.state ? ` ${JSON.stringify(location.state)}` : ''}
    </output>
  );
}

function setup() {
  let source: SourceChoice = { tab: 'builtin', builtin: 'iris', upload: 7 };
  const data = {
    setSource: vi.fn((update: SourceChoice | ((current: SourceChoice) => SourceChoice)) => {
      source = typeof update === 'function' ? update(source) : update;
    }),
    reloadDatasets: vi.fn(),
  };
  const training = { open: vi.fn(), opened: false, data } as unknown as TrainingController;
  vi.mocked(useTraining).mockReturnValue(training);
  render(
    <MemoryRouter initialEntries={['/']}>
      <BuilderPage />
      <Probe />
    </MemoryRouter>
  );
  return { training, data, source: () => source };
}

beforeEach(() => {
  vi.clearAllMocks();
  localStorage.clear();
  vi.mocked(useAuth).mockReturnValue({
    user: null,
    ready: true,
    login: vi.fn(),
    register: vi.fn(),
    logout: vi.fn(),
  });
});

describe('BuilderPage', () => {
  it('opens the training panel with the dataset from the Data tab, also when it is open already', () => {
    const { training, data, source } = setup();
    expect(training.open).not.toHaveBeenCalled();

    // E.g. "Train with it" in a notification that outlived the Data tab.
    act(() => navigate('/', { state: { openTraining: true, dataset: 42 } }));
    expect(training.open).toHaveBeenCalledTimes(1);
    expect(source()).toEqual({ tab: 'upload', builtin: 'iris', upload: 42 });
    expect(data.reloadDatasets).toHaveBeenCalled();
    // The state is cleared, so reloading the page does not open the panel again.
    expect(screen.getByTestId('location')).toHaveTextContent(/^\/$/);
  });

  it('only opens the panel without a dataset', () => {
    const { training, data } = setup();
    act(() => navigate('/', { state: { openTraining: true } }));
    expect(training.open).toHaveBeenCalledTimes(1);
    expect(data.setSource).not.toHaveBeenCalled();
  });
});
