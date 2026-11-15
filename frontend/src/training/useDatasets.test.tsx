/** The dataset list of the training panel (useDatasets). */
import { useEffect } from 'react';
import { render, screen, waitFor } from '@test-utils';
import { beforeEach, vi } from 'vitest';
import { api } from '@/api';
import { dataset } from './fixtures';
import { useDatasets } from './useDatasets';

vi.mock('@/api', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/api')>();
  return { ...actual, api: { training: { datasets: { list: vi.fn() } } } };
});

/** Like the setup view: a child that loads the list whenever `reloadDatasets` changes. */
function SetupLike({ reload }: { reload: () => void }) {
  useEffect(reload, [reload]);
  return null;
}

function Panel({ active }: { active: boolean }) {
  const data = useDatasets(active, null, () => true);
  return (
    <>
      <SetupLike reload={data.reloadDatasets} />
      <p>
        {data.datasets === null
          ? 'loading'
          : `${data.datasets.length} datasets: ${data.datasets.map((d) => d.name).join(', ')}`}
      </p>
    </>
  );
}

beforeEach(() => {
  vi.mocked(api.training.datasets.list).mockReset();
});

describe('useDatasets', () => {
  it('loads the list when it becomes active while the setup view is already open', async () => {
    // E.g. "Train with this data" in the Data tab opens the panel before the training status
    // (and with it `active`) has arrived. The child's request of that commit is dropped by
    // the version bump of becoming active; the list must still arrive.
    vi.mocked(api.training.datasets.list).mockResolvedValue([dataset({ id: 7, name: 'titanic' })]);
    const { rerender } = render(<Panel active={false} />);
    expect(screen.getByText('loading')).toBeInTheDocument();
    expect(api.training.datasets.list).not.toHaveBeenCalled();

    rerender(<Panel active />);
    expect(await screen.findByText('1 datasets: titanic')).toBeInTheDocument();
  });

  it('forgets the list when it becomes inactive', async () => {
    vi.mocked(api.training.datasets.list).mockResolvedValue([dataset({ id: 7, name: 'titanic' })]);
    const { rerender } = render(<Panel active />);
    expect(await screen.findByText('1 datasets: titanic')).toBeInTheDocument();
    rerender(<Panel active={false} />);
    await waitFor(() => expect(screen.getByText('loading')).toBeInTheDocument());
  });
});
