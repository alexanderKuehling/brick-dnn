import type { ProjectRole } from '@/api';
import {
  canDeleteRun,
  canStopRun,
  canTrain,
  filterRuns,
  isOwnDataset,
  isRelevantRun,
  NO_SCOPE,
  otherStarter,
  trainingProject,
  type TrainingScope,
} from './access';
import { summary } from './fixtures';

const scope = (role: ProjectRole | null, username = 'ada'): TrainingScope => ({
  username,
  project: { id: 5, role, shared: true },
});

const own = summary({ id: 'own', started_by: 'ada', project: null });
const ownInNetwork = summary({ id: 'own-net', started_by: 'ada', project: 5 });
const othersInNetwork = summary({ id: 'bob-net', started_by: 'bob', project: 5 });
const othersElsewhere = summary({ id: 'bob-else', started_by: 'bob', project: 8 });
const ALL = [own, ownInNetwork, othersInNetwork, othersElsewhere];

describe('training access', () => {
  it('lists the runs of the network, from everyone', () => {
    expect(filterRuns(ALL, scope('viewer'), 'network').map((r) => r.id)).toEqual([
      'own-net',
      'bob-net',
    ]);
  });

  it('lists all own runs, in and outside of networks', () => {
    expect(filterRuns(ALL, scope('editor'), 'mine').map((r) => r.id)).toEqual(['own', 'own-net']);
    // Without an open network there is nothing else to show.
    const noNetwork = { username: 'ada', project: null };
    expect(filterRuns(ALL, noNetwork, 'network').map((r) => r.id)).toEqual(['own', 'own-net']);
  });

  it('counts every run as the own one when the user is not known', () => {
    expect(filterRuns(ALL, NO_SCOPE, 'mine')).toHaveLength(4);
    expect(otherStarter(othersInNetwork, NO_SCOPE)).toBeNull();
  });

  it('names who started a run of someone else', () => {
    expect(otherStarter(othersInNetwork, scope('owner'))).toBe('bob');
    expect(otherStarter(ownInNetwork, scope('owner'))).toBeNull();
  });

  it('follows the own runs and those of the open network', () => {
    expect(ALL.filter((run) => isRelevantRun(run, scope('viewer'))).map((r) => r.id)).toEqual([
      'own',
      'own-net',
      'bob-net',
    ]);
  });

  it('lets editors and the owner stop runs of others in the network', () => {
    expect(canStopRun(othersInNetwork, scope('viewer'))).toBe(false);
    expect(canStopRun(othersInNetwork, scope('editor'))).toBe(true);
    expect(canStopRun(othersInNetwork, scope('owner'))).toBe(true);
    expect(canStopRun(own, scope('viewer'))).toBe(true);
  });

  it('lets only the owner delete runs of others in the network', () => {
    expect(canDeleteRun(othersInNetwork, scope('editor'))).toBe(false);
    expect(canDeleteRun(othersInNetwork, scope('owner'))).toBe(true);
    expect(canDeleteRun(ownInNetwork, scope('viewer'))).toBe(true);
  });

  it('leaves runs of other networks to the server', () => {
    expect(canStopRun(othersElsewhere, scope('viewer'))).toBe(true);
    expect(canDeleteRun(othersElsewhere, scope('viewer'))).toBe(true);
  });

  it('trains in the network for owners and editors only', () => {
    expect(trainingProject(scope('owner'))?.id).toBe(5);
    expect(trainingProject(scope('editor'))?.id).toBe(5);
    expect(trainingProject(scope('viewer'))).toBeNull();
    expect(trainingProject(scope(null))).toBeNull();
    expect(canTrain(scope('viewer'))).toBe(false);
    expect(canTrain(scope('editor'))).toBe(true);
    expect(canTrain(NO_SCOPE)).toBe(true);
  });

  it('tells own datasets from shared ones', () => {
    const base = { owner: 'ada', shared_in: [] };
    expect(isOwnDataset({ ...base, mine: true } as never)).toBe(true);
    expect(isOwnDataset({ ...base, mine: false } as never)).toBe(false);
  });
});
