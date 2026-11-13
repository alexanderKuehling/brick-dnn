import { GP_ANALYSIS, VOTING_ANALYSIS } from '@/builder/classicalFixtures';
import { classicalMaxRows, MODEL_KINDS } from './classical';
import { CLASSICAL_STATUS } from './fixtures';

describe('row limits of classical models', () => {
  const status = {
    ...CLASSICAL_STATUS,
    classical: {
      ...CLASSICAL_STATUS.classical!,
      max_rows: { ...CLASSICAL_STATUS.classical!.max_rows, HistGradientBoostingClassifier: 1e6 },
    },
  };

  it('are those of the model', () => {
    expect(classicalMaxRows(status, GP_ANALYSIS)).toBe(5000);
    expect(classicalMaxRows(null, GP_ANALYSIS)).toBeNull();
  });

  it('are the strictest one of the models of an ensemble', () => {
    expect(classicalMaxRows(status, VOTING_ANALYSIS)).toBe(50000); // SVC, not gradient boosting
    // A model the server does not know: no limit to show.
    expect(classicalMaxRows(CLASSICAL_STATUS, VOTING_ANALYSIS)).toBeNull();
    expect(MODEL_KINDS.VotingClassifier).toBe('Voting classifier');
  });
});
