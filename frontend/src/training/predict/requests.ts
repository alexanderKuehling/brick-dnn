import {
  api,
  ApiError,
  type PredictionErrorBody,
  type PredictionRequest,
  type PredictionResult,
} from '@/api';
import { chunks, shiftMessage } from './rows';

/** Why a prediction failed: a message, the problems per row and column, and whether a budget is used up. */
export interface PredictionProblem {
  message: string;
  errors: string[];
  budget: boolean;
}

export function predictionProblem(error: unknown, offset = 0): PredictionProblem {
  if (error instanceof ApiError) {
    const body = (error.data ?? {}) as Partial<PredictionErrorBody> & { code?: string };
    const errors = Array.isArray(body.errors)
      ? body.errors.map((e) => shiftMessage(e, offset))
      : [];
    const message = shiftMessage(error.message, offset);
    return {
      message,
      errors: errors.length > 1 ? errors : [],
      budget: error.status === 403 && body.code === 'budget_exceeded',
    };
  }
  return {
    message: (error as Error).message || 'The prediction failed.',
    errors: [],
    budget: false,
  };
}

/**
 * Predict all `inputs`, in requests of at most `size` samples (the server's limit), one after
 * the other. `onProgress` gets the samples done so far. Throws a PredictionProblem whose row
 * numbers count from the first sample of all.
 */
export async function predictAll(
  runId: string,
  inputs: unknown[],
  body: Omit<PredictionRequest, 'inputs'>,
  size: number,
  onProgress?: (done: number) => void,
  signal?: AbortSignal
): Promise<PredictionResult> {
  const answers: PredictionResult[] = [];
  const warnings: string[] = [];
  let done = 0;
  for (const part of chunks(inputs, size)) {
    let answer: PredictionResult;
    try {
      answer = await api.training.runs.predict(runId, { ...body, inputs: part }, signal);
    } catch (error) {
      const problem = predictionProblem(error, done);
      if (problem.errors.length === 0 && done > 0) {
        problem.message = `${problem.message} (rows ${done + 1}–${done + part.length})`;
      }
      throw problem;
    }
    const offset = done;
    warnings.push(...(answer.warnings ?? []).map((w) => shiftMessage(w, offset)));
    answers.push(answer);
    done += part.length;
    onProgress?.(done);
  }
  if (answers.length === 0) {
    const problem: PredictionProblem = {
      message: 'There is nothing to predict.',
      errors: [],
      budget: false,
    };
    throw problem;
  }
  return {
    ...answers[0],
    predictions: answers.flatMap((a) => a.predictions),
    warnings,
  };
}

export function isProblem(value: unknown): value is PredictionProblem {
  return typeof value === 'object' && value !== null && 'message' in value && 'errors' in value;
}
