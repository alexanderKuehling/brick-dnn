import { ApiError } from '@/api';

/** The first message the API gave for `field`, or null. */
export function fieldError(error: unknown, field: string): string | null {
  if (error instanceof ApiError && error.data && typeof error.data === 'object') {
    const value = (error.data as Record<string, unknown>)[field];
    if (Array.isArray(value) && typeof value[0] === 'string') {
      return value[0];
    }
    if (typeof value === 'string') {
      return value;
    }
  }
  return null;
}
