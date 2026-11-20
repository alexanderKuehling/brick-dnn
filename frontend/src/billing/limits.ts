/** The limit in an answer of the API (see LimitNotice.tsx for what is shown about it). */
import { ApiError, type LimitInfo } from '@/api';

/** A limit as an answer of the API carries it: the info, and the code of the refusal. */
export type Limit = LimitInfo & { code?: string };

/** The limit in the body of a refused request (or in a part of an answer); null: none. */
export function limitOf(source: unknown): Limit | null {
  const data = source instanceof ApiError ? source.data : source;
  if (!data || typeof data !== 'object') {
    return null;
  }
  const info = data as Partial<Limit>;
  return info.limit && typeof info.limit.kind === 'string' ? (info as Limit) : null;
}

/** "/pricing?plan=starter": the pricing page of this app with the plan preselected. */
export function pricingPath(plan?: string | null): string {
  return plan ? `/pricing?plan=${encodeURIComponent(plan)}` : '/pricing';
}
