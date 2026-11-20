import { ApiError } from '@/api';
import { currentLanguage, i18n } from '@/i18n';

const CODES = [
  'token_invalid',
  'token_read_only',
  'forbidden',
  'name_taken',
  'too_large',
  'busy',
  'unreachable',
  'timeout',
  'not_found',
] as const;
type HubErrorCode = (typeof CODES)[number];

function isCode(value: unknown): value is HubErrorCode {
  return typeof value === 'string' && (CODES as readonly string[]).includes(value);
}

/**
 * The message of a failed request to the Hugging Face endpoints. The server's English text is
 * the most precise (it names the repository and quotes the Hub); in other languages its
 * `code` picks a translated one.
 */
export function hubErrorMessage(error: unknown): string {
  if (error instanceof ApiError) {
    const data = error.data as { code?: unknown; created_repo?: unknown } | null;
    const code = data?.code;
    if (isCode(code) && currentLanguage() !== 'en') {
      const text = i18n.t(`huggingface.errors.${code}`);
      // The upload failed after the repository was created: it is there, and empty.
      return typeof data?.created_repo === 'string'
        ? `${text} ${i18n.t('huggingface.errors.repoCreated', { repo: data.created_repo })}`
        : text;
    }
    return error.message;
  }
  return (error as Error).message;
}

/** Repository names of the Hub: "name" or "owner/name". */
const PART = '[A-Za-z0-9](?:[A-Za-z0-9._-]{0,94}[A-Za-z0-9])?';
const REPO = new RegExp(`^(?:${PART}/)?${PART}$`);

export function validRepoName(repo: string): boolean {
  const value = repo.trim();
  return REPO.test(value) && !value.includes('--') && !value.includes('..');
}

/** Labels of the Hub's licence identifiers. */
export function licenseLabel(license: string): string {
  const names: Record<string, string> = {
    'apache-2.0': 'Apache 2.0',
    mit: 'MIT',
    'bsd-3-clause': 'BSD 3-Clause',
    'cc-by-4.0': 'CC BY 4.0',
    'cc-by-sa-4.0': 'CC BY-SA 4.0',
    'cc-by-nc-4.0': 'CC BY-NC 4.0',
    'cc0-1.0': 'CC0 1.0',
    'gpl-3.0': 'GPL 3.0',
    openrail: 'OpenRAIL',
  };
  return license === 'other'
    ? i18n.t('huggingface.publish.licenseOther')
    : (names[license] ?? license);
}
