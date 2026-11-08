/** The host of an absolute URL (e.g. "claude.ai"), or null if there is none. */
export function urlHost(url: string | null | undefined): string | null {
  if (!url) {
    return null;
  }
  try {
    return new URL(url).host || null;
  } catch {
    return null;
  }
}
