/** Let the browser download `blob` as a file. */
export function downloadBlob(filename: string, blob: Blob): void {
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/** Let the browser download `text` as a file. */
export function downloadText(filename: string, text: string, type = 'text/plain'): void {
  downloadBlob(filename, new Blob([text], { type }));
}
