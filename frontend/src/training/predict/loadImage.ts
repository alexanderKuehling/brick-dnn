/**
 * Reading an image file in the browser: decoded and resized to the size the network takes
 * (smooth resampling, like the bilinear resize of the training).
 */

/** The RGBA pixels of `file` resized to `width` × `height`. Throws for files that are no images. */
export async function imagePixels(
  file: Blob,
  width: number,
  height: number
): Promise<Uint8ClampedArray> {
  const url = URL.createObjectURL(file);
  try {
    const image = await new Promise<HTMLImageElement>((resolve, reject) => {
      const element = new Image();
      element.onload = () => resolve(element);
      element.onerror = () => reject(new Error('This file cannot be read as an image.'));
      element.src = url;
    });
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext('2d');
    if (!context) {
      throw new Error('This browser cannot read images.');
    }
    context.imageSmoothingEnabled = true;
    context.imageSmoothingQuality = 'high';
    context.drawImage(image, 0, 0, width, height);
    return context.getImageData(0, 0, width, height).data;
  } finally {
    URL.revokeObjectURL(url);
  }
}
