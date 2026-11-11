import { useEffect, useState } from 'react';
import { Center, Loader } from '@mantine/core';

/** The dark modules of a QR code as one SVG path (one unit per module, `margin` around). */
export function qrPath(isDark: (row: number, col: number) => boolean, count: number, margin = 0) {
  const parts: string[] = [];
  for (let row = 0; row < count; row += 1) {
    for (let col = 0; col < count; col += 1) {
      if (isDark(row, col)) {
        parts.push(`M${col + margin},${row + margin}h1v1h-1z`);
      }
    }
  }
  return parts.join('');
}

interface Matrix {
  path: string;
  size: number;
}

const MARGIN = 4; // the quiet zone the standard asks for

/**
 * `value` as a QR code, drawn as SVG. The generator (qrcode-generator, about 20 kB) is loaded
 * only when a code is shown, e.g. while setting up two-factor authentication.
 */
export function QrCode({
  value,
  size = 200,
  label,
}: {
  value: string;
  size?: number;
  label: string;
}) {
  const [matrix, setMatrix] = useState<Matrix | null>(null);

  useEffect(() => {
    let cancelled = false;
    import('qrcode-generator').then(({ default: qrcode }) => {
      const code = qrcode(0, 'M');
      code.addData(value);
      code.make();
      const count = code.getModuleCount();
      if (!cancelled) {
        setMatrix({
          path: qrPath((row, col) => code.isDark(row, col), count, MARGIN),
          size: count + 2 * MARGIN,
        });
      }
    });
    return () => {
      cancelled = true;
    };
  }, [value]);

  if (!matrix) {
    return (
      <Center w={size} h={size}>
        <Loader size="sm" />
      </Center>
    );
  }
  return (
    <svg
      role="img"
      aria-label={label}
      width={size}
      height={size}
      viewBox={`0 0 ${matrix.size} ${matrix.size}`}
      shapeRendering="crispEdges"
      style={{ display: 'block', background: '#fff', borderRadius: 4 }}
    >
      <path d={matrix.path} fill="#000" />
    </svg>
  );
}
