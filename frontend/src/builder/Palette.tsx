import { useMemo, useRef, useState } from 'react';
import { IconSearch } from '@tabler/icons-react';
import { ScrollArea, Stack, Text, TextInput, Tooltip, UnstyledButton } from '@mantine/core';
import type { LayerCatalog, LayerSpec } from '@/api/types';
import { categoryColor } from '@/theme';
import { startPointerDrag, type Point } from './pointerDrag';
import classes from './Palette.module.css';

interface PaletteProps {
  catalog: LayerCatalog;
  onAdd: (spec: LayerSpec) => void;
  /** A layer was dragged out of the palette and released at `point` (client coordinates). */
  onDrop?: (spec: LayerSpec, point: Point) => void;
  /** A drag out of the palette started. */
  onDragStart?: () => void;
}

export function Palette({ catalog, onAdd, onDrop, onDragStart }: PaletteProps) {
  const [query, setQuery] = useState('');

  const groups = useMemo(() => {
    const q = query.trim().toLowerCase();
    const matches = catalog.layers.filter(
      (layer) =>
        !q ||
        layer.label.toLowerCase().includes(q) ||
        layer.category.toLowerCase().includes(q) ||
        layer.description.toLowerCase().includes(q)
    );
    return catalog.categories
      .map((category) => ({ category, layers: matches.filter((l) => l.category === category) }))
      .filter((group) => group.layers.length > 0);
  }, [catalog, query]);

  return (
    <Stack gap="xs" p="sm" h="100%">
      <TextInput
        placeholder="Search layers"
        aria-label="Search layers"
        leftSection={<IconSearch size={16} />}
        value={query}
        onChange={(event) => setQuery(event.currentTarget.value)}
      />
      <ScrollArea style={{ flex: 1 }} offsetScrollbars type="auto">
        {groups.map(({ category, layers }) => (
          <div key={category}>
            <Text size="xs" fw={700} tt="uppercase" c="dimmed" mt="sm" mb={6}>
              {category}
            </Text>
            {layers.map((spec) => (
              <PaletteItem
                key={spec.type}
                spec={spec}
                onAdd={onAdd}
                onDrop={onDrop}
                onDragStart={onDragStart}
              />
            ))}
          </div>
        ))}
        {groups.length === 0 && (
          <Text size="sm" c="dimmed" mt="md">
            No layer matches “{query}”.
          </Text>
        )}
      </ScrollArea>
      <Text size="xs" c="dimmed">
        Drag a layer onto the canvas, or tap it to append it after the selected layer.
      </Text>
    </Stack>
  );
}

interface PaletteItemProps {
  spec: LayerSpec;
  onAdd: (spec: LayerSpec) => void;
  onDrop?: (spec: LayerSpec, point: Point) => void;
  onDragStart?: () => void;
}

function PaletteItem({ spec, onAdd, onDrop, onDragStart }: PaletteItemProps) {
  const color = categoryColor(spec.category);
  const itemColor = `var(--mantine-color-${color}-6)`;
  const style = { '--item-color': itemColor } as React.CSSProperties;
  // A mouse drag ends with a click on the item; that click must not add the layer again.
  const dragged = useRef(false);

  const createGhost = () => {
    const ghost = document.createElement('div');
    ghost.className = classes.ghost;
    ghost.style.setProperty('--item-color', itemColor);
    const swatch = document.createElement('span');
    swatch.className = classes.swatch;
    ghost.append(swatch, spec.label);
    return ghost;
  };

  return (
    <Tooltip label={spec.description} position="right" multiline w={260} openDelay={500} withArrow>
      <UnstyledButton
        className={classes.item}
        style={style}
        onPointerDown={(event) => {
          dragged.current = false;
          if (onDrop) {
            startPointerDrag(event, {
              createGhost,
              onStart: () => {
                dragged.current = true;
                onDragStart?.();
              },
              onDrop: (point) => onDrop(spec, point),
            });
          }
        }}
        onContextMenu={(event) => event.preventDefault()}
        onClick={() => {
          if (!dragged.current) {
            onAdd(spec);
          }
          dragged.current = false;
        }}
      >
        <span className={classes.swatch} />
        {spec.label}
      </UnstyledButton>
    </Tooltip>
  );
}
