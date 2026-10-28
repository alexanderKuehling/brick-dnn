import { useMemo, useState } from 'react';
import { IconSearch } from '@tabler/icons-react';
import { ScrollArea, Stack, Text, TextInput, Tooltip, UnstyledButton } from '@mantine/core';
import type { LayerCatalog, LayerSpec } from '@/api/types';
import { categoryColor } from '@/theme';
import classes from './Palette.module.css';

/** MIME type used when dragging a layer from the palette onto the canvas. */
export const DRAG_MIME = 'application/x-brickdnn-layer';

interface PaletteProps {
  catalog: LayerCatalog;
  onAdd: (spec: LayerSpec) => void;
}

export function Palette({ catalog, onAdd }: PaletteProps) {
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
              <PaletteItem key={spec.type} spec={spec} onAdd={onAdd} />
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
        Drag a layer onto the canvas, or click it to append it after the selected layer.
      </Text>
    </Stack>
  );
}

function PaletteItem({ spec, onAdd }: { spec: LayerSpec; onAdd: (spec: LayerSpec) => void }) {
  const color = categoryColor(spec.category);
  const style = {
    '--item-color': `var(--mantine-color-${color}-6)`,
  } as React.CSSProperties;

  return (
    <Tooltip label={spec.description} position="right" multiline w={260} openDelay={500} withArrow>
      <UnstyledButton
        className={classes.item}
        style={style}
        draggable
        onDragStart={(event) => {
          event.dataTransfer.setData(DRAG_MIME, spec.type);
          event.dataTransfer.effectAllowed = 'move';
        }}
        onClick={() => onAdd(spec)}
      >
        <span className={classes.swatch} />
        {spec.label}
      </UnstyledButton>
    </Tooltip>
  );
}
