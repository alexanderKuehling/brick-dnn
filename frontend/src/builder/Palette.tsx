import { Fragment, useMemo, useRef, useState } from 'react';
import { IconChartDots3, IconSearch, IconTopologyStar3 } from '@tabler/icons-react';
import { Group, ScrollArea, Stack, Text, TextInput, Tooltip, UnstyledButton } from '@mantine/core';
import type { LayerCatalog, LayerSpec } from '@/api/types';
import { i18n, useTranslation } from '@/i18n';
import { categoryColor } from '@/theme';
import { categoryLabel } from './categories';
import { isClassicalSpec, outputKind } from './kinds';
import { startPointerDrag, type Point } from './pointerDrag';
import classes from './Palette.module.css';

type Section = 'neural' | 'classical';

/** Headings between the neural layers and the classical bricks (scikit-learn). */
const SECTION_ICONS: Record<Section, React.ReactNode> = {
  neural: <IconTopologyStar3 size={14} aria-hidden />,
  classical: <IconChartDots3 size={14} aria-hidden />,
};

/** Short descriptions under the category headings (`builder.palette.hints`). */
function categoryHint(category: string): string | null {
  const key = `builder.palette.hints.${category}`;
  return i18n.exists(key) ? String(i18n.t(key as 'builder.palette.hints.Kernels')) : null;
}

interface PaletteProps {
  catalog: LayerCatalog;
  onAdd: (spec: LayerSpec) => void;
  /** A layer was dragged out of the palette and released at `point` (client coordinates). */
  onDrop?: (spec: LayerSpec, point: Point) => void;
  /** A drag out of the palette started. */
  onDragStart?: () => void;
}

export function Palette({ catalog, onAdd, onDrop, onDragStart }: PaletteProps) {
  const { t } = useTranslation();
  const [query, setQuery] = useState('');

  const groups = useMemo(() => {
    const q = query.trim().toLowerCase();
    const matches = catalog.layers.filter(
      (layer) =>
        !q ||
        layer.label.toLowerCase().includes(q) ||
        layer.category.toLowerCase().includes(q) ||
        categoryLabel(layer.category).toLowerCase().includes(q) ||
        layer.description.toLowerCase().includes(q)
    );
    return catalog.categories
      .map((category) => ({ category, layers: matches.filter((l) => l.category === category) }))
      .filter((group) => group.layers.length > 0);
  }, [catalog, query]);
  // Neural layers and classical bricks get a heading each, if the server has both.
  const sectioned = useMemo(() => catalog.layers.some(isClassicalSpec), [catalog]);
  const sectionOf = (layers: LayerSpec[]): Section | null => {
    if (!sectioned || layers.every((layer) => layer.family === 'shared')) {
      return null;
    }
    return layers.every(isClassicalSpec) ? 'classical' : 'neural';
  };

  return (
    <Stack gap="xs" p="sm" h="100%">
      <TextInput
        placeholder={t('builder.palette.search')}
        aria-label={t('builder.palette.search')}
        leftSection={<IconSearch size={16} />}
        value={query}
        onChange={(event) => setQuery(event.currentTarget.value)}
      />
      <ScrollArea style={{ flex: 1 }} offsetScrollbars type="auto">
        {groups.map(({ category, layers }, index) => {
          const section = sectionOf(layers);
          const previous = index > 0 ? sectionOf(groups[index - 1].layers) : null;
          return (
            <Fragment key={category}>
              {section && section !== previous && <SectionHeading section={section} />}
              <div>
                <Text
                  size="xs"
                  fw={700}
                  tt="uppercase"
                  c="dimmed"
                  mt="sm"
                  mb={categoryHint(category) ? 2 : 6}
                >
                  {categoryLabel(category)}
                </Text>
                {categoryHint(category) && (
                  <Text size="xs" c="dimmed" mb={6} lh={1.35}>
                    {categoryHint(category)}
                  </Text>
                )}
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
            </Fragment>
          );
        })}
        {groups.length === 0 && (
          <Text size="sm" c="dimmed" mt="md">
            {t('builder.palette.noMatch', { query })}
          </Text>
        )}
      </ScrollArea>
      <Text size="xs" c="dimmed" className={classes.hint}>
        {t('builder.palette.hint')}
      </Text>
    </Stack>
  );
}

function SectionHeading({ section }: { section: Section }) {
  const { t } = useTranslation();
  const icon = SECTION_ICONS[section];
  const title = t(`builder.palette.sections.${section}.title`);
  const hint = t(`builder.palette.sections.${section}.hint`);
  return (
    <div className={classes.section} data-section={section}>
      <Tooltip label={hint} position="right" multiline w={240} openDelay={300} withArrow>
        <Group gap={6} wrap="nowrap" className={classes.sectionTitle}>
          {icon}
          <Text size="xs" fw={700} span>
            {title}
          </Text>
        </Group>
      </Tooltip>
    </div>
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
  // Kernels have a diamond, like the handle their connections start from.
  const kernel = outputKind(spec) === 'kernel';
  // A mouse drag ends with a click on the item; that click must not add the layer again.
  const dragged = useRef(false);

  const createGhost = () => {
    const ghost = document.createElement('div');
    ghost.className = classes.ghost;
    ghost.style.setProperty('--item-color', itemColor);
    const swatch = document.createElement('span');
    swatch.className = classes.swatch;
    if (kernel) {
      swatch.dataset.kernel = 'true';
    }
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
        <span className={classes.swatch} data-kernel={kernel || undefined} />
        {spec.label}
      </UnstyledButton>
    </Tooltip>
  );
}
