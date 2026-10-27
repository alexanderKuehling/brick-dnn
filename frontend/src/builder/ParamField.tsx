import { useEffect, useState } from 'react';
import { Group, Input, NumberInput, Select, Switch, TextInput } from '@mantine/core';
import type { ParamSpec, ParamValue } from '@/api/types';
import { optionLabel } from './format';

interface ParamFieldProps {
  param: ParamSpec;
  value: ParamValue;
  /** The value the backend inferred or resolved (shown as placeholder for `auto` params). */
  inferred?: ParamValue;
  onChange: (value: ParamValue) => void;
}

function toNumber(raw: number | string): number | null | undefined {
  if (raw === '') {
    return null;
  }
  const n = typeof raw === 'number' ? raw : Number(raw);
  return Number.isFinite(n) ? n : undefined; // undefined = incomplete input like "-"
}

function placeholderFor(param: ParamSpec, inferred: ParamValue | undefined): string {
  const fallback = inferred ?? param.default;
  const text = Array.isArray(fallback) ? fallback.join(', ') : fallback;
  if (param.auto) {
    return text === null || text === undefined ? 'auto' : `auto (${text})`;
  }
  return text === null || text === undefined ? '' : String(text);
}

export function ParamField({ param, value, inferred, onChange }: ParamFieldProps) {
  const description = param.help || undefined;

  switch (param.kind) {
    case 'bool':
      return (
        <Switch
          label={param.label}
          description={description}
          checked={Boolean(value ?? param.default)}
          onChange={(event) => onChange(event.currentTarget.checked)}
        />
      );

    case 'select':
      return (
        <Select
          label={param.label}
          description={description}
          data={param.options.map((option) => ({
            value: option,
            label: optionLabel(param.name, option),
          }))}
          value={String(value ?? param.default)}
          allowDeselect={false}
          comboboxProps={{ withinPortal: true }}
          onChange={(next) => next !== null && onChange(next)}
        />
      );

    case 'int':
    case 'float': {
      const isFloat = param.kind === 'float';
      return (
        <NumberInput
          label={param.label}
          description={description}
          value={typeof value === 'number' ? value : ''}
          placeholder={placeholderFor(param, inferred)}
          min={param.min ?? undefined}
          max={param.max ?? undefined}
          step={isFloat ? (param.max !== null && param.max <= 1 ? 0.05 : 0.1) : 1}
          allowDecimal={isFloat}
          decimalScale={isFloat ? 4 : undefined}
          onChange={(raw) => {
            const next = toNumber(raw);
            if (next !== undefined) {
              onChange(next);
            }
          }}
        />
      );
    }

    case 'ints':
      return param.length ? (
        <FixedIntsField param={param} value={value} inferred={inferred} onChange={onChange} />
      ) : (
        <ShapeField param={param} value={value} onChange={onChange} />
      );

    default:
      return null;
  }
}

const AXES: Record<number, string[]> = { 1: [''], 2: ['H', 'W'], 3: ['D', 'H', 'W'] };

/** Fixed number of integers, e.g. a 2-D kernel size: one small input per axis. */
function FixedIntsField({ param, value, inferred, onChange }: ParamFieldProps) {
  const length = param.length ?? 1;
  const current = Array.isArray(value) ? value : null;
  const fallback = (Array.isArray(inferred) ? inferred : param.default) as number[] | null;
  const axes = AXES[length] ?? Array.from({ length }, (_, i) => String(i + 1));

  return (
    <Input.Wrapper label={param.label} description={param.help || undefined}>
      <Group grow gap="xs" mt={4}>
        {axes.map((axis, index) => (
          <NumberInput
            key={axis || index}
            aria-label={`${param.label} ${axis}`.trim()}
            leftSection={axis ? <span style={{ fontSize: 11 }}>{axis}</span> : undefined}
            value={current ? current[index] : ''}
            placeholder={
              param.auto && !current
                ? `auto${fallback ? ` (${fallback[index]})` : ''}`
                : fallback
                  ? String(fallback[index])
                  : ''
            }
            min={param.min ?? undefined}
            allowDecimal={false}
            onChange={(raw) => {
              const next = toNumber(raw);
              if (next === undefined) {
                return;
              }
              if (next === null) {
                onChange(null);
                return;
              }
              const base = current ?? fallback ?? Array(length).fill(1);
              const updated = [...base];
              updated[index] = next;
              onChange(updated);
            }}
          />
        ))}
      </Group>
    </Input.Wrapper>
  );
}

export function parseIntList(text: string): number[] | null {
  const parts = text
    .split(/[\s,x×]+/)
    .map((part) => part.trim())
    .filter(Boolean);
  if (parts.length === 0) {
    return null;
  }
  const values = parts.map(Number);
  return values.every((v) => Number.isInteger(v)) ? values : null;
}

/** Variable-length list of integers such as an input shape: typed as "1, 28, 28". */
function ShapeField({ param, value, onChange }: Omit<ParamFieldProps, 'inferred'>) {
  const format = (v: ParamValue) => (Array.isArray(v) ? v.join(', ') : '');
  const [text, setText] = useState(format(value));
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // Sync when the value changes from outside (undo, other node selected...).
    if (JSON.stringify(parseIntList(text)) !== JSON.stringify(value)) {
      setText(format(value));
      setError(null);
    }
    // `text` is intentionally not a dependency: only external changes should reset it.
  }, [value]);

  return (
    <TextInput
      label={param.label}
      description={param.help || 'Comma separated, e.g. 1, 28, 28'}
      placeholder={format(param.default)}
      value={text}
      error={error}
      onChange={(event) => {
        const next = event.currentTarget.value;
        setText(next);
        if (next.trim() === '') {
          setError(null);
          onChange(null);
          return;
        }
        const parsed = parseIntList(next);
        if (parsed) {
          setError(null);
          onChange(parsed);
        } else {
          setError('Enter whole numbers separated by commas.');
        }
      }}
    />
  );
}
