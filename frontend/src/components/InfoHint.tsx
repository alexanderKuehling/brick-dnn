import type { ReactNode } from 'react';
import { IconInfoCircle } from '@tabler/icons-react';
import { ActionIcon, Group, Input, Popover, Text } from '@mantine/core';
import { useDisclosure } from '@mantine/hooks';
import { i18n, useTranslation } from '@/i18n';

/**
 * A small (i) button next to a heading or a label that shows an explanation on demand, in
 * place of help text that would always be on screen. Click or Enter opens it, Escape closes it.
 */
export function InfoHint({ topic, children }: { topic: string; children: ReactNode }) {
  useTranslation();
  const [opened, { toggle, close }] = useDisclosure(false);
  return (
    <Popover
      opened={opened}
      onChange={(value) => !value && close()}
      onDismiss={close}
      position="bottom-start"
      withArrow
      shadow="md"
      width="min(20rem, calc(100vw - 2rem))"
      returnFocus
    >
      <Popover.Target>
        <ActionIcon
          variant="subtle"
          color="gray"
          size="sm"
          radius="xl"
          onClick={toggle}
          onKeyDown={(event) => {
            // Focus stays on the button: Escape closes the hint here (and not the drawer or
            // dialog around it).
            if (event.key === 'Escape' && opened) {
              event.stopPropagation();
              close();
            }
          }}
          aria-expanded={opened}
          aria-label={i18n.t('common.moreAbout', { topic })}
        >
          <IconInfoCircle size={16} aria-hidden />
        </ActionIcon>
      </Popover.Target>
      <Popover.Dropdown>
        <Text size="sm" component="div">
          {children}
        </Text>
      </Popover.Dropdown>
    </Popover>
  );
}

/**
 * The label of a field with an InfoHint beside it. The field takes `id` and no label of its
 * own, so its name stays the label alone.
 */
export function HintedLabel({
  id,
  label,
  children,
}: {
  id: string;
  label: string;
  children: ReactNode;
}) {
  return (
    <Group gap={2} wrap="nowrap" mb={2}>
      <Input.Label htmlFor={id}>{label}</Input.Label>
      <InfoHint topic={label}>{children}</InfoHint>
    </Group>
  );
}
