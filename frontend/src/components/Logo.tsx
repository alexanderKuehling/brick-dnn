import { Group, Text } from '@mantine/core';

export function Logo({ size = 30 }: { size?: number }) {
  return (
    <Group gap={8} wrap="nowrap">
      <img src={`${import.meta.env.BASE_URL}favicon.svg`} width={size} height={size} alt="" />
      <Text fw={800} size="lg" style={{ letterSpacing: '-0.02em' }}>
        Brick DNN
      </Text>
    </Group>
  );
}
