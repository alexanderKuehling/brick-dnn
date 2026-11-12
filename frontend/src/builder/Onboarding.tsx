/**
 * A short guide for a first visit to the builder (Mantine's Tour): the palette, connecting
 * layers, the analysis, code and export, training. It starts once per browser (remembered in
 * localStorage) and can be opened again with the "?" button of the toolbar. And the cards of
 * an empty canvas: start from an example.
 */
import { useEffect, useState } from 'react';
import { IconSchool } from '@tabler/icons-react';
import { Paper, SimpleGrid, Text, Title, Tour, UnstyledButton } from '@mantine/core';
import type { Example } from '@/api/types';
import { DEMO } from '@/demo';
import { LegalLinks } from '@/legal/LegalLinks';

/** Set once the guide was finished or skipped in this browser. */
export const GUIDE_KEY = 'brickdnn:guide-seen';

export function guideSeen(): boolean {
  try {
    return localStorage.getItem(GUIDE_KEY) !== null;
  } catch {
    return true; // Storage unavailable: better never than on every visit.
  }
}

export function markGuideSeen(): void {
  try {
    localStorage.setItem(GUIDE_KEY, new Date().toISOString());
  } catch {
    // Nothing to remember it in: it starts again next time, which is acceptable.
  }
}

/** Whether the guide starts by itself on a first visit; tests start it themselves. */
export function guideStartsByItself(): boolean {
  return import.meta.env.MODE !== 'test';
}

/** An element to point at (a CSS selector) and where the tooltip goes next to it. */
interface GuideTarget {
  selector: string;
  position: 'top' | 'bottom' | 'left' | 'right';
}

const tour = (name: string, position: GuideTarget['position'] = 'bottom'): GuideTarget => ({
  selector: `[data-tour="${name}"]`,
  position,
});

interface GuideStep {
  /** Elements to point at; the first one that is visible is used (none: centred). */
  targets: GuideTarget[];
  title: string;
  body: string;
}

export const GUIDE_STEPS: GuideStep[] = [
  {
    targets: [tour('palette', 'right'), tour('palette-button')],
    title: 'Layers come from the palette',
    body:
      'Drag a layer onto the canvas, or click it: it is added after the selected layer and ' +
      'connected to it. On a phone, the palette opens with this button. Every network ' +
      'starts with an Input layer.',
  },
  {
    // The first layer and its dots; an empty canvas: the middle of the screen.
    targets: [{ selector: '.react-flow__node', position: 'bottom' }],
    title: 'Connect the layers',
    body:
      'Drag from the dot on the right of a layer to the left side of the next one. Several ' +
      'connections into one layer need Add or Concat; select a connection and press Delete ' +
      'to remove it.',
  },
  {
    targets: [tour('status')],
    title: 'Brick DNN checks while you build',
    body:
      'Every layer shows its output shape and parameters, and this badge says whether the ' +
      'network is valid (click it for a summary). Select a layer to change it in the ' +
      'properties: sizes like in_features fill themselves in.',
  },
  {
    targets: [tour('code'), tour('export'), tour('export-icon')],
    title: 'Code and export',
    body:
      'See the generated PyTorch code (scikit-learn for Gaussian processes and SVMs), or ' +
      'download it as a .py file that runs on its own.',
  },
  {
    targets: [tour('train'), tour('train-icon')],
    title: 'Train it',
    body: DEMO
      ? 'In the full app at brickdnn.com, press Train, pick a built-in dataset such as MNIST ' +
        'or Fashion-MNIST, and watch loss and accuracy live. The ? button shows this guide again.'
      : 'Press Train, pick a built-in dataset such as MNIST or Fashion-MNIST (or your own CSV ' +
        'file), and watch loss and accuracy live; it needs an account. The ? button shows ' +
        'this guide again.',
  },
];

function visible(target: GuideTarget): boolean {
  const element = document.querySelector<HTMLElement>(target.selector);
  if (!element) {
    return false;
  }
  const rect = element.getBoundingClientRect();
  return rect.width > 0 && rect.height > 0;
}

/** Each step's first visible target (undefined: the step is centred). */
function resolveTargets(): (GuideTarget | undefined)[] {
  return GUIDE_STEPS.map((step) => step.targets.find(visible));
}

export function OnboardingGuide({ opened, onClose }: { opened: boolean; onClose: () => void }) {
  const [targets, setTargets] = useState<(GuideTarget | undefined)[]>([]);
  const [step, setStep] = useState(0);

  // Where each step points is decided when the guide opens (phones show other buttons).
  useEffect(() => {
    if (opened) {
      setTargets(resolveTargets());
      setStep(0);
    } else {
      setTargets([]);
    }
  }, [opened]);

  const close = () => {
    markGuideSeen();
    onClose();
  };

  return (
    <Tour
      // Once the targets are known: a step reads its target when it opens.
      active={opened && targets.length > 0}
      step={step}
      onStepChange={setStep}
      onClose={close}
      maxWidth={340}
      // Skip (on every step) and Escape end it; "Done" on the last step.
      withCloseButton={false}
      spotlightRadius={8}
      labels={{ skip: 'Skip', back: 'Back', next: 'Next', close: 'Done' }}
      aria-label="Guide to the builder"
    >
      {GUIDE_STEPS.map((guideStep, index) => (
        <Tour.Step
          key={guideStep.title}
          target={targets[index]?.selector}
          title={guideStep.title}
          position={targets[index]?.position ?? 'bottom'}
        >
          {guideStep.body}
        </Tour.Step>
      ))}
    </Tour>
  );
}

/** The examples an empty canvas offers first: one of each kind. */
const START_EXAMPLES = ['mlp_mnist', 'cnn_mnist', 'unet', 'gp_regression'];

export function startExamples(examples: Example[]): Example[] {
  const chosen = START_EXAMPLES.map((id) => examples.find((example) => example.id === id)).filter(
    (example): example is Example => Boolean(example)
  );
  return chosen.length > 0 ? chosen : examples.slice(0, 4);
}

/** The empty canvas: how to start, and examples to start from. */
export function EmptyCanvas({
  examples,
  onLoadExample,
}: {
  examples: Example[];
  onLoadExample: (example: Example) => void;
}) {
  const shown = startExamples(examples);
  return (
    <Paper
      withBorder
      shadow="sm"
      p="lg"
      mt={{ base: 'md', sm: 60 }}
      // The panel is centred with left: 50%, which would halve its width on phones.
      w={{ base: 'calc(100vw - 48px)', xs: 'auto' }}
      maw={560}
      ta="center"
    >
      <Text size="sm" mb="sm">
        <b>Brick DNN</b> builds neural networks from bricks: drag layers together, see the shapes as
        you go and get clean PyTorch code.
        {DEMO
          ? ' This online demo runs in your browser.'
          : ' Prepare your data and train the network on this server, free to start.'}
      </Text>
      <Title order={4}>Start building</Title>
      <Text size="sm" c="dimmed" mt={6}>
        Drag an <b>Input</b> layer from the palette onto the canvas, then add layers by clicking
        them.
      </Text>
      {shown.length > 0 && (
        <>
          <Text size="sm" fw={600} mt="md" mb="xs">
            Or start from an example
          </Text>
          <SimpleGrid cols={{ base: 1, xs: 2 }} spacing="xs" ta="left">
            {shown.map((example) => (
              <UnstyledButton
                key={example.id}
                onClick={() => onLoadExample(example)}
                aria-label={`Start from the example ${example.name}`}
              >
                <Paper withBorder p="xs" radius="md" h="100%">
                  <Text size="sm" fw={600} style={{ display: 'flex', gap: 6 }}>
                    <IconSchool size={16} style={{ flexShrink: 0, marginTop: 2 }} />
                    {example.name}
                  </Text>
                  <Text size="xs" c="dimmed" lineClamp={2} mt={2}>
                    {example.description}
                  </Text>
                </Paper>
              </UnstyledButton>
            ))}
          </SimpleGrid>
        </>
      )}
      <LegalLinks mt="md" />
    </Paper>
  );
}
