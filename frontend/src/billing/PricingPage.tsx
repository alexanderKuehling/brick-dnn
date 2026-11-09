import { useEffect, useState } from 'react';
import { IconCheck, IconFlask } from '@tabler/icons-react';
import { Link, useLocation } from 'react-router-dom';
import {
  Alert,
  Badge,
  Button,
  Center,
  Container,
  Group,
  List,
  Loader,
  Paper,
  SimpleGrid,
  Stack,
  Text,
  ThemeIcon,
  Title,
} from '@mantine/core';
import { api, type BillingEnabled, type BillingPlan, type BillingState } from '@/api';
import { useAuth } from '@/auth/AuthContext';
import { LegalLinks } from '@/legal/LegalLinks';
import { NotFoundPage } from '@/pages/NotFoundPage';
import { ChangePlanDialog } from './ChangePlanDialog';
import { formatPrice, planFeatures } from './format';
import { loadPaddle } from './paddle';
import { useBillingInfo } from './useBillingInfo';
import { useSubscribe } from './useSubscribe';

function PlanCard({
  plan,
  info,
  current,
  action,
}: {
  plan: BillingPlan;
  info: BillingEnabled;
  current: boolean;
  action: React.ReactNode;
}) {
  const free = !plan.price;
  return (
    <Paper
      component="section"
      withBorder
      radius="md"
      p="lg"
      aria-label={`${plan.name} plan`}
      style={current ? { borderColor: 'var(--mantine-primary-color-filled)' } : undefined}
    >
      <Stack gap="sm" h="100%">
        <Group justify="space-between" align="center" gap="xs" wrap="nowrap">
          <Title order={3} size="h4">
            {plan.name}
          </Title>
          {current && <Badge variant="light">Your plan</Badge>}
        </Group>
        <div>
          <Text span fz={30} fw={700} lh={1.1}>
            {free ? 'Free' : formatPrice(plan.price ?? 0, plan.currency ?? info.currency)}
          </Text>
          {!free && (
            <Text span c="dimmed" size="sm">
              {' '}
              per month
            </Text>
          )}
        </div>
        <List
          size="sm"
          spacing={6}
          style={{ flex: 1 }}
          icon={
            <ThemeIcon size={18} radius="xl" variant="light">
              <IconCheck size={12} />
            </ThemeIcon>
          }
        >
          {planFeatures(plan, info.overage_billed).map((line) => (
            <List.Item key={line}>{line}</List.Item>
          ))}
        </List>
        {action}
      </Stack>
    </Paper>
  );
}

function Plans({ info }: { info: BillingEnabled }) {
  const { user, ready } = useAuth();
  const location = useLocation();
  const [state, setState] = useState<BillingState | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [changing, setChanging] = useState<BillingPlan | null>(null);
  const { subscribe, busy, activating } = useSubscribe(info, setState);

  useEffect(() => {
    if (!user) {
      setState(null);
      return undefined;
    }
    const controller = new AbortController();
    api.billing
      .state(controller.signal)
      .then(setState)
      .catch((err: Error) => !controller.signal.aborted && setError(err.message));
    return () => controller.abort();
  }, [user]);

  const subscription = state?.subscription ?? null;
  const currentKey = state && !state.unlimited ? state.plan.key : null;
  const currentName = subscription?.plan_name ?? state?.plan.name ?? '';

  const action = (plan: BillingPlan): React.ReactNode => {
    if (!ready || (user && !state && !error)) {
      return <Button disabled>…</Button>;
    }
    if (!user) {
      return plan.purchasable ? (
        <Button component={Link} to="/login" state={{ from: location.pathname }} variant="light">
          Log in to choose {plan.name}
        </Button>
      ) : (
        <Button component={Link} to="/register" variant="default">
          Sign up for free
        </Button>
      );
    }
    if (!state || state.unlimited) {
      return null;
    }
    if (plan.key === currentKey) {
      return (
        <Button component={Link} to="/account" variant="default">
          Manage on your account page
        </Button>
      );
    }
    if (!plan.purchasable) {
      // The free plan: where a paid one goes back to.
      return (
        <Text size="xs" c="dimmed">
          {subscription?.can_cancel
            ? 'Cancel your subscription on your account page to go back to this plan.'
            : 'The plan of every account without a subscription.'}
        </Text>
      );
    }
    if (state.can_subscribe) {
      return (
        <Button
          onClick={() => subscribe(plan)}
          loading={busy === plan.key || activating === plan.key}
          disabled={(busy !== null || activating !== null) && busy !== plan.key}
        >
          Choose {plan.name}
        </Button>
      );
    }
    if (subscription?.can_change_plan) {
      return (
        <Button variant="light" onClick={() => setChanging(plan)}>
          Switch to {plan.name}
        </Button>
      );
    }
    return (
      <Text size="xs" c="dimmed">
        {subscription?.can_reactivate
          ? 'Continue your subscription on your account page to change its plan.'
          : 'Update your payment on your account page to change the plan.'}
      </Text>
    );
  };

  return (
    <>
      {error && (
        <Alert color="red" mb="md">
          {error}
        </Alert>
      )}
      {state?.unlimited && (
        <Alert color="green" mb="md">
          No limits apply to your account: you do not need a plan.
        </Alert>
      )}
      {activating && (
        <Alert color="blue" mb="md" icon={<Loader size={16} />}>
          Payment received. Activating your plan…
        </Alert>
      )}
      <SimpleGrid cols={{ base: 1, xs: 2, lg: info.plans.length > 3 ? 4 : info.plans.length }}>
        {info.plans.map((plan) => (
          <PlanCard
            key={plan.key}
            plan={plan}
            info={info}
            current={plan.key === currentKey}
            action={action(plan)}
          />
        ))}
      </SimpleGrid>
      {changing && (
        <ChangePlanDialog
          plan={changing}
          currentName={currentName}
          onClose={() => setChanging(null)}
          onChanged={setState}
        />
      )}
    </>
  );
}

/** /pricing: the plans of this server and what they cost. Only on a server that sells plans. */
export function PricingPage() {
  const info = useBillingInfo();
  const { search } = useLocation();
  useEffect(() => {
    // Paddle's e-mails (e.g. to pay an open invoice) link here with ?_ptxn=<transaction>:
    // Paddle.js opens the checkout for it by itself once it is loaded.
    if (info && new URLSearchParams(search).has('_ptxn')) {
      loadPaddle(info).catch(() => undefined);
    }
  }, [info, search]);
  if (info === undefined) {
    return (
      <Center h={300}>
        <Loader />
      </Center>
    );
  }
  if (info === null) {
    return <NotFoundPage />;
  }
  const overage = info.overage_billed && info.plans.some((p) => p.overage_price_per_cpu_hour);
  return (
    <Container size="lg" py="xl">
      <Title order={2}>Plans and prices</Title>
      <Text c="dimmed" size="sm" mt={4} maw={640}>
        Building networks and generating code is free. Plans differ in how much the server trains
        for you each month and how much it stores.
      </Text>
      {info.environment === 'sandbox' && (
        <Alert color="yellow" mt="md" icon={<IconFlask size={18} />} title="Test mode">
          Payments are simulated with Paddle’s sandbox: nothing is charged.
        </Alert>
      )}
      <Stack gap="lg" mt="xl">
        <Plans info={info} />
        <Text size="sm" c="dimmed" maw={760}>
          Prices are per month{info.prices_include_tax && ' and include VAT'}; Paddle charges them.
          A CPU hour is one hour of one processor core working on your trainings or predictions;
          unused hours do not carry over. You can cancel at any time to the end of the month you
          paid for. Nothing is deleted when a plan ends: the limits of the free plan apply.
          {overage &&
            ' Compute beyond what a plan includes is only used and billed if you switch it on yourself on your account page, up to a limit you set.'}
        </Text>
        <Text size="sm" c="dimmed" maw={760}>
          Paid plans are sold by Paddle (Paddle.com Market Ltd), which handles the payment, the
          invoice and the tax as the merchant of record.
        </Text>
      </Stack>
      <LegalLinks mt="xl" />
    </Container>
  );
}
