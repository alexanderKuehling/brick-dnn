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
import { i18n, useTranslation } from '@/i18n';
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
  useTranslation();
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
          {current && <Badge variant="light">{i18n.t('ui.billing.PricingPage.yourPlan')}</Badge>}
        </Group>
        <div>
          <Text span fz={30} fw={700} lh={1.1}>
            {free
              ? i18n.t('ui.billing.PricingPage.free')
              : formatPrice(plan.price ?? 0, plan.currency ?? info.currency)}
          </Text>
          {!free && (
            <Text span c="dimmed" size="sm">
              {' '}
              {i18n.t('ui.billing.PricingPage.perMonth')}
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
  useTranslation();
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
          {i18n.t('ui.billing.PricingPage.logInToChoose', { name: plan.name })}
        </Button>
      ) : (
        <Button component={Link} to="/register" variant="default">
          {i18n.t('ui.billing.PricingPage.signUpForFree')}
        </Button>
      );
    }
    if (!state || state.unlimited) {
      return null;
    }
    if (plan.key === currentKey) {
      return (
        <Button component={Link} to="/account" variant="default">
          {i18n.t('ui.billing.PricingPage.manageOnYourAccountPage')}
        </Button>
      );
    }
    if (!plan.purchasable) {
      // The free plan: where a paid one goes back to.
      return (
        <Text size="xs" c="dimmed">
          {subscription?.can_cancel
            ? i18n.t('ui.billing.PricingPage.cancelYourSubscriptionOnYour')
            : i18n.t('ui.billing.PricingPage.thePlanOfEveryAccount')}
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
          {i18n.t('ui.billing.PricingPage.choose', { name: plan.name })}
        </Button>
      );
    }
    if (subscription?.can_change_plan) {
      return (
        <Button variant="light" onClick={() => setChanging(plan)}>
          {i18n.t('ui.billing.PricingPage.switchTo', { name: plan.name })}
        </Button>
      );
    }
    return (
      <Text size="xs" c="dimmed">
        {subscription?.can_reactivate
          ? i18n.t('ui.billing.PricingPage.continueYourSubscriptionOnYour')
          : i18n.t('ui.billing.PricingPage.updateYourPaymentOnYour')}
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
          {i18n.t('ui.billing.PricingPage.noLimitsApplyToYour')}
        </Alert>
      )}
      {activating && (
        <Alert color="blue" mb="md" icon={<Loader size={16} />}>
          {i18n.t('ui.billing.PricingPage.paymentReceivedActivatingYourPlan')}
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
  useTranslation();
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
      <Title order={2}>{i18n.t('ui.billing.PricingPage.plansAndPrices')}</Title>
      <Text c="dimmed" size="sm" mt={4} maw={640}>
        {i18n.t('ui.billing.PricingPage.buildingNetworksAndGeneratingCode')}
      </Text>
      {info.environment === 'sandbox' && (
        <Alert
          color="yellow"
          mt="md"
          icon={<IconFlask size={18} />}
          title={i18n.t('ui.billing.PricingPage.testMode')}
        >
          {i18n.t('ui.billing.PricingPage.paymentsAreSimulatedWithPaddle')}
        </Alert>
      )}
      <Stack gap="lg" mt="xl">
        <Plans info={info} />
        <Text size="sm" c="dimmed" maw={760}>
          {i18n.t('ui.billing.PricingPage.pricesArePerMonth')}
          {info.prices_include_tax && i18n.t('ui.billing.PricingPage.andIncludeVat')}
          {i18n.t('ui.billing.PricingPage.paddleChargesThemACpu')}
          {overage && i18n.t('ui.billing.PricingPage.computeBeyondWhatAPlan')}
        </Text>
        <Text size="sm" c="dimmed" maw={760}>
          {i18n.t('ui.billing.PricingPage.paidPlansAreSoldBy')}
        </Text>
      </Stack>
      <LegalLinks mt="xl" />
    </Container>
  );
}
