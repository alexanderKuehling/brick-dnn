import { useEffect, useRef, useState } from 'react';
import { IconAlertCircle, IconCheck, IconFlask } from '@tabler/icons-react';
import { Link, useLocation } from 'react-router-dom';
import {
  Alert,
  Anchor,
  Badge,
  Button,
  Center,
  Container,
  Group,
  List,
  Loader,
  Paper,
  SegmentedControl,
  SimpleGrid,
  Stack,
  Text,
  ThemeIcon,
  Title,
} from '@mantine/core';
import {
  api,
  type BillingEnabled,
  type BillingInterval,
  type BillingPlan,
  type BillingState,
  type Plan,
} from '@/api';
import { useAuth } from '@/auth/AuthContext';
import { EDITOR_PATH } from '@/components/RedirectToEditor';
import { i18n, Trans, useTranslation } from '@/i18n';
import { LegalLinks } from '@/legal/LegalLinks';
import { legalPages, useLegalInfo } from '@/legal/useLegalInfo';
import { ChangePlanDialog } from './ChangePlanDialog';
import { formatPrice, planAmount, planLines } from './format';
import { pricingPath } from './limits';
import { loadPaddle } from './paddle';
import { useBillingStatus } from './useBillingInfo';
import { useSubscribe } from './useSubscribe';

/** "17": how much less a year costs than twelve months, in percent (null: not less). */
export function yearlySaving(plan: Plan): number | null {
  if (!plan.price || !plan.price_per_year) {
    return null;
  }
  const saving = Math.round((1 - plan.price_per_year / (plan.price * 12)) * 100);
  return saving > 0 ? saving : null;
}

/** Buttons of a card: a long label ("Create a free account to choose Starter") wraps. */
const WRAP = {
  root: { height: 'auto', minHeight: 36, paddingBlock: 6 },
  label: { whiteSpace: 'normal', textAlign: 'center' },
} as const;

function PlanCard({
  plan,
  info,
  current,
  selected,
  interval,
  compareWith,
  priority,
  action,
}: {
  plan: BillingPlan;
  info: BillingEnabled;
  current: boolean;
  /** The plan a limit message led here for (/pricing?plan=…). */
  selected: boolean;
  interval: BillingInterval;
  /** The account's own plan: lines that differ from it say what it has now. */
  compareWith: Plan | null;
  /** The plan's trainings start before those of cheaper plans. */
  priority: boolean;
  action: React.ReactNode;
}) {
  useTranslation();
  const card = useRef<HTMLElement>(null);
  useEffect(() => {
    if (selected) {
      card.current?.scrollIntoView?.({ block: 'nearest' });
    }
  }, [selected]);
  const free = !plan.price;
  // A plan without a yearly price is shown with its monthly one.
  const yearly = interval === 'year' && Boolean(plan.price_per_year);
  const amount = planAmount(plan, yearly ? 'year' : 'month') ?? 0;
  const currency = plan.currency ?? info.currency;
  const saving = yearly ? yearlySaving(plan) : null;
  const now = new Map(
    (compareWith && !current ? planLines(compareWith) : []).map((line) => [line.key, line.amount])
  );
  return (
    <Paper
      component="section"
      ref={card}
      withBorder
      radius="md"
      p="lg"
      aria-label={i18n.t('plans.pricing.planLabel', { name: plan.name })}
      aria-current={selected ? 'true' : undefined}
      style={
        current || selected
          ? {
              borderColor: 'var(--mantine-primary-color-filled)',
              borderWidth: selected ? 2 : 1,
            }
          : undefined
      }
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
            {free ? i18n.t('ui.billing.PricingPage.free') : formatPrice(amount, currency)}
          </Text>
          {!free && (
            <Text span c="dimmed" size="sm">
              {' '}
              {yearly ? i18n.t('plans.pricing.perYear') : i18n.t('ui.billing.PricingPage.perMonth')}
            </Text>
          )}
          {yearly && (
            <Text size="xs" c="dimmed" mt={2}>
              {i18n.t('plans.pricing.perMonthEquivalent', {
                price: formatPrice(Math.round((amount / 12) * 100) / 100, currency),
              })}
              {saving !== null && ` · ${i18n.t('plans.pricing.yearlySaving', { percent: saving })}`}
            </Text>
          )}
          {interval === 'year' && !yearly && !free && (
            <Text size="xs" c="dimmed" mt={2}>
              {i18n.t('plans.pricing.yearlyOnlyMonthly')}
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
          {planLines(plan, { overage: info.overage_billed, priority }).map((line) => {
            const before = now.get(line.key);
            return (
              <List.Item key={line.key}>
                {line.text}
                {before && line.amount && before !== line.amount && (
                  <Text span c="dimmed" size="xs">
                    {' '}
                    {i18n.t('plans.features.now', { value: before })}
                  </Text>
                )}
              </List.Item>
            );
          })}
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
  const [chosen, setChosen] = useState<BillingInterval>('month');
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
  const selectedKey = new URLSearchParams(location.search).get('plan');
  // A subscription keeps the way it is paid: its plans are shown with those prices.
  const subscribed = Boolean(subscription && !state?.can_subscribe);
  const interval: BillingInterval = subscribed ? (subscription?.interval ?? 'month') : chosen;
  const lowest = Math.min(...info.plans.map((plan) => plan.priority ?? 0));

  const action = (plan: BillingPlan): React.ReactNode => {
    const selected = plan.key === selectedKey;
    if (!ready || (user && !state && !error)) {
      return <Button disabled>…</Button>;
    }
    if (!user) {
      // New here: an account first (it is free), then back to this plan.
      const back = { from: pricingPath(plan.purchasable ? plan.key : null) };
      return (
        <Stack gap={4}>
          <Button
            styles={WRAP}
            component={Link}
            to="/register"
            state={back}
            variant={plan.purchasable ? (selected ? 'filled' : 'light') : 'default'}
          >
            {plan.purchasable
              ? i18n.t('plans.gate.toChoose', { name: plan.name })
              : i18n.t('ui.billing.PricingPage.signUpForFree')}
          </Button>
          {plan.purchasable && (
            <Anchor component={Link} to="/login" state={back} size="xs" ta="center">
              {i18n.t('ui.billing.PricingPage.logInToChoose', { name: plan.name })}
            </Anchor>
          )}
        </Stack>
      );
    }
    if (!state || state.unlimited) {
      return null;
    }
    if (plan.key === currentKey) {
      return (
        <Button component={Link} to="/account" variant="default" styles={WRAP}>
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
      const yearly = interval === 'year' && Boolean(plan.price_per_year);
      return (
        <Button
          styles={WRAP}
          variant={selected ? 'filled' : 'light'}
          onClick={() => subscribe(plan, yearly ? 'year' : 'month')}
          loading={busy === plan.key || activating === plan.key}
          disabled={(busy !== null || activating !== null) && busy !== plan.key}
        >
          {i18n.t('ui.billing.PricingPage.choose', { name: plan.name })}
        </Button>
      );
    }
    if (subscription?.can_change_plan) {
      if (interval === 'year' && !plan.price_per_year) {
        return (
          <Text size="xs" c="dimmed">
            {i18n.t('plans.pricing.yearlyOnlyMonthly')}
          </Text>
        );
      }
      return (
        <Button
          styles={WRAP}
          variant={selected ? 'filled' : 'light'}
          onClick={() => setChanging(plan)}
        >
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
      {info.yearly && !subscribed && (
        <SegmentedControl
          mb="md"
          aria-label={i18n.t('plans.pricing.billing')}
          value={chosen}
          onChange={(value) => setChosen(value as BillingInterval)}
          data={[
            { value: 'month', label: i18n.t('plans.pricing.monthly') },
            { value: 'year', label: i18n.t('plans.pricing.yearly') },
          ]}
        />
      )}
      <SimpleGrid cols={{ base: 1, xs: 2, lg: info.plans.length > 3 ? 4 : info.plans.length }}>
        {info.plans.map((plan) => (
          <PlanCard
            key={plan.key}
            plan={plan}
            info={info}
            current={plan.key === currentKey}
            selected={plan.key === selectedKey && plan.key !== currentKey}
            interval={interval}
            compareWith={currentKey ? (state?.plan ?? null) : null}
            priority={(plan.priority ?? 0) > lowest}
            action={action(plan)}
          />
        ))}
      </SimpleGrid>
      {changing && (
        <ChangePlanDialog
          plan={changing}
          currentName={currentName}
          interval={interval}
          onClose={() => setChanging(null)}
          onChanged={setState}
        />
      )}
    </>
  );
}

/** The server did not answer: say so and offer to ask again (not "page not found"). */
function Failed({ retry }: { retry: () => void }) {
  useTranslation();
  return (
    <Container size="sm" py="xl">
      <Alert
        color="red"
        icon={<IconAlertCircle size={18} />}
        title={i18n.t('plans.pricing.failed')}
      >
        <Text size="sm">{i18n.t('plans.pricing.failedText')}</Text>
        <Button size="compact-sm" variant="light" color="red" mt="xs" onClick={retry}>
          {i18n.t('plans.pricing.retry')}
        </Button>
      </Alert>
    </Container>
  );
}

/** A server that sells no plans: the page exists (links lead here), with nothing to buy. */
function NotSold() {
  useTranslation();
  const { user } = useAuth();
  return (
    <Container size="sm" py="xl">
      <Title order={2}>{i18n.t('plans.pricing.notSold')}</Title>
      <Text size="sm" mt="sm" maw={560}>
        {i18n.t('plans.pricing.notSoldText')}
      </Text>
      <Group mt="md" gap="sm">
        {user && (
          <Button component={Link} to="/account" variant="light">
            {i18n.t('plans.pricing.toAccount')}
          </Button>
        )}
        <Button component={Link} to={EDITOR_PATH} variant="default">
          {i18n.t('plans.pricing.toEditor')}
        </Button>
      </Group>
      <LegalLinks mt="xl" />
    </Container>
  );
}

/**
 * /pricing: the plans of this server, what each includes and costs, and the way to buy one.
 * A server that sells none says so in a sentence, without prices.
 */
export function PricingPage() {
  useTranslation();
  const { info, failed, retry } = useBillingStatus();
  const legal = legalPages(useLegalInfo());
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
    return failed ? <Failed retry={retry} /> : <NotSold />;
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
        {overage && (
          <Text size="sm" maw={760}>
            {i18n.t('plans.pricing.overage')}
          </Text>
        )}
        <Text size="sm" c="dimmed" maw={760}>
          {info.yearly
            ? i18n.t('plans.pricing.periodPaid')
            : i18n.t('ui.billing.PricingPage.pricesArePerMonth')}
          {info.prices_include_tax && i18n.t('ui.billing.PricingPage.andIncludeVat')}
          {i18n.t('ui.billing.PricingPage.paddleChargesThemACpu')}
        </Text>
        {legal.includes('imprint') && (
          <Text size="sm" c="dimmed" maw={760}>
            <Trans
              i18nKey="plans.pricing.education"
              components={{ contact: <Anchor component={Link} to="/imprint" size="sm" /> }}
            />
          </Text>
        )}
        <Text size="sm" c="dimmed" maw={760}>
          {i18n.t('ui.billing.PricingPage.paidPlansAreSoldBy')}
        </Text>
      </Stack>
      <LegalLinks mt="xl" />
    </Container>
  );
}
