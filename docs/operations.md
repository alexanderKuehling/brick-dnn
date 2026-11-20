# Operations: is the service working as a business?

For the operator of a hosted Brick DNN. [monitoring.md](monitoring.md) says whether the site
is *up*; this page is about whether it is *used* and *paid for*: the business report, how each
number is counted, and what to do with it. Where users meet limits and how those lead to a
plan is in [billing.md](billing.md#where-limits-lead).

- [The business report](#the-business-report)
- [What the numbers mean](#what-the-numbers-mean)
- [What it cannot tell](#what-it-cannot-tell)
- [Reading it](#reading-it)

## The business report

```bash
docker compose exec backend python manage.py business_report            # the last 6 months
docker compose exec backend python manage.py business_report --period week   # 8 weeks
docker compose exec backend python manage.py business_report --period day --count 30
docker compose exec backend python manage.py business_report --json > report.json
```

The same is a page of the Django admin, **Business report** (`/admin/business-report/`, linked
on the admin's start page), with a switch between days, weeks and months. It is for staff
accounts only and follows the admin's rules, two-factor authentication included
(`BRICKDNN_REQUIRE_2FA_FOR_STAFF`).

**Nothing is tracked for it.** Every number is a count or a sum over rows the service keeps
anyway: accounts, training runs, the usage ledger, what Paddle reported about subscriptions,
share links. There are no page views, no events and no identifiers, and the report names no
account. It is computed when it is asked for; nothing is stored.

Periods are calendar days, weeks from Monday and calendar months in UTC, the running one
included (marked *running*: its numbers still grow).

## What the numbers mean

**Now**

| Line | Counted as |
| --- | --- |
| Accounts | Accounts of users. Staff members and superusers are left out everywhere in the account numbers |
| Paying accounts | Accounts whose subscription is `active` or `trialing` as Paddle last reported it, by plan, and their share of all accounts |
| Estimated MRR | The price of each paying account's plan from `BRICKDNN_PLANS` (a yearly subscription with a twelfth of `price_per_year`). VAT included, before Paddle's fees, without overage |
| Of those | How many of the paying ones cancelled to the end of their period, and how many more have a failed payment and keep their plan for the grace period |
| At their monthly allowance | Accounts that used up their plan's compute this month: the people who met the limit message |
| Stored by accounts | Imported files, datasets and the files of training runs, against the size and the free space of the data disk |
| Links that are on | Networks shared by link, and shared comparisons of runs |

**Per period**

| Column | Counted as |
| --- | --- |
| Sign-ups | Accounts created in the period |
| Activated | Of those, the ones that started their first training within 7 days of signing up (a run on this server or a row of the usage ledger, which outlives a deleted run). `*`: the 7 days are not over for all of them |
| Trained | Accounts that started a training in the period |
| New subs. / Ended subs. | Subscriptions whose first paid status, or whose end, Paddle reported in the period |
| Paying, MRR | At the end of the period, from the notifications up to then |
| Converted | Of the accounts created in the period, how many have paid since (free to paid, by sign-up month) |
| CPU h used | CPU hours of trainings and predictions counted in the period, by all accounts |
| CPU h capacity, Used | What the training workers could have delivered: slots × CPU threads of each worker for the time it ran. Without the training queue: trainings at a time × the CPU threads of the server, around the clock |
| Runs logged | Runs logged from scripts elsewhere ([experiments.md](experiments.md#runs-that-ran-elsewhere)) created in the period |
| Links shared | Comparisons of runs shared, and network links switched on, in the period |

With paid plans off (`manage.py billing_sync --check`), the subscription columns are left out.

## What it cannot tell

The report says these itself, under "How to read this":

- **Deleted accounts are gone from every period**, with their runs and usage, and unused runs
  are deleted after `BRICKDNN_DELETE_UNUSED_UPLOADS_DAYS`. Numbers of past periods can
  shrink. To keep a history, save `--json` once a month (a cron job on the server that writes
  it next to the backups is enough).
- **MRR is an estimate** from the prices in the settings, not what Paddle paid out. Paddle's
  dashboard has the money; this has the trend.
- The history of subscriptions is read from Paddle's notifications as they were applied
  (*Billing events* in the admin). A notification that never arrived is missing there, while
  "now" is corrected by the daily `billing_sync`: the two can differ by such a subscription.
  Subscriptions from before this was recorded (and test subscriptions of the sandbox) are not
  in the history.
- CPU time is counted in the period a training started counting, not spread over the days it
  ran; the capacity of periods older than 90 days is unknown with the training queue (the
  rows of old workers are deleted) and shown as `-`.
- A network link that was on before the date was recorded has none: it counts under "now"
  only.
- There are no visits, no "opened the editor" and no "saw the pricing page": the service does
  not track those.

## Reading it

- **Sign-ups without activation**: people create accounts and do not train. Look at what the
  first training needs (the training panel after signing up, the built-in datasets).
- **Activated but nobody at their allowance**: the free plan is generous for what people do;
  nobody has a reason to pay yet.
- **Many at their allowance, few converted**: the limit is met and the offer does not
  convince. The plan a limit message names is the cheapest one that lifts the limit
  ([billing.md](billing.md#where-limits-lead)): check that its price and contents fit.
- **CPU hours used near the capacity**: trainings wait (see *Training workers* in the admin for
  waiting times). More worker slots or a larger server ([scaling.md](scaling.md)).
- **Storage near the disk**: lower `BRICKDNN_DELETE_UNUSED_UPLOADS_DAYS`, or a larger disk.
- **Ended subscriptions rising with flat sign-ups**: ask the people who left; the report
  cannot.
