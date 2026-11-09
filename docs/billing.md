# Paid plans with Paddle

A server can sell its [plans](training.md#plans-and-limits) as monthly subscriptions. The
payments run through [Paddle Billing](https://www.paddle.com): Paddle is the *merchant of
record*, i.e. the seller. It shows the checkout, takes the payment, charges and pays the VAT of
the buyer's country, sends the invoice and handles refunds; the operator gets payouts from
Paddle. Brick DNN only learns which subscription an account has and gives it the matching plan.

**The whole feature is off unless it is configured**, and then the app looks and behaves as
without it: no prices, no pricing page, no request to Paddle, and every billing endpoint except
`GET /api/billing/` answers 404.

- [How it works](#how-it-works)
- [Settings](#settings)
- [Setting it up, step by step](#setting-it-up-step-by-step)
- [Running it](#running-it)
- [Overage](#overage-compute-beyond-the-allowance)
- [Privacy and legal texts](#privacy-and-legal-texts)

## How it works

1. **Plans** come from `BRICKDNN_PLANS` (limits and the prices that are *shown*). Each paid
   plan gets a *price* in Paddle's catalog; `BRICKDNN_PADDLE_PRICES` says which Paddle price
   belongs to which plan. The plan without a price there is the free plan
   (`BRICKDNN_DEFAULT_PLAN`). What a customer really pays is the price in Paddle: keep the two
   in line (`manage.py billing_sync --check` compares them).
2. **Checkout.** `/pricing` lists the plans. A logged-in user chooses one; the server creates
   a *customer* at Paddle with the account's e-mail address (once) and tells the browser the
   price and the customer. Only then does the browser load Paddle.js from `cdn.paddle.com` and
   open Paddle's overlay checkout. Payment data is entered there and never reaches this server.
3. **Notifications.** Paddle reports every change of a subscription to
   `/api/billing/paddle/webhook/`. The server checks the signature (HMAC-SHA256 with the
   secret of the notification destination, over the body as received, at most
   `BRICKDNN_PADDLE_WEBHOOK_TOLERANCE` seconds old), processes each event once, ignores events
   that are older than what it already knows, and sets the account's plan:

   | Paddle says                         | The account has                                        |
   | ----------------------------------- | ------------------------------------------------------ |
   | `active`, `trialing`                | the plan of the subscription's price                   |
   | a cancellation is scheduled         | the plan until Paddle reports the end of the period    |
   | `past_due` (a payment failed)       | the plan for `BRICKDNN_PADDLE_GRACE_DAYS`, then free   |
   | `paused`, `canceled`                | the free plan                                          |

   Nothing the browser says is trusted: a subscription only counts for the account whose
   Paddle customer it belongs to (the server created that customer), and the plan comes from
   the price Paddle reports. Nothing is deleted when a plan ends; the limits of the free plan
   apply (uploads are refused while more is stored than the free plan allows).
4. **The account page** ("Plan & billing") shows the plan and when it renews or ends, changes
   the plan (an upgrade is charged pro rata at once, a downgrade is credited to the next
   invoice; the dialog shows the amount Paddle previews), cancels to the end of the paid
   period, takes a cancellation back, and opens Paddle's customer portal for the payment
   method and the invoices.
5. **Once a day** the `cleanup` service runs `manage.py billing_sync`: it asks Paddle for
   every subscription again (in case a notification was lost), ends grace periods and bills
   [overage](#overage-compute-beyond-the-allowance).

Accounts marked **unlimited** (staff by default, or `manage.py set_plan <user> --unlimited`)
cannot buy a plan and are never changed by billing. A plan the operator sets by hand stays
until that account's own subscription changes. **Deleting an account** cancels its
subscription at Paddle at once; if Paddle cannot be reached, the deletion is refused (and can
be repeated), because otherwise somebody would go on paying for an account that no longer
exists. Accounts with a subscription are never deleted for inactivity.

## Settings

| Variable | Where | Meaning |
| --- | --- | --- |
| `BRICKDNN_PADDLE_ENV` | site settings or `.env` | `sandbox` (Paddle's test system) or `live` |
| `BRICKDNN_PADDLE_CLIENT_TOKEN` | site settings or `.env` | Client-side token for Paddle.js (`test_…` / `live_…`); public |
| `BRICKDNN_PADDLE_PRICES` | site settings or `.env` | `starter=pri_…,plus=pri_…,pro=pri_…`: the Paddle price of each paid plan. Several prices of one plan are separated by `\|`: the first is sold, the others stay valid for whoever subscribed to them (after a price change) |
| `BRICKDNN_PADDLE_OVERAGE_PRODUCT` | site settings or `.env` | Paddle product (`pro_…`) that overage is billed as; empty: overage is not offered |
| `BRICKDNN_PADDLE_API_KEY` | **`.env` only** | Server-side API key (secret) |
| `BRICKDNN_PADDLE_WEBHOOK_SECRET` | **`.env` only** | Secret of the notification destination (secret); two separated by a comma while rotating |
| `BRICKDNN_PADDLE_GRACE_DAYS` | optional, default `7` | Days an account keeps its plan after a failed payment |
| `BRICKDNN_PADDLE_WEBHOOK_TOLERANCE` | optional, default `300` | Seconds a notification's signature may be old (`0`: any age) |
| `BRICKDNN_PADDLE_OVERAGE_MIN_CHARGE` | optional, default `0.5` | Overage below this amount is not billed for a month |
| `BRICKDNN_BILLING_RATE`, `BRICKDNN_BILLING_WEBHOOK_RATE` | optional | Rate limits: `60/hour` per account for checkout and changes, `600/minute` per IP address for notifications |

"Site settings" is `deploy/sites/<domain>/settings.env` in the repository (public!), `.env`
is the file on the server. The two secrets are ignored if they appear in the site settings.
Billing is on only if the first three and the two secrets are set, every price belongs to an
existing plan and the token fits the environment; otherwise it stays off and
`manage.py check` (every start) and `manage.py billing_sync --check` say what is missing.

## Setting it up, step by step

Do everything in the **sandbox** first, then repeat it live: the sandbox is a separate Paddle
account with its own ids, tokens and keys, and no real money.

1. **Paddle account.** Sign up at paddle.com (live) and for a sandbox account (linked from
   Paddle's developer documentation). The live account needs identity and business
   verification and an approved website; that takes days, the sandbox works at once.
2. **Website approval** (live, *Checkout → Website approval*): request approval for the
   domain. Paddle looks at the site: it wants to see what is sold and for how much (the
   pricing page; switch billing on in the sandbox first so `/pricing` is there, or describe
   the plans on the start page), the terms of use, the privacy policy and a refund policy,
   all reachable without logging in. See [Privacy and legal texts](#privacy-and-legal-texts).
3. **Default payment link** (*Checkout → Checkout settings*): `https://<domain>/pricing`.
   Paddle sends customers there from its e-mails (e.g. to pay an open invoice); the pricing
   page then opens the checkout for that transaction.
4. **Products and prices** (*Catalog → Products*): one product per paid plan (e.g. "Brick DNN
   Starter"), each with one price: billed **monthly**, in the currency of
   `BRICKDNN_PRICE_CURRENCY`, with the amount of the plan in `BRICKDNN_PLANS` and tax
   **included in the price** (the app shows prices incl. VAT). Note the price ids (`pri_…`).
   For overage, one more product without a price (e.g. "Brick DNN extra compute"); note its
   id (`pro_…`).
5. **Client-side token** (*Developer tools → Authentication*): create one →
   `BRICKDNN_PADDLE_CLIENT_TOKEN`.
6. **API key** (same page): create one that may read and write customers, subscriptions and
   customer portal sessions and read products and prices → `BRICKDNN_PADDLE_API_KEY` in the
   server's `.env`.
7. **Notification destination** (*Developer tools → Notifications*): type *URL*,
   `https://<domain>/api/billing/paddle/webhook/`, with the events `subscription.created`,
   `subscription.activated`, `subscription.trialing`, `subscription.updated`,
   `subscription.past_due`, `subscription.paused`, `subscription.resumed`,
   `subscription.canceled`, `transaction.completed` and `transaction.payment_failed`. Its
   secret key → `BRICKDNN_PADDLE_WEBHOOK_SECRET` in the server's `.env`.
8. **Customer portal** (*Checkout → Customer portal*): leave it enabled; "Payment method and
   invoices" on the account page opens it.
9. **Settings.** Put `BRICKDNN_PADDLE_ENV`, `BRICKDNN_PADDLE_CLIENT_TOKEN`,
   `BRICKDNN_PADDLE_PRICES` and `BRICKDNN_PADDLE_OVERAGE_PRODUCT` into the site settings (the
   placeholders are there) or the `.env`, the two secrets into the `.env`, deploy, and check:

   ```bash
   docker compose exec backend python manage.py billing_sync --check
   ```

   It lists every price with its amount, currency, tax mode and interval as Paddle has them
   and says what does not fit the plans.
10. **Test in the sandbox:** buy a plan with Paddle's test card `4242 4242 4242 4242` (any
    future date, CVC `100`; Paddle's documentation lists more, e.g. cards that are declined
    or need 3-D Secure). Check that the account page shows the plan within a few seconds,
    change the plan, cancel, continue, open the customer portal, and look at
    *Billing events* in the Django admin. Paddle's *Developer tools → Simulations* can send
    events such as a failed payment.
11. **Go live:** repeat steps 3 to 8 in the live account and replace the values
    (`BRICKDNN_PADDLE_ENV=live`, the live token, prices, product, key and secret). Sandbox
    subscriptions of test accounts end with the next daily run. Have the legal texts checked
    before (below).

## Running it

- **Django admin:** *Billing customers* (the subscription of each account as Paddle last
  reported it), *Billing events* (every notification and what was done with it) and *Overage
  charges*, all read-only and linked to the user. Refunds, credits and cancelling for a
  customer are done in Paddle; the notification brings the result.
- **Logs** (`docker compose logs backend cleanup | grep Billing`): plan changes, refused
  notifications (wrong signature), events for customers no account has, charges.
- An event with the outcome *rejected* means that somebody paid in a way that does not belong
  to the account: a second subscription for an account that has one, or a checkout that named
  another account. Nothing was changed; look at it in Paddle and refund if needed.
- **Changing a price:** create a new price in Paddle and put it first:
  `plus=pri_new|pri_old`. New checkouts use the new one, existing subscribers keep theirs.
- **Notifications that did not arrive** are retried by Paddle; whatever is still missing is
  caught by the daily `billing_sync`, and the pricing page asks Paddle itself after a
  checkout when the notification takes longer than a few seconds.
- With Docker Compose, the `backend` and the `cleanup` service both need the Paddle settings
  **and the plans** (`BRICKDNN_PLANS`); a server set up with `deploy/server` reads them from
  the site settings.

## Overage: compute beyond the allowance

Plans can have `overage_price_per_cpu_hour`. Users of such a plan can switch overage on for
their account (with a cap of their own); the usage ledger then marks what goes beyond the
monthly allowance. With billing on, overage is only possible while it can be billed: an
overage product is configured and the account has a subscription that is paid and not
cancelled.

After a calendar month (UTC) ends, the daily run bills the month's overage to the
subscription as a one-time charge: one transaction, named e.g. "Brick DNN: 3.25 CPU hours
beyond the Plus plan in October 2026 (0.40 EUR per CPU hour)", tax included, at most up to the
account's cap. When a subscription is cancelled (on the account page) or the account is
deleted, what is open is billed right away, because no later chance comes.

Every charge is a row in *Overage charges*, saved as *pending* before Paddle is asked:

- *charged*: Paddle accepted it.
- *failed*: Paddle refused it (e.g. the card was declined). It is tried again with the next
  daily runs, five times in total.
- *pending*: the answer got lost (a timeout or a crash). It is **never billed again by
  itself**. Look in Paddle whether the subscription has a transaction of that amount, then
  use the admin action "Paddle did charge it" or "Paddle did not charge it (bill it again)".
- *skipped*: the subscription ended before it could be billed (e.g. cancelled in Paddle's
  portal at once). It is recorded so that you can see it; nothing is collected.

Paddle takes its fee per transaction, so very small charges cost more than they bring:
amounts below `BRICKDNN_PADDLE_OVERAGE_MIN_CHARGE` are not billed.

## Privacy and legal texts

What goes to Paddle and what this server keeps is described in
[privacy.md](privacy.md#paid-plans-paddle). The templates in `deploy/legal/` have a section on
Paddle for the privacy policy and one on paid plans for the terms of use. Before going live:

- Have the terms checked by a lawyer, in particular what the templates say about a
  "non-commercial" project, a service "free of charge" and liability: with paid plans the
  operator runs a business (in Germany: imprint with a postal address, trade registration and
  taxes are the operator's own matters; Paddle handles the VAT of the sales).
- Paddle asks for a **refund policy** on the site. As the seller, Paddle applies its own
  buyer terms and refund rules; the terms should say so and say what the operator grants on
  top, if anything.
- Paddle's fees are a percentage plus a fixed amount per transaction, which weighs on cheap
  plans, and Paddle may ask for separate terms for products below about 10 USD: clarify this
  with Paddle before offering a plan at 5 EUR.
