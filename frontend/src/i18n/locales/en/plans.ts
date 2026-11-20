/**
 * Plans, limits and usage: what a plan includes, the message at a limit with the plan that
 * lifts it, the usage indicator of the training panel, the pricing page and the buttons that
 * lead to an account.
 */
export const plans = {
  /** What a plan includes, one line each (pricing cards, the account page). */
  features: {
    compute_one: '{{value}} CPU hour per month for trainings and predictions',
    compute_other: '{{value}} CPU hours per month for trainings and predictions',
    computeUnlimited: 'Unlimited compute for trainings and predictions',
    storage: '{{value}} GB storage for data and trained models',
    storageUnlimited: 'Unlimited storage',
    perRunHours_one: 'Up to {{value}} CPU hour per training',
    perRunHours_other: 'Up to {{value}} CPU hours per training',
    perRunMinutes: 'Up to {{value}} CPU minutes per training',
    concurrent_one: 'One training at a time',
    concurrent_other: '{{count}} trainings at the same time',
    priority: 'Starts before trainings of cheaper plans when the server is busy',
    overage: 'More compute if you switch it on: {{price}} per CPU hour',
    now: '(now {{value}})',
  },
  /** The same in a few words, parts of one sentence ("10 CPU hours per month, 3 GB …"). */
  short: {
    compute_one: '{{value}} CPU hour per month',
    compute_other: '{{value}} CPU hours per month',
    computeUnlimited: 'unlimited compute',
    storage: '{{value}} GB of storage',
    storageUnlimited: 'unlimited storage',
    perRunHours_one: 'up to {{value}} CPU hour per training',
    perRunHours_other: 'up to {{value}} CPU hours per training',
    perRunMinutes: 'up to {{value}} CPU minutes per training',
    concurrent_one: '{{count}} training at a time',
    concurrent_other: '{{count}} trainings at a time',
    perMonth: '{{price}} per month',
    perYear: '{{price}} per year',
    noCharge: 'no charge',
    hours: '{{value}} CPU h',
  },
  /** The message where a limit stops the user. */
  limit: {
    cpuTime:
      'This month’s compute of your account is used up: {{used}} of {{limit}}. It is back on {{date}}.',
    cpuTimePlan:
      'This month’s compute of the {{plan}} plan is used up: {{used}} of {{limit}}. It is back on {{date}}.',
    reserved:
      'What is left of this month’s compute is reserved by your trainings that are still running. Wait until they finish, or stop one.',
    overageCap:
      'Your plan’s compute and the extra compute you allowed are used up: {{used}} of {{limit}}. You can raise your limit on the account page.',
    ownBudget:
      'The monthly budget you set for your account is used up: {{used}} of {{limit}}. You can raise it on the account page.',
    runTime: 'One training may use up to {{limit}}; this one needs about {{estimate}}.',
    runTimePlan:
      'With the {{plan}} plan one training may use up to {{limit}}; this one needs about {{estimate}}.',
    runTimeStopped: 'This training reached {{limit}}, the most one training may use.',
    runTimeStoppedPlan:
      'This training reached {{limit}}, the most one training may use with the {{plan}} plan.',
    leftThisMonth:
      '{{limit}} are left of this month’s compute; this training needs about {{estimate}}.',
    probablyStopped: 'It will probably stop before it finishes; what it learned by then is kept.',
    fewerEpochs: 'Train fewer epochs, on fewer samples or a smaller network.',
    storage:
      'The storage of your account is full: {{used}} of {{limit}}. Delete data or training runs you no longer need.',
    storagePlan:
      'The storage of the {{plan}} plan is full: {{used}} of {{limit}}. Delete data or training runs you no longer need.',
    concurrent_one:
      'Your account trains one network at a time. Wait until your running training finishes, or stop it.',
    concurrent_other:
      'Your account trains {{count}} networks at a time. Wait until one of them finishes, or stop one.',
    concurrentPlan_one:
      'The {{plan}} plan trains one network at a time. Wait until your running training finishes, or stop it.',
    concurrentPlan_other:
      'The {{plan}} plan trains {{count}} networks at a time. Wait until one of them finishes, or stop one.',
    queued_one:
      'One training of an account can wait in the queue. Wait until yours has started, or cancel it.',
    queued_other:
      '{{count}} trainings of an account can wait in the queue. Wait until one of yours has started, or cancel one.',
    titles: {
      cpuTime: 'This month’s compute is used up',
      runTime: 'Too long for one training',
      storage: 'The storage is full',
      concurrent: 'A training is already running',
      queued: 'Too many trainings are waiting',
      budget: 'Not enough compute budget',
    },
  },
  /** The plan that lifts a limit, and the way to it. */
  upgrade: {
    offer: 'The {{name}} plan includes {{includes}} for {{price}} per month.',
    button: 'Upgrade to {{name}}',
    newTab: '(opens in a new tab)',
    overage:
      'You can also allow compute beyond your plan at {{price}} per CPU hour, up to a limit you set.',
    overageButton: 'Allow extra compute',
    overageTitle: 'Compute beyond your plan',
    seePlans: 'See plans',
    usageAndBudget: 'Usage & budget',
  },
  /** The usage line where trainings are started, and in the account menu. */
  usage: {
    label: 'Usage this month',
    compute: '{{used}} of {{limit}}',
    storage: '{{used}} of {{limit}} storage',
    computeNearly: '{{percent}} % of this month’s compute is used.',
    computeFull: 'This month’s compute is used up; it is back on {{date}}.',
    storageNearly: '{{percent}} % of your storage is used.',
    storageFull: 'Your storage is full.',
    offer: '{{name}} includes {{includes}} for {{price}} per month.',
    fits_one: 'About {{count}} training like this one fits into what is left.',
    fits_other: 'About {{count}} trainings like this one fit into what is left.',
    menu: '{{plan}} plan · {{usage}}',
    menuNoPlan: 'Compute this month: {{usage}}',
    trainings: 'trainings {{value}}',
    predictions: 'predictions {{value}}',
    moreSources_one: '{{count}} more token or app',
    moreSources_other: '{{count}} more tokens or apps',
    noLimit: '{{used}} · no limit',
    menuLabel: 'Your plan and usage',
  },
  /** The training panel's estimate. */
  estimate: {
    mayUseRunLimit: 'It may use up to {{limit}} (the limit per training).',
    mayUseBudget: 'It may use up to {{limit}} (what is left of your budget).',
    left: '{{remaining}} left this month, until {{date}}.',
  },
  /** /pricing */
  pricing: {
    monthly: 'Monthly',
    yearly: 'Yearly',
    billing: 'How to pay',
    perYear: 'per year',
    perMonthEquivalent: '{{price}} per month',
    yearlySaving: '{{percent}} % less than paying monthly',
    yearlyOnlyMonthly: 'Only monthly',
    overage:
      'When the compute of a paid plan is used up, trainings stop unless you allow more yourself on your account page: billed per CPU hour at the price of your plan, up to a limit you set.',
    education:
      'Discounts for teaching and research are available on request: <contact>write to us</contact>.',
    notSold: 'No plans are sold here',
    notSoldText:
      'This server sells no plans. Building networks and generating code is free; what your account may train and store is shown on your account page.',
    toAccount: 'Usage & budget on your account page',
    toEditor: 'Open the editor',
    failed: 'The plans could not be loaded',
    failedText: 'The server did not answer. Your account and your data are not affected.',
    retry: 'Try again',
    periodPaid: 'Prices are per month or per year',
    planLabel: '{{name}} plan',
  },
  /** Where an account is needed. */
  gate: {
    create: 'Create free account',
    logIn: 'Log in',
    toChoose: 'Create a free account to choose {{name}}',
  },
  /** The footer row. */
  footer: {
    support: 'Support this project',
  },
};
