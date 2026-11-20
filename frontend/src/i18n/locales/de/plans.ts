import type { plans as en } from '../en/plans';
import type { Messages } from '../types';

/** Tarife, Grenzen und Nutzung (dieselben Schlüssel wie `../en/plans`). */
export const plans: Messages<typeof en> = {
  features: {
    compute_one: '{{value}} CPU-Stunde pro Monat für Trainings und Vorhersagen',
    compute_other: '{{value}} CPU-Stunden pro Monat für Trainings und Vorhersagen',
    computeUnlimited: 'Unbegrenzte Rechenzeit für Trainings und Vorhersagen',
    storage: '{{value}} GB Speicher für Daten und trainierte Modelle',
    storageUnlimited: 'Unbegrenzter Speicher',
    perRunHours_one: 'Bis zu {{value}} CPU-Stunde pro Training',
    perRunHours_other: 'Bis zu {{value}} CPU-Stunden pro Training',
    perRunMinutes: 'Bis zu {{value}} CPU-Minuten pro Training',
    concurrent_one: 'Ein Training zur selben Zeit',
    concurrent_other: '{{count}} Trainings zur selben Zeit',
    priority: 'Startet vor Trainings günstigerer Tarife, wenn der Server ausgelastet ist',
    overage: 'Mehr Rechenzeit, wenn Sie sie einschalten: {{price}} pro CPU-Stunde',
    now: '(jetzt {{value}})',
  },
  short: {
    compute_one: '{{value}} CPU-Stunde pro Monat',
    compute_other: '{{value}} CPU-Stunden pro Monat',
    computeUnlimited: 'unbegrenzte Rechenzeit',
    storage: '{{value}} GB Speicher',
    storageUnlimited: 'unbegrenzten Speicher',
    perRunHours_one: 'bis zu {{value}} CPU-Stunde pro Training',
    perRunHours_other: 'bis zu {{value}} CPU-Stunden pro Training',
    perRunMinutes: 'bis zu {{value}} CPU-Minuten pro Training',
    concurrent_one: '{{count}} Training zur selben Zeit',
    concurrent_other: '{{count}} Trainings zur selben Zeit',
    perMonth: '{{price}} pro Monat',
    perYear: '{{price}} pro Jahr',
    noCharge: 'kostenlos',
    hours: '{{value}} CPU-Std.',
  },
  limit: {
    cpuTime:
      'Die Rechenzeit Ihres Kontos für diesen Monat ist aufgebraucht ({{used}} von {{limit}}). Am {{date}} steht sie wieder bereit.',
    cpuTimePlan:
      'Die Rechenzeit des Tarifs {{plan}} für diesen Monat ist aufgebraucht ({{used}} von {{limit}}). Am {{date}} steht sie wieder bereit.',
    reserved:
      'Der Rest der Rechenzeit dieses Monats ist für Ihre noch laufenden Trainings reserviert. Warten Sie, bis sie fertig sind, oder stoppen Sie eines.',
    overageCap:
      'Die Rechenzeit Ihres Tarifs und die zusätzlich erlaubte Rechenzeit sind aufgebraucht ({{used}} von {{limit}}). Ihre Grenze können Sie auf der Kontoseite erhöhen.',
    ownBudget:
      'Das Monatsbudget, das Sie für Ihr Konto gesetzt haben, ist aufgebraucht ({{used}} von {{limit}}). Sie können es auf der Kontoseite erhöhen.',
    runTime:
      'Dieses Training braucht voraussichtlich {{estimate}}; erlaubt sind höchstens {{limit}} pro Training.',
    runTimePlan:
      'Dieses Training braucht voraussichtlich {{estimate}}; im Tarif {{plan}} sind höchstens {{limit}} pro Training erlaubt.',
    runTimeStopped: 'Dieses Training hat {{limit}} erreicht, das Höchste für ein Training.',
    runTimeStoppedPlan:
      'Dieses Training hat {{limit}} erreicht, das Höchste für ein Training im Tarif {{plan}}.',
    leftThisMonth:
      'Dieses Training braucht voraussichtlich {{estimate}}, von der Rechenzeit dieses Monats sind aber nur noch {{limit}} übrig.',
    probablyStopped:
      'Es wird wahrscheinlich stoppen, bevor es fertig ist; das bis dahin Gelernte bleibt erhalten.',
    fewerEpochs: 'Trainieren Sie weniger Epochen, mit weniger Stichproben oder ein kleineres Netz.',
    storage:
      'Der Speicher Ihres Kontos ist voll: {{used}} von {{limit}}. Löschen Sie Daten oder Trainingsläufe, die Sie nicht mehr brauchen.',
    storagePlan:
      'Der Speicher des Tarifs {{plan}} ist voll: {{used}} von {{limit}}. Löschen Sie Daten oder Trainingsläufe, die Sie nicht mehr brauchen.',
    concurrent_one:
      'Ihr Konto trainiert ein Netz zur selben Zeit. Warten Sie, bis Ihr laufendes Training fertig ist, oder stoppen Sie es.',
    concurrent_other:
      'Ihr Konto trainiert {{count}} Netze zur selben Zeit. Warten Sie, bis eines fertig ist, oder stoppen Sie eines.',
    concurrentPlan_one:
      'Der Tarif {{plan}} trainiert ein Netz zur selben Zeit. Warten Sie, bis Ihr laufendes Training fertig ist, oder stoppen Sie es.',
    concurrentPlan_other:
      'Der Tarif {{plan}} trainiert {{count}} Netze zur selben Zeit. Warten Sie, bis eines fertig ist, oder stoppen Sie eines.',
    queued_one:
      'Ein Training eines Kontos kann in der Warteschlange warten. Warten Sie, bis Ihres gestartet ist, oder brechen Sie es ab.',
    queued_other:
      '{{count}} Trainings eines Kontos können in der Warteschlange warten. Warten Sie, bis eines gestartet ist, oder brechen Sie eines ab.',
    titles: {
      cpuTime: 'Die Rechenzeit dieses Monats ist aufgebraucht',
      runTime: 'Zu lang für ein Training',
      storage: 'Der Speicher ist voll',
      concurrent: 'Es läuft bereits ein Training',
      queued: 'Zu viele Trainings warten',
      budget: 'Nicht genug Rechenbudget',
    },
  },
  upgrade: {
    offer: 'Der Tarif {{name}} enthält {{includes}} für {{price}} pro Monat.',
    button: 'Zu {{name}} wechseln',
    newTab: '(öffnet in einem neuen Tab)',
    overage:
      'Sie können auch Rechenzeit über Ihren Tarif hinaus erlauben: {{price}} pro CPU-Stunde, bis zu einer Grenze, die Sie setzen.',
    overageButton: 'Zusätzliche Rechenzeit erlauben',
    overageTitle: 'Rechenzeit über Ihren Tarif hinaus',
    seePlans: 'Tarife ansehen',
    usageAndBudget: 'Nutzung & Budget',
  },
  usage: {
    label: 'Nutzung in diesem Monat',
    compute: '{{used}} von {{limit}}',
    storage: '{{used}} von {{limit}} Speicher',
    computeNearly: '{{percent}} % der Rechenzeit dieses Monats sind verbraucht.',
    computeFull:
      'Die Rechenzeit dieses Monats ist aufgebraucht; am {{date}} steht sie wieder bereit.',
    storageNearly: '{{percent}} % Ihres Speichers sind belegt.',
    storageFull: 'Ihr Speicher ist voll.',
    offer: '{{name}} enthält {{includes}} für {{price}} pro Monat.',
    fits_one: 'In den Rest passt etwa {{count}} Training wie dieses.',
    fits_other: 'In den Rest passen etwa {{count}} Trainings wie dieses.',
    menu: 'Tarif {{plan}} · {{usage}}',
    menuNoPlan: 'Rechenzeit diesen Monat: {{usage}}',
    trainings: 'Trainings {{value}}',
    predictions: 'Vorhersagen {{value}}',
    moreSources_one: '{{count}} weiteres Token oder eine weitere App',
    moreSources_other: '{{count}} weitere Token oder Apps',
    noLimit: '{{used}} · keine Grenze',
    menuLabel: 'Ihr Tarif und Ihre Nutzung',
  },
  estimate: {
    mayUseRunLimit: 'Es darf bis zu {{limit}} verwenden (die Grenze pro Training).',
    mayUseBudget: 'Es darf bis zu {{limit}} verwenden (der Rest Ihres Budgets).',
    left: 'Noch {{remaining}} in diesem Monat, bis {{date}}.',
  },
  pricing: {
    monthly: 'Monatlich',
    yearly: 'Jährlich',
    billing: 'Zahlungsweise',
    perYear: 'pro Jahr',
    perMonthEquivalent: '{{price}} pro Monat',
    yearlySaving: '{{percent}} % weniger als bei monatlicher Zahlung',
    yearlyOnlyMonthly: 'Nur monatlich',
    overage:
      'Ist die Rechenzeit eines bezahlten Tarifs aufgebraucht, stoppen Trainings, sofern Sie auf Ihrer Kontoseite nicht selbst mehr erlauben: abgerechnet pro CPU-Stunde zum Preis Ihres Tarifs, bis zu einer Grenze, die Sie setzen.',
    education:
      'Für Lehre und Forschung gibt es auf Anfrage einen Rabatt: <contact>schreiben Sie uns</contact>.',
    notSold: 'Hier werden keine Tarife verkauft',
    notSoldText:
      'Dieser Server verkauft keine Tarife. Netze bauen und Code erzeugen ist kostenlos; was Ihr Konto trainieren und speichern darf, steht auf Ihrer Kontoseite.',
    toAccount: 'Nutzung & Budget auf Ihrer Kontoseite',
    toEditor: 'Editor öffnen',
    failed: 'Die Tarife konnten nicht geladen werden',
    failedText: 'Der Server hat nicht geantwortet. Ihr Konto und Ihre Daten sind nicht betroffen.',
    retry: 'Erneut versuchen',
    periodPaid: 'Die Preise gelten pro Monat oder pro Jahr',
    planLabel: 'Tarif {{name}}',
  },
  gate: {
    create: 'Kostenloses Konto erstellen',
    logIn: 'Anmelden',
    toChoose: 'Kostenloses Konto erstellen, um {{name}} zu wählen',
  },
  footer: {
    support: 'Dieses Projekt unterstützen',
  },
};
