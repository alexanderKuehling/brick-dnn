/**
 * Die deutsche Preisseite (/de/preise/). Die Zahlen kommen aus site.ts (den Tarifen von
 * brickdnn.com), die Bedingungen aus den Nutzungsbedingungen und der Erstattungsregel
 * (deploy/sites/brickdnn.com/legal/) und aus docs/billing.md. Gebucht wird auf /pricing in der App.
 */
import { CONTACT_EMAIL, OVERAGE_PRICE, PLANS, type PlanFacts } from '../site.ts';
import type { PricingPageContent } from './types.ts';

function number(value: number): string {
  return String(value).replace('.', ',');
}

function euro(amount: number): string {
  return `${Number.isInteger(amount) ? amount : amount.toFixed(2).replace('.', ',')} €`;
}

function cpuHours(hours: number): string {
  return `${number(hours)} CPU-${hours === 1 ? 'Stunde' : 'Stunden'}`;
}

function cpuTime(minutes: number): string {
  return minutes >= 60 && minutes % 60 === 0 ? cpuHours(minutes / 60) : `${minutes} CPU-Minuten`;
}

const free = PLANS.find((plan) => plan.price === 0)!;
const paid = PLANS.filter((plan) => plan.price > 0);
const cheapest = paid[0];
const mail = `<a href="mailto:${CONTACT_EMAIL}">${CONTACT_EMAIL}</a>`;

export const pricingDe: PricingPageContent = {
  title: 'Preise von Brick DNN – Tarife für Training, Rechenzeit und Speicher',
  description: `Bauen und Code-Export sind kostenlos. Tarife ab ${euro(cheapest.price)} im Monat inkl. MwSt.: Rechenzeit, Speicher und Laufzeit je Training im Vergleich.`,
  h1: 'Preise und Tarife',
  lead: `Netze bauen, prüfen und als Code exportieren ist kostenlos und geht ohne Konto. Die Tarife unterscheiden sich darin, wie viel Rechenzeit der Server jeden Monat für Ihre Trainings und Vorhersagen aufwendet und wie viel er für Sie speichert. Alle Preise gelten pro Monat und enthalten die Umsatzsteuer.`,
  caption: 'Tarife auf brickdnn.com, Preise pro Monat inklusive Umsatzsteuer',
  columns: [
    'Tarif',
    'Preis pro Monat',
    'Rechenzeit pro Monat',
    'Speicher',
    'Längstes Training',
    'Trainings gleichzeitig',
  ],
  row: (plan: PlanFacts) => [
    plan.name,
    euro(plan.price),
    cpuHours(plan.cpuHoursPerMonth),
    `${number(plan.storageGb)} GB`,
    cpuTime(plan.cpuMinutesPerRun),
    String(plan.concurrentTrainings),
  ],
  buy: {
    title: 'Einen Tarif buchen',
    html: 'Gebucht wird auf der <a href="/pricing">Preisseite der App</a>: Melden Sie sich an, wählen Sie einen Tarif, und Paddle öffnet den Bezahlvorgang. Der Tarif gilt, sobald die Zahlung bestätigt ist, meist nach wenigen Sekunden. Unter <strong>Account</strong> sehen Sie danach Ihren Tarif, wann er sich verlängert, und können ihn wechseln oder kündigen.',
    button: 'Zur Preisseite der App',
  },
  sections: [
    {
      id: 'kostenlos',
      title: 'Was kostenlos ist',
      html: `<ul class="checks"><li>Netze und klassische Modelle bauen, die Formen prüfen lassen und den PyTorch- oder scikit-learn-Code exportieren: ohne Konto und ohne Grenze.</li><li>Der kostenlose Tarif ${free.name} für jedes Konto ohne Abo: ${cpuHours(free.cpuHoursPerMonth)} Rechenzeit im Monat, ${number(free.storageGb)} GB Speicher und bis zu ${cpuTime(free.cpuMinutesPerRun)} je Training. Das reicht etwa für ein U-Net auf den eingebauten Shapes-Daten oder ein kleines CNN auf MNIST. Eine Zahlungsmethode brauchen Sie dafür nicht.</li><li>Als Rechenzeit zählen nur Trainings und Vorhersagen (auch die Erklärungen eines trainierten Modells); Netzanalysen, Prüfungen und das Erzeugen von Code zählen nicht.</li></ul>`,
    },
    {
      id: 'rechenzeit',
      title: 'Rechenzeit',
      html: '<p>Eine CPU-Stunde ist eine Stunde eines Prozessorkerns, der an Ihren Trainings oder Vorhersagen arbeitet. Ein Training mit vier Threads verbraucht in einer Viertelstunde also eine CPU-Stunde. Das Kontingent gilt je Kalendermonat; nicht genutzte Stunden verfallen am Monatsende.</p><p>Vor jedem Start schätzt Brick DNN die Rechenzeit des geplanten Trainings und warnt, wenn sie knapp wird. Ist das Kontingent aufgebraucht, beginnt kein neues Training mehr, bis der nächste Monat beginnt, Sie den Tarif wechseln oder zusätzliche Rechenzeit einschalten.</p>',
    },
    {
      id: 'je-training',
      title: 'Grenze je Training',
      html: '<p>„Längstes Training“ ist die Rechenzeit, die ein einzelnes Training höchstens verbrauchen darf. Erreicht ein Training diese Grenze oder den Rest des Monatskontingents, hält es von selbst an und behält die Gewichte der besten Epoche bis dahin, wie bei einem Stopp von Hand. Ein klassisches Modell, das gerade angepasst wird, lässt sich nicht unterbrechen und wird dann nicht gespeichert. Die Spalte „Trainings gleichzeitig“ sagt, wie viele Trainings Ihres Kontos zugleich laufen dürfen; weitere warten in der Warteschlange.</p>',
    },
    {
      id: 'speicher',
      title: 'Speicher',
      html: '<p>Zum Speicher zählen importierte Tabellen im Bereich Daten, Trainingsdatensätze und die Dateien abgeschlossener Trainings (Gewichte und Protokoll). Netze selbst zählen nicht. Ist der Speicher voll, nimmt Brick DNN keine neuen Uploads an, bis Sie etwas löschen oder in einen größeren Tarif wechseln. Endet ein Abo, wird nichts gelöscht; solange mehr gespeichert ist, als der kostenlose Tarif vorsieht, können Sie nur nichts Neues hochladen.</p>',
    },
    {
      id: 'zusaetzlich',
      title: 'Zusätzliche Rechenzeit',
      html: `<p>In den kostenpflichtigen Tarifen können Sie Rechenzeit über das Kontingent hinaus für ${euro(OVERAGE_PRICE)} je CPU-Stunde nutzen. Das ist ausgeschaltet, bis Sie es unter <strong>Account</strong> selbst einschalten, auf Wunsch mit einer Obergrenze in CPU-Stunden. Abgerechnet wird nach Ablauf des Kalendermonats über Ihr Abo, bei einer Kündigung oder Kontolöschung sofort. Ohne den Schalter endet ein Training, wenn das Kontingent aufgebraucht ist.</p>`,
    },
    {
      id: 'umsatzsteuer',
      title: 'Umsatzsteuer, Zahlung und Rechnung',
      html: '<p>Alle Preise enthalten die Umsatzsteuer. Verkäufer der Tarife ist Paddle.com Market Ltd (Paddle) als Händler (Merchant of Record): Paddle zeigt den Bezahlvorgang, nimmt die Zahlung an, berechnet und führt die Umsatzsteuer Ihres Landes ab und stellt die Rechnung aus. Ihre Zahlungsdaten geben Sie nur bei Paddle ein; Brick DNN erfährt lediglich, welches Abo Ihr Konto hat. Zahlungsmethode und Rechnungen finden Sie im Kundenportal von Paddle, das <strong>Account</strong> verlinkt.</p>',
    },
    {
      id: 'lehre',
      title: 'Lehre und Forschung',
      html: `<p>Studierende bauen und exportieren Netze ohne Konto, und der kostenlose Tarif reicht für kleine Trainings. Für Lehrveranstaltungen und Forschungsgruppen gibt es auf Anfrage einen Rabatt: ${mail}.</p>`,
    },
  ],
  faq: {
    title: 'Häufige Fragen zur Abrechnung',
    items: [
      {
        question: 'Wie kündige ich?',
        html: 'Jederzeit zum Ende des bezahlten Monats, unter <strong>Account</strong> oder im Kundenportal von Paddle. Bis dahin behalten Sie den Tarif und können die Kündigung zurücknehmen. Danach wird nichts mehr abgebucht.',
      },
      {
        question: 'Kann ich den Tarif wechseln?',
        html: 'Ja, der Wechsel gilt sofort. Beim Wechsel in einen höheren Tarif wird der Unterschied für den Rest des bezahlten Monats sofort berechnet, beim Wechsel in einen niedrigeren Tarif mit der nächsten Rechnung verrechnet. Den Betrag sehen Sie vor dem Wechsel.',
      },
      {
        question: 'Bekomme ich mein Geld zurück, wenn der Tarif nicht passt?',
        html: `Innerhalb von 14 Tagen nach dem ersten Kauf eines Tarifs erhalten Sie auf Anfrage den vollen Betrag zurück, ohne Angabe von Gründen; das Abo endet dann sofort. Beantragen Sie die Erstattung bei Paddle (paddle.net, mit der E-Mail-Adresse des Kaufs) oder per E-Mail an ${mail}. Spätere Monate werden nicht anteilig erstattet. Ihr gesetzliches Widerrufsrecht bleibt unberührt. Die Einzelheiten stehen in der <a href="/refund">Erstattungsregel</a> und den <a href="/terms">Nutzungsbedingungen</a>.`,
      },
      {
        question: 'Was passiert mit meinen Daten, wenn das Abo endet?',
        html: 'Nichts wird gelöscht: Netze, Daten und Trainings bleiben erhalten, und es gelten wieder die Grenzen des kostenlosen Tarifs. Solange mehr gespeichert ist, als er vorsieht, können Sie nichts Neues hochladen.',
      },
      {
        question: 'Was passiert, wenn eine Zahlung fehlschlägt?',
        html: 'Paddle versucht es erneut. Sie behalten den Tarif währenddessen noch 7 Tage; danach gelten die Grenzen des kostenlosen Tarifs, bis die Zahlung eingegangen ist.',
      },
      {
        question: 'Brauche ich für den kostenlosen Tarif eine Kreditkarte?',
        html: 'Nein. Jedes Konto ohne Abo hat den kostenlosen Tarif, ohne Zahlungsmethode und ohne Laufzeit.',
      },
      {
        question: 'Wird außer dem Monatspreis noch etwas berechnet?',
        html: `Nur zusätzliche Rechenzeit, und nur, wenn Sie sie selbst einschalten (${euro(OVERAGE_PRICE)} je CPU-Stunde, bis zu Ihrer Obergrenze). Die Umsatzsteuer ist in allen Preisen enthalten.`,
      },
    ],
  },
};
