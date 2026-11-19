/**
 * Die Seiten über Brick DNN selbst, auf Deutsch: für Lehrende (/de/lehre/), wer dahintersteht
 * (/de/ueber/) und Sicherheit (/de/sicherheit/). Dieselben Aussagen wie in info.en.ts, jede
 * durch das Repository gedeckt (site.ts, deploy/sites/brickdnn.com/legal/, SECURITY.md, docs/).
 */
import {
  absolute,
  BADGE_PATHS,
  CHANGELOG_URL,
  CONTACT_EMAIL,
  DATA_PROTECTION_PATHS,
  EDITOR_PATH,
  editorExampleUrl,
  EXAMPLES_PATHS,
  OUTAGES_URL,
  PLANS,
  PRICING_DE_PATH,
  REPOSITORY_URL,
  SECURITY_ADVISORY_URL,
  SECURITY_PATHS,
  SECURITY_TXT_PATH,
  TEACHING_PATHS,
  UPTIME_URL,
} from '../site.ts';
import { CITATION_BIBTEX, CITATION_TEXT } from './info.en.ts';
import type { InfoPage } from './types.ts';

const free = PLANS.find((plan) => plan.price === 0)!;
const mail = `<a href="mailto:${CONTACT_EMAIL}">${CONTACT_EMAIL}</a>`;
const number = (value: number) => String(value).replace('.', ',');

const BADGE_MARKDOWN = `[![In Brick DNN öffnen](${absolute(BADGE_PATHS.de)})](${absolute(editorExampleUrl('unet'))})`;

const teaching: InfoPage = {
  id: 'teaching',
  title: 'Brick DNN in der Lehre – Deep Learning in Vorlesung, Übung und Praktikum',
  description:
    'Für Lehrende: Studierende bauen Netze aus Schichten und exportieren PyTorch-Code ohne Konto. Geteilte Netze für Aufgaben, eingebaute Datensätze, eigene Instanz.',
  kicker: 'Für Lehrende',
  h1: 'Brick DNN in einer Deep-Learning-Lehrveranstaltung',
  lead: 'Studierende verbinden Schichten, sehen die Form jedes Tensors und nehmen den PyTorch-Code mit. Ohne Installation, und zum Bauen und Exportieren ohne Konto.',
  action: { label: 'Editor öffnen', href: EDITOR_PATH },
  sections: [
    {
      id: 'kurs',
      title: 'Worauf sich ein Kurs verlassen kann',
      html: `<ul class="grid two">
<li><h3>Kein Konto für Studierende</h3><p>Netze bauen, prüfen und den Code exportieren geht ohne Konto. Eine Übung kann also stattfinden, ohne dass sich jemand registriert. Für das Training auf dem Server braucht es ein kostenloses Konto (${number(free.cpuHoursPerMonth)} CPU-Stunden im Monat).</p></li>
<li><h3>Aufgaben als Link</h3><p>Teilen Sie ein Netz per Link als Ausgangspunkt einer Aufgabe. Wer ihn öffnet, sieht die Schichten und den Code und kann eine Kopie im Editor öffnen. Vergleiche von Trainingsläufen lassen sich genauso teilen, nur zum Ansehen, etwa für einen Bericht oder die Besprechung im Kurs.</p></li>
<li><h3>Code, der überall läuft</h3><p>Jedes Netz wird zu einer PyTorch-Datei, klassische Modelle zu einem scikit-learn-Skript. Der Code importiert nichts aus Brick DNN und läuft im Notebook, auf dem Laptop oder auf dem Cluster des Instituts.</p></li>
<li><h3>Eingebaute Datensätze</h3><p>MNIST, Fashion-MNIST und CIFAR-10 für Bilder, <em>Sentiment</em>, <em>Key lookup</em> und <em>Topics</em> für Text, <em>Shapes</em> und <em>Cells</em> für die Segmentierung. Die erste Übung kommt ohne Upload aus.</p></li>
<li><h3>Eine eigene Instanz</h3><p>Brick DNN ist Open Source unter der Apache-Lizenz 2.0 und läuft mit Docker Compose. Ein Institut kann es selbst betreiben, die öffentliche Registrierung abschalten und auf der eigenen GPU trainieren.</p></li>
<li><h3>Daten in der Schweiz</h3><p>brickdnn.com läuft auf einem Server in Genf, ohne Cookies und ohne Tracking. Das <a href="${DATA_PROTECTION_PATHS.de}">Faktenblatt für Datenschutzbeauftragte</a> sagt, was in einem Kurs verarbeitet wird; die <a href="/accessibility">Erklärung zur Barrierefreiheit</a> sagt, was mit Tastatur und Screenreader funktioniert.</p></li>
</ul>`,
    },
    {
      id: 'badge',
      title: 'Aus dem Kursmaterial verlinken',
      html: `<div class="prose"><p>Dieses Badge öffnet eines der <a href="${EXAMPLES_PATHS.de}">eingebauten Beispiele</a> im Editor. In einem README oder Notebook:</p>
<p><img src="${BADGE_PATHS.de}" width="172" height="28" alt="Das Badge: eine Schaltfläche mit dem Logo von Brick DNN und den Worten „In Brick DNN öffnen“" loading="lazy"></p>
<pre class="snippet" tabindex="0" aria-label="Markdown des Badges"><code>${BADGE_MARKDOWN}</code></pre>
<p>Für ein eigenes Netz schalten Sie im Editor unter <strong>Teilen</strong> den Link ein. Der Dialog enthält das Badge für diesen Link, als Markdown und als HTML.</p></div>`,
    },
    {
      id: 'rabatt',
      title: 'Tarife für Lehrveranstaltungen',
      html: `<div class="prose"><p>Bauen und Exportieren kosten nichts, und der kostenlose Tarif reicht für kleine Trainings. Für größere gibt es <a href="${PRICING_DE_PATH}">monatliche Tarife</a>.</p></div>
<p class="discount">Für Lehre und Forschung gibt es auf Anfrage einen Rabatt: ${mail}.</p>`,
    },
    {
      id: 'zitieren',
      title: 'Brick DNN zitieren',
      html: `<div class="prose"><p>Wenn Brick DNN Teil von Kursmaterial oder einer Veröffentlichung ist, zitieren Sie bitte die Software:</p>
<p lang="en">${CITATION_TEXT}</p>
<pre class="snippet" tabindex="0" aria-label="Die Quellenangabe als BibTeX"><code>${CITATION_BIBTEX}</code></pre>
<p>Dieselben Angaben stehen in der Datei <a href="${REPOSITORY_URL}/blob/main/CITATION.cff">CITATION.cff</a> des Repositorys.</p></div>`,
    },
  ],
};

const about: InfoPage = {
  id: 'about',
  title: 'Über Brick DNN – wer es baut, wo die Daten liegen, wie Sie Hilfe bekommen',
  description:
    'Brick DNN wird von einer Person gebaut und betrieben, einem unabhängigen Entwickler in Deutschland. Open Source, Daten in der Schweiz, Support per E-Mail.',
  kicker: 'Über Brick DNN',
  h1: 'Wer hinter Brick DNN steht',
  lead: 'Brick DNN wird von einer Person gebaut und betrieben. Ich bin Alexander Kühling, unabhängiger Entwickler in Deutschland.',
  sections: [
    {
      id: 'kurz',
      title: 'In Kürze',
      html: `<dl class="glance">
<div><dt>Eine Person</dt><dd>Ich schreibe die Software, betreibe den Server und beantworte die E-Mails selbst. Die rechtlichen Angaben stehen im <a href="/imprint">Impressum</a>.</dd></div>
<div><dt>Open Source</dt><dd>Die gesamte Anwendung liegt auf <a href="${REPOSITORY_URL}">GitHub</a> und steht unter der <a href="${REPOSITORY_URL}/blob/main/LICENSE">Apache-Lizenz 2.0</a>. Das <a href="${CHANGELOG_URL}">Änderungsprotokoll</a> (auf Englisch) zeigt, was sich wann geändert hat.</dd></div>
<div><dt>Wo die Daten liegen</dt><dd>Auf einem Server von Infomaniak in Genf, Schweiz: die Anwendung, die Datenbank, hochgeladene Dateien und Trainingsergebnisse. Keine Cookies, kein Tracking. Einzelheiten stehen in der <a href="/privacy">Datenschutzerklärung</a> und auf der Seite <a href="${SECURITY_PATHS.de}">Sicherheit</a>.</dd></div>
<div><dt>Support</dt><dd>Schreiben Sie an ${mail}. Fehler und Wünsche können Sie auch im <a href="${REPOSITORY_URL}/issues">Issue-Tracker</a> auf GitHub melden.</dd></div>
<div><dt>Kostenpflichtige Tarife</dt><dd>Monatlich, kündbar zum Ende des bezahlten Monats. Innerhalb von 14 Tagen nach dem ersten Kauf eines Tarifs erhalten Sie auf Anfrage den vollen Betrag zurück, ohne Angabe von Gründen. Die Bedingungen stehen in der <a href="/refund">Erstattungsregel</a> und den <a href="/terms">Nutzungsbedingungen</a>.</dd></div>
</dl>`,
    },
    {
      id: 'fortbestand',
      title: 'Wenn der Dienst endet',
      html: `<div class="prose"><p>Bei einem Dienst, den eine Person betreibt, ist das eine berechtigte Frage. Was Sie hier erstellen, hängt nicht an brickdnn.com: Exportierte Skripte importieren nichts aus Brick DNN, und die Anwendung ist Open Source. Sollte der gehostete Dienst einmal enden müssen, sage ich Folgendes zu:</p>
<ul class="checks">
<li><strong>90 Tage Vorlauf.</strong> Mindestens 90 Tage vor dem Ende kündige ich es an, per E-Mail an jedes Konto und auf dieser Website.</li>
<li><strong>Exporte bis zum letzten Tag.</strong> In diesen 90 Tagen funktioniert alles weiter: Netze, Daten, Ergebnisse, trainierte Gewichte und Skripte lassen sich wie bisher herunterladen, und <em>Meine Daten herunterladen</em> liefert alles als ZIP-Datei.</li>
<li><strong>Geld zurück für ungenutzte Zeit.</strong> Was Sie für die Zeit nach dem Ende bezahlt haben, erstatte ich anteilig.</li>
<li><strong>Der Code bleibt.</strong> Der Quellcode bleibt unter der Apache-Lizenz 2.0 verfügbar; die letzte Version läuft also auf einem eigenen Server.</li>
</ul></div>`,
    },
  ],
};

const security: InfoPage = {
  id: 'security',
  title: 'Sicherheit bei Brick DNN – Anmeldung, Backups, Schwachstellen melden',
  description:
    'Wie brickdnn.com Konten und Daten schützt: verschlüsselte Verbindungen, Zwei-Faktor-Anmeldung, stündliche Datenbank-Backups mit Kopie in der Schweiz.',
  kicker: 'Sicherheit',
  h1: 'Sicherheit',
  lead: 'Was Konten und Daten auf brickdnn.com schützt und wie Sie ein Problem melden.',
  sections: [
    {
      id: 'massnahmen',
      title: 'Was eingerichtet ist',
      html: `<dl class="glance">
<div><dt>Verbindungen</dt><dd>Jede Verbindung zu brickdnn.com ist verschlüsselt (HTTPS), und Browser werden angewiesen, nichts anderes zu verwenden (HSTS).</dd></div>
<div><dt>Anmeldung</dt><dd>Passwörter werden nur als Hash mit Salt gespeichert. Jedes Konto kann die Zwei-Faktor-Anmeldung mit einer Authenticator-App und Wiederherstellungscodes einschalten.</dd></div>
<div><dt>Tokens</dt><dd>API-Tokens für Skripte und KI-Assistenten werden als Hash gespeichert und nur einmal angezeigt. Ein Token lässt sich auf das Lesen beschränken, und die Aktivitätsseite listet auf, was jedes Token und jede verbundene App getan hat.</dd></div>
<div><dt>Backups</dt><dd>Die Datenbank wird stündlich und täglich gesichert. Die Sicherungen werden in einen zweiten, getrennten Speicher in der Schweiz kopiert (Swiss Backup von Infomaniak) und dort nach spätestens 35 Tagen gelöscht.</dd></div>
<div><dt>Training</dt><dd>Ein Training führt Code aus, den Brick DNN aus geprüften Einstellungen der Schichten erzeugt: Zahlen, Auswahlwerte und Namen, nie Python-Code von Nutzerinnen und Nutzern. Hochgeladene ZIP-Dateien werden mit Prüfungen gegen Pfadangriffe, Verknüpfungen und übergroße Inhalte entpackt.</dd></div>
<div><dt>Server</dt><dd>Ein Server von Infomaniak in Genf, Schweiz, mit der Datenbank auf derselben Maschine. Die <a href="/privacy">Datenschutzerklärung</a> nennt alle weiteren Dienstleister und was sie erhalten.</dd></div>
<div><dt>Diese Seiten</dt><dd>Statisches HTML mit einer strengen Content-Security-Policy. Sie laden nichts von anderen Servern.</dd></div>
<div><dt>Status</dt><dd>Eine automatische Prüfung fragt alle 15 Minuten, ob die Website antwortet. Ihre <a href="${UPTIME_URL}">Läufe</a> sind öffentlich, ebenso jeder <a href="${OUTAGES_URL}">Ausfall</a>, den sie gemeldet hat (beides auf GitHub, auf Englisch).</dd></div>
</dl>`,
    },
    {
      id: 'melden',
      title: 'Eine Schwachstelle melden',
      html: `<div class="prose"><p>Bitte eröffnen Sie kein öffentliches Issue. Melden Sie die Schwachstelle vertraulich über <a href="${SECURITY_ADVISORY_URL}">GitHub Security Advisories</a> oder per E-Mail an ${mail}. Sie erhalten so schnell wie möglich eine Antwort.</p>
<p>Derselbe Kontakt steht in der Datei <a href="${SECURITY_TXT_PATH}">security.txt</a>. Der Quellcode ist öffentlich: <a href="${REPOSITORY_URL}">github.com/alexanderKuehling/brick-dnn</a>.</p></div>`,
    },
  ],
};

const dataProtection: InfoPage = {
  id: 'dataProtection',
  title: 'Datenschutz im Kurs mit Brick DNN – Faktenblatt für Datenschutzbeauftragte',
  description:
    'Für Datenschutzbeauftragte und IT: was brickdnn.com verarbeitet, wenn ein Kurs es ohne Konten der Studierenden nutzt, wo es gespeichert wird und wie lange.',
  kicker: 'Faktenblatt',
  h1: 'Datenschutz in einer Lehrveranstaltung',
  lead: 'Für Datenschutzbeauftragte und IT-Abteilungen: was brickdnn.com verarbeitet, wenn eine Lehrveranstaltung es nutzt. Verbindlich ist die <a href="/privacy">Datenschutzerklärung</a>; diese Seite fasst sie zusammen.',
  sections: [
    {
      id: 'ohne-konten',
      title: 'Eine Übung ohne Konten der Studierenden',
      html: `<dl class="glance">
<div><dt>Konten</dt><dd>Keine. Netze bauen, prüfen und den Code exportieren geht ohne Registrierung. Der Server erfährt nicht, wer eine Studentin oder ein Student ist.</dd></div>
<div><dt>An den Server gesendet</dt><dd>Das Netz, an dem gerade gearbeitet wird, also seine Schichten und deren Einstellungen, damit der Server es prüfen und den Code erzeugen kann. Es wird für die Antwort verwendet und nicht gespeichert.</dd></div>
<div><dt>Im Browser gespeichert</dt><dd>Die Arbeitskopie des Netzes, die letzten Einstellungen, die Sprache und das Farbschema, im lokalen Speicher des Browsers. Keine Cookies.</dd></div>
<div><dt>Server-Protokolle</dt><dd>Die gekürzte IP-Adresse (bei IPv4 ist der letzte Block 0), Datum und Uhrzeit, die Seite ohne Parameter, Status, Datenmenge, Antwortzeit und die Browserkennung. Die Protokolle werden nach Größe überschrieben und können einige Wochen alt werden. Zähler je IP-Adresse, die die Zahl der Anfragen begrenzen, bestehen höchstens eine Stunde.</dd></div>
<div><dt>Fehlerberichte</dt><dd>Technische Fehler der App gehen an die EU-Region von Sentry: die Meldung, die Seite ohne Parameter, Browser und Betriebssystem und das aus der IP-Adresse abgeleitete Land. Kein Benutzername, keine Formularinhalte; die IP-Adresse wird dort nicht gespeichert.</dd></div>
<div><dt>Tracking</dt><dd>Keines. Keine Analyse, keine Werbung, keine Inhalte von anderen Servern.</dd></div>
</dl>`,
    },
    {
      id: 'mit-konten',
      title: 'Wenn Studierende auf dem Server trainieren',
      html: `<dl class="glance">
<div><dt>Konto</dt><dd>Benutzername, E-Mail-Adresse und das Passwort als Hash. Dieses kostenlose Konto braucht, wer auf dem Server trainieren, Netze speichern oder Daten hochladen will.</dd></div>
<div><dt>Inhalte</dt><dd>Die gespeicherten Netze, hochgeladene Dateien und die Ergebnisse der Trainings. Personenbezogene Daten anderer gehören nicht in Uploads; die eingebauten Datensätze kommen ohne aus.</dd></div>
<div><dt>E-Mail</dt><dd>Nur zum Konto: Adresse bestätigen, Passwort zurücksetzen, Sicherheitshinweise. Der Versand läuft über Brevo in der EU.</dd></div>
<div><dt>Löschung</dt><dd>Studierende löschen einzelne Uploads und Trainings oder das ganze Konto selbst. Uploads und Trainings, die 180 Tage niemand genutzt hat, werden automatisch gelöscht, Konten nach zwei Jahren ohne Nutzung.</dd></div>
</dl>`,
    },
    {
      id: 'wo',
      title: 'Wo, und wer beteiligt ist',
      html: `<dl class="glance">
<div><dt>Verantwortlicher</dt><dd>Alexander Kühling; die Anschrift steht im <a href="/imprint">Impressum</a>.</dd></div>
<div><dt>Server und Datenbank</dt><dd>Ein Server von Infomaniak in Genf, Schweiz, mit einem Vertrag zur Auftragsverarbeitung (Art. 28 DSGVO). Für die Schweiz hat die EU ein angemessenes Datenschutzniveau festgestellt (Art. 45 DSGVO).</dd></div>
<div><dt>Weitere Dienstleister</dt><dd>Sentry für Fehlerberichte und Brevo für E-Mails zum Konto, beide mit Verträgen zur Auftragsverarbeitung. Paddle nur, wenn jemand einen Tarif bucht.</dd></div>
<div><dt>Backups</dt><dd>Von der Datenbank stündlich und täglich, kopiert in einen zweiten Speicher in der Schweiz und dort nach spätestens 35 Tagen gelöscht.</dd></div>
<div><dt>Ohne brickdnn.com</dt><dd>Ein Institut kann Brick DNN auf einem eigenen Server betreiben; dann verlässt nichts davon die Einrichtung. Der exportierte Code läuft ohnehin auf den eigenen Rechnern.</dd></div>
</dl>
<div class="prose"><p>Braucht Ihre Einrichtung etwas, das hier fehlt? Schreiben Sie an ${mail}. Mehr für Lehrende: <a href="${TEACHING_PATHS.de}">Brick DNN in der Lehre</a>.</p></div>`,
    },
  ],
};

export const infoDe: InfoPage[] = [teaching, about, security, dataProtection];
