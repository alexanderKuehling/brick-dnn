import { CONTACT_EMAIL, OVERAGE_PRICE, PLANS, type PlanFacts } from '../site.ts';
import type { LandingContent } from './types.ts';

/** 0.5 → "0,5" */
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

const paid = PLANS.filter((plan) => plan.price > 0);
const paidPrices = `${paid
  .slice(0, -1)
  .map((plan) => euro(plan.price))
  .join(', ')} oder ${euro(paid[paid.length - 1].price)}`;
const free = PLANS.find((plan) => plan.price === 0)!;
const mail = `<a href="mailto:${CONTACT_EMAIL}">${CONTACT_EMAIL}</a>`;

export const de: LandingContent = {
  lang: 'de',
  locale: 'de_DE',
  title: 'Brick DNN – Deep Learning in Europa, DSGVO-konform, mit PyTorch-Export',
  description:
    'Deep-Learning-Modelle im Browser aus Schichten bauen, DSGVO-konform auf einem Server in der Schweiz trainieren und lesbaren PyTorch-Code exportieren. Kostenloser Tarif.',
  languageName: 'Deutsch',
  skipLink: 'Zum Inhalt springen',
  nav: {
    label: 'Abschnitte',
    useCases: 'Einsatzgebiete',
    features: 'Funktionen',
    pricing: 'Preise',
    faq: 'Häufige Fragen',
    editor: 'Editor öffnen',
  },
  hero: {
    title: 'Deep Learning in Europa: Netze aus Schichten bauen, PyTorch-Code exportieren.',
    lead: 'Brick DNN ist ein Editor im Browser für Deep Learning und klassisches maschinelles Lernen. Sie verbinden Schichten auf einer Arbeitsfläche, sehen schon beim Bauen die Form jedes Tensors, trainieren auf einem Server in der Schweiz und erhalten ein lesbares Skript für PyTorch oder scikit-learn. Gemacht für Hochschulen, Forschende und Unternehmen, deren Trainingsdaten in Europa bleiben sollen.',
    primary: 'Editor öffnen',
    secondary: 'Preise',
    facts: [
      'Kostenloser Tarif',
      'Bauen und Exportieren ohne Konto',
      'Open Source, Apache 2.0',
      'In der Schweiz gehostet, DSGVO-konform',
    ],
    image: {
      id: 'editor',
      alt: 'Der Editor von Brick DNN: links die Palette der Schichten, in der Mitte ein Faltungsnetz mit der Ausgabeform jeder Schicht, rechts die Einstellungen der ausgewählten Schicht.',
    },
    caption:
      'Ein Faltungsnetz für MNIST im Editor. Jede Schicht zeigt ihre Ausgabeform und die Zahl ihrer Parameter.',
  },
  glance: {
    title: 'Das Wichtigste in Kürze',
    items: [
      {
        term: 'Was es ist',
        html: 'Eine Webanwendung, mit der Sie Deep-Learning-Modelle und klassische Modelle des maschinellen Lernens entwerfen, prüfen, trainieren und exportieren, ohne zuerst Code zu schreiben. Ein Modell ist ein Graph aus Schichten („Bricks“); Brick DNN leitet die Form jedes Tensors ab und arbeitet als PyTorch-Code-Generator: Es erzeugt ein eigenständiges PyTorch-Skript, für klassische Modelle ein scikit-learn-Skript.',
      },
      {
        term: 'Für wen',
        html: 'Für Hochschulen, für Studierende und Lehrende im Bereich maschinelles Lernen und Deep Learning, für Forschende mit tabellarischen Daten oder Bilddaten und für Unternehmen in Europa, die schnell einen korrekten Ausgangspunkt brauchen und wissen wollen, wo ihre Daten liegen.',
      },
      {
        term: 'Was es kostet',
        html: `Bauen, Prüfen und Code-Export sind kostenlos und ohne Konto möglich. Der kostenlose Tarif enthält Training auf dem Server (${cpuHours(free.cpuHoursPerMonth)} im Monat, ${number(free.storageGb)} GB Speicher). Die kostenpflichtigen Tarife kosten ${paidPrices} im Monat, inklusive Umsatzsteuer.`,
      },
      {
        term: 'Wo die Daten liegen',
        html: 'In Europa. Die Anwendung, hochgeladene Dateien und Trainingsergebnisse liegen auf einem Server in der Schweiz, und dort laufen auch die Trainings. Konten und Projektangaben liegen in einer Datenbank in London (Vereinigtes Königreich). Keine Cookies, kein Tracking. Einzelheiten unter <a href="#privacy">Daten in Europa</a>.',
      },
      {
        term: 'Quellcode',
        html: 'Open Source unter der Apache-Lizenz 2.0. Brick DNN läuft mit Docker Compose auch auf Ihrem eigenen Server.',
      },
    ],
  },
  useCases: {
    title: 'Deep Learning in Lehre, Forschung und Praxis',
    intro:
      'Brick DNN wird in der Lehre eingesetzt, in der Forschung mit Tabellen und Bildern und überall dort, wo aus einer Architekturskizze lauffähiger Code werden soll.',
    items: [
      {
        title: 'Deep Learning unterrichten',
        html: 'Studierende verbinden Schichten und sehen, wie sich die Form der Daten von Schicht zu Schicht ändert. Passt etwas nicht, sagt eine verständliche Meldung warum, etwa dass der Kernel größer ist als die Eingabe. <em>How data flows</em> schickt ein Beispiel Schritt für Schritt durch das Netz; der Inspektor zeigt, wie sich die Parameter einer Schicht zusammensetzen und wie groß ihr rezeptives Feld ist. Für Gruppenarbeit lassen sich Netze teilen und gemeinsam in Echtzeit bearbeiten.',
        image: {
          id: 'data-flow',
          alt: '„How data flows“ im Editor: Eine Karte führt ein Beispiel Schritt für Schritt durch ein CNN und zeigt, wie Max-Pooling aus 32 × 28 × 28 Werten 32 × 14 × 14 macht, unter den Schichten auf der Arbeitsfläche.',
        },
      },
      {
        title: 'Tabellarische Forschungsdaten',
        html: 'Sie importieren CSV-, Excel-, Parquet- oder JSON-Dateien, bereinigen sie in nachvollziehbaren Schritten (fehlende Werte, Ausreißer, Kategorien, Datumsangaben) und prüfen, ob die Daten für das Training bereit sind. Gradient Boosting, Support Vector Machines und Gauß-Prozesse lassen sich auf derselben festen Aufteilung in Trainings-, Validierungs- und Testdaten mit neuronalen Netzen vergleichen; Vorhersagen erklären Shapley-Werte oder die Permutation Importance.',
        image: {
          id: 'data-prep',
          alt: 'Eine Aufbereitung von Feldversuchsdaten im Daten-Tab: links die Schritte, die fehlende Werte füllen, Ausreißer begrenzen und Kategorien kodieren, jeweils mit ihrem Ergebnis; rechts die aufbereitete Tabelle mit einer Verteilung im Kopf jeder Spalte.',
        },
      },
      {
        title: 'Bildsegmentierung',
        html: 'Trainieren Sie ein U-Net oder ein anderes Netz, das die Bildgröße beibehält, mit einer ZIP-Datei aus Bildern und Masken – oder ohne Upload mit dem eingebauten Datensatz <em>Shapes</em>. Für jede Epoche werden Mean IoU, Dice und Pixelgenauigkeit berichtet, und die vorhergesagten Masken erscheinen neben den echten.',
        image: {
          id: 'segmentation',
          alt: 'Vorhersagen eines U-Nets, trainiert auf dem eingebauten Datensatz Shapes: zu jedem Validierungsbild das Bild, die echte Maske, die vorhergesagte Maske und eine Überlagerung, mit IoU und Pixelgenauigkeit.',
        },
      },
      {
        title: 'Reproduzierbarer Code für Publikationen',
        html: 'Jedes Modell lässt sich als eigenständige Python-Datei exportieren: als <code>nn.Module</code> mit benannten Schichten für PyTorch oder als scikit-learn-Skript für klassische Modelle und Ensembles. Die Datenaufbereitung wird zum pandas-Skript. Der Code läuft ohne Brick DNN und kann mit der Veröffentlichung geteilt werden.',
        image: {
          id: 'code-export',
          alt: 'Der Code-Dialog mit dem erzeugten PyTorch-Modul eines kleinen CNN: die Schichten in __init__ und die Methode forward, bereit zum Herunterladen als Python-Datei.',
        },
      },
      {
        title: 'Ein KI-Assistent, der baut und trainiert',
        html: 'Verbinden Sie Claude oder einen anderen MCP-Client per OAuth mit Ihrem Konto. Der Assistent kann Netze bauen und prüfen, Daten aufbereiten, Trainings starten und Vorhersagen treffen. Sie legen fest, ob er Änderungen vornehmen oder nur lesen darf, können ihm ein Monatsbudget geben und sehen jede seiner Aktionen in einem Protokoll.',
        image: {
          id: 'agent-activity',
          alt: 'Die Seite „Agent activity“: die Aufrufe eines KI-Assistenten über den MCP-Server mit einem API-Token, etwa ein Netz analysieren, ein Training abrufen und Code erzeugen, jeweils mit Ergebnis und Dauer.',
        },
      },
    ],
  },
  steps: {
    title: 'So funktioniert es',
    items: [
      {
        title: 'Bauen',
        html: 'Ziehen Sie Schichten auf die Arbeitsfläche und verbinden Sie sie, oder beginnen Sie mit einem Beispiel: MLP, CNN, Residualblock, U-Net, LSTM, Transformer, Gradient Boosting oder ein Ensemble. Felder wie <code>in_features</code> füllt Brick DNN aus dem eingehenden Tensor selbst aus.',
        image: {
          id: 'build',
          alt: 'Ein Voting-Ensemble im Editor: ein Eingang, verbunden mit Gradient Boosting, einer Support Vector Machine und einem Gauß-Prozess, zusammengefasst von einem Voting-Klassifikator.',
        },
      },
      {
        title: 'Trainieren',
        html: 'Wählen Sie Demodaten, MNIST, Fashion-MNIST, CIFAR-10 oder eigene CSV- und Bilddateien. Brick DNN prüft vorab, ob Netz und Daten zusammenpassen, und trainiert dann auf dem Server – mit Verlust- und Metrikkurven in Echtzeit, Early Stopping und auf Wunsch einer Hyperparametersuche.',
        image: {
          id: 'training-setup',
          alt: 'Das Trainingsfenster mit einem ausgewählten aufbereiteten Datensatz und der Prüfung „Ready to train: classification into 2 classes from 18 features“.',
        },
      },
      {
        title: 'Exportieren',
        html: 'Laden Sie das PyTorch- oder scikit-learn-Skript, die trainierten Gewichte und die aufbereiteten Daten herunter. Der Code gehört Ihnen: Sie können ihn lesen, ändern und ausführen, wo Sie möchten.',
        image: {
          id: 'export',
          alt: 'Das Ende eines abgeschlossenen Trainings: Schaltflächen zum Herunterladen der Gewichte, der Vorverarbeitung und des Skripts oder um das Netz in den Editor zu laden.',
        },
      },
    ],
  },
  features: {
    title: 'Funktionen',
    intro:
      'Ein Werkzeug von der ersten Skizze bis zum trainierten und erklärten Deep-Learning-Modell.',
    items: [
      {
        title: 'Formen und Prüfungen',
        html: 'Jede Schicht zeigt schon beim Bauen ihre Ausgabeform und Parameterzahl. Fehler werden markiert, bevor irgendetwas läuft – mit einer Meldung, was zu ändern ist.',
        image: {
          id: 'checks',
          alt: 'Ein CNN mit einem Fehler: Eine Conv2d-Schicht ist rot markiert mit der Meldung „Kernel size 16 is larger than the (padded) input size 14“, die Schichten danach haben keine Form.',
        },
      },
      {
        title: 'Training auf dem Server',
        html: 'Warteschlange, Early Stopping, das die besten Gewichte behält, und Kurven für Verlust und Metriken zu jeder Epoche, während das Netz lernt.',
        image: {
          id: 'training-curves',
          alt: 'Verlust und Genauigkeit je Epoche eines kleinen CNN, trainiert auf MNIST, für Trainings- und Validierungsdaten, darüber die Testgenauigkeit von 98,6 %.',
        },
      },
      {
        title: 'Bewertung auf zurückgehaltenen Daten',
        html: 'Testdaten, einmal nach dem Training gemessen: Accuracy, Balanced Accuracy, F1 und eine Konfusionsmatrix; bei zwei Klassen außerdem ROC- und PR-Kurven und eine verschiebbare Entscheidungsschwelle.',
        image: {
          id: 'evaluation',
          alt: 'Die Konfusionsmatrix des CNN auf 3.000 Validierungsbildern von MNIST: wahre Ziffern in den Zeilen, vorhergesagte in den Spalten, fast alle auf der Diagonale.',
        },
      },
      {
        title: 'Hyperparametersuche',
        html: 'Zufalls- oder Gittersuche über Lernrate, Batchgröße und Optimierer oder über die Parameter eines klassischen Modells mit Kreuzvalidierung, bewertet nach dem Verlust oder nach Metriken wie Balanced Accuracy, ROC AUC, PR AUC, MAE oder R².',
        image: {
          id: 'search-results',
          alt: 'Die Durchläufe einer Zufallssuche für ein Gradient-Boosting-Modell: Lernrate, Tiefe, Zeilen je Blatt und Klassengewichte jedes Durchlaufs mit kreuzvalidiertem Verlust, Accuracy und PR AUC; der beste Durchlauf ist hervorgehoben.',
        },
      },
      {
        title: 'Erklärungen',
        html: 'Saliency Maps, Grad-CAM und Occlusion für Bilder; Shapley-Werte, Integrated Gradients und Permutation Importance für Tabellen; die Ausgabe jeder Schicht für ein einzelnes Beispiel.',
        image: {
          id: 'explain',
          alt: 'Permutation Importance des Gradient-Boosting-Modells: wie stark die Validierungsgenauigkeit sinkt, wenn ein Merkmal zufällig vertauscht wird – am stärksten beim Niederschlag, dann bei Temperatur und Stickstoff.',
        },
      },
      {
        title: 'Zusammenarbeit',
        html: 'Teilen Sie ein Netz mit Personen oder per Link, bearbeiten Sie es gemeinsam in Echtzeit, kommentieren Sie es und kehren Sie zu früheren Versionen zurück.',
        image: {
          id: 'collaboration',
          alt: 'Ein Netz, das zwei Personen teilen: ihre Avatare und die Anzeige „Live“ in der Werkzeugleiste, die von der anderen Person ausgewählte Schicht mit ihrem Namen markiert und ihre Frage dazu in der Kommentarleiste.',
        },
      },
      {
        title: 'Echte Graphen',
        html: 'Skip-Verbindungen (<code>Add</code>), Zusammenführungen (<code>Concat</code>), mehrere Ein- und Ausgänge. 24 Schichttypen für MLPs, CNNs, RNN, LSTM, GRU und Transformer.',
      },
      {
        title: 'Klassische Modelle und Ensembles',
        html: 'Gradient Boosting, Support Vector Machines und Gauß-Prozesse mit Kernels aus Bausteinen, Voting- und Stacking-Ensembles sowie hybride Netze, deren Schichten die Merkmale für ein klassisches Modell lernen.',
      },
      {
        title: 'Datenaufbereitung',
        html: 'CSV, Excel, OpenDocument, JSON, Parquet und Feather oder ein Link. Tabellen verbinden, Lücken füllen, Ausreißer begrenzen, skalieren, kodieren und Datumsangaben zerlegen – mit Diagrammen zu jedem Schritt und einer Bereitschaftsprüfung.',
      },
      {
        title: 'MCP-Server und REST-API',
        html: 'KI-Assistenten verbinden sich per OAuth, Skripte mit persönlichen API-Tokens. Beide arbeiten mit denselben Netzen, Daten und Trainings wie die Web-App.',
      },
      {
        title: 'Ihre Daten in Ihrer Hand',
        html: 'Bauen und Exportieren funktionieren ohne Konto. Mit einem Konto können Sie alle Ihre Daten herunterladen, das Konto selbst löschen und es mit einer Zwei-Faktor-Anmeldung schützen.',
      },
    ],
  },
  privacy: {
    title: 'Daten in Europa',
    intro:
      'Brick DNN wird aus Deutschland betrieben und richtet sich nach der DSGVO. Ihre Trainingsdaten bleiben in Europa. Was wo gespeichert wird und wie Sie es mitnehmen:',
    items: [
      {
        title: 'Server in der Schweiz',
        html: 'Die Anwendung, hochgeladene Dateien, aufbereitete Datensätze, trainierte Gewichte und Trainingsergebnisse liegen auf einem Server von Infomaniak in Genf, Schweiz. Dort laufen auch die Trainings; Ihre Daten verlassen den Server dafür nicht. Für die Schweiz hat die EU ein angemessenes Datenschutzniveau festgestellt (Art. 45 DSGVO).',
      },
      {
        title: 'Datenbank in London',
        html: 'Konten, Netze, Kommentare und die Angaben zu Daten und Trainings (nicht die Dateien selbst) liegen in einer PostgreSQL-Datenbank von Supabase in London, Vereinigtes Königreich. Auch dafür gibt es einen Angemessenheitsbeschluss der EU.',
      },
      {
        title: 'Export und Löschung',
        html: '<em>Meine Daten herunterladen</em> unter <em>Konto</em> liefert eine ZIP-Datei mit Ihren Netzen, Datenaufbereitungen, Trainingsergebnissen und hochgeladenen Dateien; die trainierten Gewichte laden Sie beim jeweiligen Training herunter. Einzelne Uploads und Trainings können Sie selbst löschen, ebenso das ganze Konto mit allem darin. Uploads und Trainings, die 180 Tage niemand genutzt hat, werden automatisch gelöscht.',
      },
      {
        title: 'Code, der überall läuft',
        html: 'Die exportierten Skripte für PyTorch, scikit-learn und pandas importieren nichts aus Brick DNN. Daten, die Ihre Einrichtung nicht verlassen dürfen, etwa personenbezogene Daten, bleiben dort: Sie bauen das Modell hier und trainieren den Code auf Ihren eigenen Rechnern, oder Sie betreiben Brick DNN auf einem eigenen Server. Wie das geht, zeigt die Anleitung <a href="/de/anleitungen/deep-learning-dsgvo-konform/">Deep Learning DSGVO-konform</a>.',
      },
      {
        title: 'Keine Cookies, kein Tracking',
        html: 'Keine Analyse, keine Werbung, keine Cookies in der App. Schriften, Skripte und Bilder kommen von diesem Server, abgesehen vom Bezahlvorgang bei Paddle, wenn Sie einen Tarif buchen. Diese Seite lädt nichts von anderen Servern.',
      },
      {
        title: 'Wenige, benannte Dienstleister',
        html: 'Technische Fehlerberichte der App gehen ohne Benutzername, E-Mail-Adresse oder Kontokennung an die EU-Region von Sentry; E-Mails zu Ihrem Konto verschickt Brevo in der EU. Für Hosting, Datenbank, E-Mail und Fehlerberichte bestehen Verträge zur Auftragsverarbeitung. Zahlungen laufen über Paddle, und nur, wenn Sie einen Tarif buchen.',
      },
    ],
    more: 'Die Datenschutzerklärung nennt alle Dienstleister, Zwecke und Speicherfristen.',
  },
  teaching: {
    title: 'Für Hochschulen und Forschung',
    items: [
      'Ein kostenloser Tarif mit Training auf dem Server; zum Bauen von Netzen und für den Code-Export braucht niemand ein Konto.',
      'Der exportierte PyTorch- und scikit-learn-Code läuft ohne Brick DNN. Kursmaterial und veröffentlichte Ergebnisse bleiben so nutzbar.',
      'Die Daten bleiben in Europa, auf Servern in der Schweiz und im Vereinigten Königreich, ohne Tracking.',
      'Open Source: Hochschulen und Institute können eine eigene Instanz betreiben.',
    ],
    discount: `Für Lehre und Forschung gibt es auf Anfrage einen Rabatt: ${mail}.`,
  },
  pricing: {
    title: 'Preise',
    intro:
      'Bauen, Prüfen und Exportieren sind kostenlos. Die Tarife unterscheiden sich in Rechenzeit und Speicher für das Training auf dem Server.',
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
    notes: [
      `In den kostenpflichtigen Tarifen ist zusätzliche Rechenzeit über das Kontingent hinaus für ${euro(OVERAGE_PRICE)} je CPU-Stunde möglich, nur wenn Sie sie selbst einschalten.`,
      'Monatliche Abos, zum Monatsende kündbar. Zahlung und Rechnung übernimmt Paddle.',
    ],
    more: 'Alle Details auf der Preisseite',
  },
  faq: {
    title: 'Häufige Fragen',
    items: [
      {
        question: 'Was ist Brick DNN?',
        html: 'Brick DNN ist eine Webanwendung für Deep Learning: Sie setzen neuronale Netze und klassische Modelle des maschinellen Lernens ohne Code aus Schichten („Bricks“) zusammen. Sie leitet die Form jedes Tensors ab, zeigt Fehler, bevor etwas läuft, trainiert Modelle auf dem Server und exportiert sie als lesbaren Code für PyTorch oder scikit-learn.',
      },
      {
        question: 'Brauche ich ein Konto?',
        html: 'Zum Bauen und Prüfen von Netzen und für den Code-Export nicht. Um Netze zu speichern, Daten hochzuladen und zu trainieren, brauchen Sie ein kostenloses Konto.',
      },
      {
        question: 'Welche Frameworks nutzt der exportierte Code?',
        html: 'Neuronale Netze werden als PyTorch-Code exportiert (ein <code>nn.Module</code> mit einem Funktionstest und für Tabellendaten einer Trainingsschleife für CSV-Dateien). Klassische Modelle und Ensembles werden zu scikit-learn-Skripten, Datenaufbereitungen zu pandas-Skripten. Trainierte Gewichte lassen sich zusammen mit dem passenden Skript herunterladen.',
      },
      {
        question: 'Wo werden meine Daten gespeichert und verarbeitet?',
        html: 'In Europa. Hochgeladene Dateien, trainierte Gewichte und Trainingsergebnisse werden auf einem Server von Infomaniak in der Schweiz gespeichert und verarbeitet. Konten und Projektangaben liegen in einer Datenbank von Supabase in London. Es gibt keine Cookies und kein Tracking; die Datenschutzerklärung nennt alle Dienstleister.',
      },
      {
        question: 'Kann ich mit Brick DNN Deep-Learning-Modelle DSGVO-konform trainieren?',
        html: 'Brick DNN richtet sich für Ihr Konto und die hochgeladenen Daten nach der DSGVO und trainiert auf seinem Server in der Schweiz. Personenbezogene Daten anderer und vertrauliche Daten gehören aber nicht auf brickdnn.com: Anonymisieren Sie sie vorher, oder bauen Sie das Modell hier und trainieren Sie den exportierten Code oder eine eigene Brick-DNN-Instanz auf den Rechnern Ihrer Einrichtung. Die <a href="/de/anleitungen/deep-learning-dsgvo-konform/">Anleitung zu Deep Learning nach der DSGVO</a> geht die Möglichkeiten durch.',
      },
      {
        question: 'Was kostet Brick DNN?',
        html: `Bauen, Prüfen und Exportieren sind kostenlos. Der kostenlose Tarif enthält ${cpuHours(free.cpuHoursPerMonth)} Training im Monat und ${number(free.storageGb)} GB Speicher. Tarife mit mehr Rechenzeit und Speicher kosten ${paidPrices} im Monat, inklusive Umsatzsteuer.`,
      },
      {
        question: 'Kann ich Brick DNN in einer Lehrveranstaltung oder für die Forschung nutzen?',
        html: `Ja. Studierende bauen und exportieren Netze ohne Konto, und der kostenlose Tarif reicht für kleinere Trainings. Für Lehre und Forschung gibt es auf Anfrage einen Rabatt: ${mail}.`,
      },
      {
        question: 'Kann ich Brick DNN auf einem eigenen Server betreiben?',
        html: 'Ja. Brick DNN ist Open Source unter der Apache-Lizenz 2.0 und läuft mit Docker Compose. Auf dem eigenen Rechner nutzt das Training eine CUDA- oder Apple-GPU, wenn PyTorch eine findet.',
      },
      {
        question: 'Welche KI-Assistenten funktionieren mit Brick DNN?',
        html: 'Jeder Client des Model Context Protocol (MCP), der sich per OAuth anmeldet, zum Beispiel Claude. Clients ohne OAuth und eigene Skripte verwenden persönliche API-Tokens.',
      },
    ],
  },
  footer: {
    tagline: 'Deep Learning aus Bausteinen, in der Schweiz gehostet, mit lesbarem PyTorch-Code.',
    legalLabel: 'Rechtliches',
    legal: {
      imprint: 'Impressum',
      privacy: 'Datenschutz',
      terms: 'Nutzungsbedingungen',
      refund: 'Erstattung',
      accessibility: 'Barrierefreiheit',
    },
    legalNote: '',
    source: 'Quellcode auf GitHub',
    contact: 'Kontakt',
    language: 'Sprache',
  },
  appDescription:
    'Editor im Browser für Deep Learning und klassische Modelle des maschinellen Lernens, mit Formprüfung in Echtzeit, DSGVO-konformem Training auf einem Server in der Schweiz und Export von lesbarem PyTorch- und scikit-learn-Code.',
  featureList: [
    'Drag-and-drop-Editor für Deep-Learning-Modelle mit Formprüfung in Echtzeit',
    'PyTorch-Code-Generator: Export von lauffähigem PyTorch- und scikit-learn-Code',
    'Hosting und Training auf einem Server in der Schweiz, DSGVO-konform',
    'Gradient Boosting, Support Vector Machines, Gauß-Prozesse, Ensembles und hybride Netze',
    'Datenaufbereitung für CSV-, Excel-, JSON- und Parquet-Dateien',
    'Training auf dem Server mit Hyperparametersuche',
    'Bildsegmentierung mit U-Net',
    'Erklärungen: Saliency Maps, Grad-CAM, Shapley-Werte, Permutation Importance',
    'Gemeinsames Bearbeiten geteilter Netze in Echtzeit',
    'MCP-Server für KI-Assistenten und REST-API',
  ],
  ogImageAlt: 'Brick DNN: der Editor mit einem Faltungsnetz und der Form jeder Schicht.',
};
