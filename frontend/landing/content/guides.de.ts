/**
 * Die Anleitungen auf Deutsch. Alles, was sie sagen, muss für die App stimmen (docs/training.md,
 * docs/data-preparation.md, docs/explain.md); die Screenshots sind die aus images/. Die
 * Bezeichnungen der Oberfläche sind die der deutschen App (src/i18n/locales/de).
 */
import type { Guide, GuideIndex } from './types.ts';

const REPOSITORY = 'https://github.com/alexanderKuehling/brick-dnn';
const DOCS = `${REPOSITORY}/blob/main/docs`;
const DATE = '2026-11-16';
const GDPR_DATE = '2026-11-17';
const BASICS_DATE = '2026-11-18';

const index: GuideIndex = {
  title: 'Anleitungen: Deep Learning, Tabellendaten, Segmentierung, DSGVO – Brick DNN',
  description:
    'Praxisnahe Anleitungen zu Deep Learning: Netz ohne Code bauen und als PyTorch exportieren, Gradient Boosting oder MLP, U-Net, seltene Klassen und DSGVO.',
  h1: 'Anleitungen',
  intro:
    'Schritt-für-Schritt-Anleitungen zum Bauen, Trainieren und Bewerten von Deep-Learning-Modellen mit Brick DNN, und zu den Entscheidungen unterwegs: welches Modell für eine Tabelle, wie man Bilder segmentiert, was zu tun ist, wenn eine Klasse selten ist, und wie Sie mit Daten trainieren, für die die DSGVO gilt.',
};

const guides: Guide[] = [
  {
    id: 'no-code',
    slug: 'neuronales-netz-ohne-code',
    title: 'Neuronales Netz ohne Code bauen und als PyTorch-Code exportieren',
    description:
      'Deep Learning ohne Code: ein neuronales Netz im Browser aus Schichten bauen, jede Tensorform prüfen lassen und sauberen PyTorch-Code herunterladen.',
    h1: 'Neuronales Netz ohne Code bauen und als PyTorch-Code exportieren',
    summary:
      'Schichten auf einer Arbeitsfläche verbinden, jede Form von Brick DNN prüfen lassen und ein PyTorch-Skript herunterladen, das für sich läuft.',
    lead: 'Ein neuronales Netz lässt sich entwerfen, indem Sie Schichten auf einer Arbeitsfläche verbinden, und am Ende steht trotzdem gewöhnlicher PyTorch-Code. Diese Anleitung baut ein kleines Faltungsnetz für 28×28-Bilder, prüft es und exportiert es, ganz ohne Konto.',
    published: DATE,
    modified: DATE,
    type: 'HowTo',
    totalTime: 'PT15M',
    sections: [
      {
        id: 'vorher',
        title: 'Was Sie brauchen',
        html: '<p>Einen aktuellen Browser. Bauen, Prüfen und Exportieren gehen ohne Konto und kosten nichts; nur um Netze in einem Konto zu speichern, Daten hochzuladen und auf dem Server zu trainieren, brauchen Sie ein kostenloses Konto. Für den exportierten Code brauchen Sie Python und PyTorch auf Ihrem Rechner.</p>',
      },
      {
        id: 'bauen',
        title: 'Das Netz bauen',
        html: '<p>Öffnen Sie den <a href="/app">Editor</a>. Beim ersten Besuch zeigt er ein Beispiel, damit gleich etwas zu sehen ist; <strong>Datei → Neues Netz</strong> leert die Arbeitsfläche.</p>',
        steps: [
          {
            title: 'Mit einem Input beginnen',
            html: 'Ziehen Sie <code>Input</code> aus der Palette auf die Arbeitsfläche und geben Sie seine Form ohne Batch-Dimension an: <code>1, 28, 28</code> für einen Graustufenkanal mit 28 × 28 Pixeln oder <code>784</code> für dasselbe Bild als flachen Vektor.',
          },
          {
            title: 'Schichten anfügen',
            html: 'Ziehen Sie <code>Conv2d</code> auf die Arbeitsfläche und verbinden Sie die rechte Kante des Inputs mit der linken Kante der neuen Schicht. Ist eine Schicht ausgewählt, hängt ein Klick auf eine Schicht der Palette (oder die Eingabetaste) die neue direkt dahinter und verbindet beide. Weiter geht es mit <code>Activation</code> (ReLU), <code>MaxPool2d</code>, einem zweiten Faltungsblock, <code>Flatten</code> und zwei <code>Linear</code>-Schichten, die letzte mit 10 Ausgaben.',
          },
          {
            title: 'Festlegen, was zählt; den Rest füllt Brick DNN aus',
            html: 'Wählen Sie eine Schicht aus, um ihre Einstellungen zu sehen. Sie legen fest, was eine Entwurfsentscheidung ist, etwa <code>out_channels</code> oder <code>kernel_size</code>; Felder, die sich aus dem eingehenden Tensor ergeben, wie <code>in_channels</code> und <code>in_features</code>, füllt Brick DNN selbst aus, auch wenn Sie später eine frühere Schicht ändern.',
          },
          {
            title: 'Die Formen lesen',
            html: 'Jede Schicht zeigt schon beim Bauen ihre Ausgabeform und Parameterzahl. Ein Fehler wird rot markiert, bevor irgendetwas läuft, mit einer Meldung, was zu ändern ist, etwa dass ein Kernel größer ist als seine Eingabe. Die Schichten dahinter haben keine Form, bis er behoben ist.',
          },
        ],
        image: {
          id: 'checks',
          alt: 'Ein Faltungsnetz im Editor, eine Conv2d-Schicht rot markiert mit der Meldung „Kernel size 16 is larger than the (padded) input size 14“; die Schichten dahinter zeigen keine Form.',
          caption:
            'Ein Fehler erscheint dort, wo er entsteht, samt Begründung. Die folgenden Schichten warten auf die Korrektur.',
        },
        after:
          '<p>Skip-Verbindungen (<code>Add</code>), Zusammenführungen (<code>Concat</code>) und mehrere Eingaben funktionieren genauso: Tensoren verbinden, und die Formprüfung kontrolliert, ob sie zusammenpassen. <strong>Layout aufräumen</strong> ordnet ein Netz, das unübersichtlich geworden ist.</p>',
      },
      {
        id: 'verstehen',
        title: 'Prüfen, ob es tut, was Sie denken',
        html: '<p>Die Werkzeugleiste zählt die trainierbaren Parameter des ganzen Netzes. Der Inspektor einer ausgewählten Schicht erklärt, wie sich ihre Parameter zusammensetzen und wie groß ihr rezeptives Feld ist. <em>Wie die Daten fließen</em> schickt eine Stichprobe Schicht für Schicht durch das Netz und zeigt, wie jede Schicht die Form verändert, zum Beispiel wie Max-Pooling aus 32 × 28 × 28 Werten 32 × 14 × 14 macht.</p>',
        image: {
          id: 'data-flow',
          alt: 'Die Karte „Wie die Daten fließen“ im Editor verfolgt eine Stichprobe durch ein CNN und erklärt, wie Max-Pooling 32 × 28 × 28 Werte auf 32 × 14 × 14 halbiert.',
          caption: 'Wie die Daten fließen: eine Stichprobe, eine Schicht nach der anderen.',
        },
      },
      {
        id: 'exportieren',
        title: 'Den PyTorch-Code exportieren',
        html: '<p><strong>Code</strong> in der Werkzeugleiste öffnet das erzeugte Skript. Es ist eine eigenständige Python-Datei: ein <code>nn.Module</code> mit einem benannten Attribut je Schicht in <code>__init__</code> und einer <code>forward</code>-Methode in der Reihenfolge des Graphen. Sie können den Klassennamen ändern und wählen, ob die Datei ein ausführbares Skript ist (mit einem <code>__main__</code>-Teil, der das Modell ausgibt und einen zufälligen Batch hindurchschickt). Für ein Netz mit einer flachen Eingabe und einer flachen Ausgabe kann sie auch eine Trainingsschleife für CSV-Dateien enthalten. <strong>.py exportieren</strong> (Strg+E) lädt sie herunter.</p>',
        image: {
          id: 'code-export',
          alt: 'Der Code-Dialog mit dem erzeugten PyTorch-Modul eines kleinen CNN: die Schichten in __init__, die forward-Methode und die Optionen für Klassenname und ausführbares Skript.',
          caption:
            'Der Code-Dialog: ein nn.Module mit benannten Schichten, bereit zum Herunterladen.',
        },
        after:
          '<p>Auf Ihrem Rechner gibt <code>python model.py</code> das Modell, die Zahl der trainierbaren Parameter und die Ausgabeform eines zufälligen Batches aus; sie sollte zu dem passen, was der Editor angezeigt hat. Der Code importiert nichts aus Brick DNN: Sie können ihn ändern, in ein Repository oder eine Veröffentlichung übernehmen und mit einer eigenen Schleife trainieren. Die <a href="/de/beispiele/cnn-mnist/">Beispielseite dieses CNN</a> zeigt das vollständige Skript.</p>',
      },
      {
        id: 'trainieren',
        title: 'Optional: auf dem Server trainieren',
        html: '<p>Mit einem kostenlosen Konto trainiert <strong>Trainieren</strong> das Netz auf dem Server von Brick DNN in der Schweiz. Zur Wahl stehen MNIST, Fashion-MNIST, CIFAR-10, eine eigene CSV-Datei oder eine ZIP-Datei mit Bildern. Brick DNN prüft vorab, ob Netz und Daten zusammenpassen, passt Input und letzte Schicht an, wenn nur diese abweichen, und zeigt dann Verlust und Genauigkeit je Epoche. Nach dem Lauf laden Sie die Gewichte (<code>.pt</code>) zusammen mit dem passenden Skript herunter:</p><pre class="snippet"><code>import torch\nfrom model import NeuralNetwork  # das heruntergeladene Skript\n\nmodel = NeuralNetwork()\nmodel.load_state_dict(torch.load("model.pt"))\nmodel.eval()</code></pre>',
        image: {
          id: 'export',
          alt: 'Das Ende eines abgeschlossenen Trainings mit Schaltflächen zum Herunterladen der Gewichte, der Vorverarbeitung und des Skripts sowie zum Laden des Netzes in den Editor.',
          caption: 'Nach dem Training: Gewichte, Vorverarbeitung und das Skript, das dazu passt.',
        },
      },
      {
        id: 'behalten',
        title: 'Das Netz behalten und teilen',
        html: `<p><strong>Als JSON-Datei speichern</strong> im Menü Datei sichert den Graphen ohne Konto, <strong>JSON-Datei öffnen …</strong> holt ihn zurück. Mit Konto werden Netze samt Versionsverlauf gespeichert und lassen sich mit Personen oder per Link teilen und gemeinsam in Echtzeit bearbeiten. Das Graphformat ist in <a href="${DOCS}/graph-format.md">graph-format.md</a> beschrieben (englisch), und dieselben Netze kann auch ein KI-Assistent über den <a href="${DOCS}/mcp.md">MCP-Server</a> bauen.</p>`,
      },
    ],
    examples: ['cnn_mnist', 'mlp_mnist', 'resnet_block'],
  },
  {
    id: 'tabular',
    slug: 'gradient-boosting-oder-mlp',
    title: 'Tabellendaten: Gradient Boosting oder MLP – und wann was passt',
    description:
      'Gradient Boosting oder neuronales Netz für Tabellendaten? Wie beide eine Tabelle sehen, was sie brauchen und wie Sie sie fair auf demselben Split vergleichen.',
    h1: 'Tabellendaten: Gradient Boosting oder MLP – und wann was passt',
    summary:
      'Wie Bäume und neuronale Netze eine Tabelle sehen, was beide brauchen und wie Sie sie fair auf demselben Split vergleichen.',
    lead: 'Für eine Tabelle aus Zeilen und Spalten ist ein Gradient Boosting meist das stärkere erste Modell, ein mehrschichtiges Perzeptron (MLP) das flexiblere. Welches gewinnt, hängt von Ihren Daten ab. Die eigentliche Frage ist daher, wie Sie es schnell und fair herausfinden.',
    published: DATE,
    modified: DATE,
    type: 'Article',
    sections: [
      {
        id: 'kurz',
        title: 'Die kurze Antwort',
        html: '<p>Beginnen Sie mit Gradient Boosting. Es braucht wenig Vorbereitung und wenig Feinabstimmung, kommt mit Merkmalen in ganz unterschiedlichen Maßstäben zurecht und findet Wechselwirkungen zwischen Spalten von selbst. Probieren Sie ein MLP daneben, wenn die Tabelle groß ist, wenn der Zusammenhang glatt ist, wenn mehrere Zielgrößen auf einmal gebraucht werden oder wenn die Tabelle nur eine Eingabe neben einem Bild oder Text ist. Vergleichen Sie dann beide auf denselben zurückgehaltenen Zeilen.</p>',
      },
      {
        id: 'sicht',
        title: 'Wie die Modelle eine Tabelle sehen',
        html: '<p><strong>Gradient Boosting</strong> baut Entscheidungsbäume nacheinander, jeder korrigiert die Fehler der vorherigen. Ein Baum teilt jeweils nach einer Spalte („Niederschlag über 41 mm“). Deshalb spielt der Maßstab einer Spalte keine Rolle, einzelne Extremwerte schaden kaum, und Sprünge in den Daten sind leicht abzubilden. Außerhalb des Bereichs der Trainingsdaten sagt ein Baum den Wert seines äußersten Blatts vorher: Er extrapoliert nicht.</p><p><strong>Ein MLP</strong> multipliziert die standardisierten Merkmale mit Gewichten, summiert sie und schickt das Ergebnis Schicht für Schicht durch nichtlineare Aktivierungen. Es bildet glatte Zusammenhänge gut ab und lernt aus allen Spalten zugleich, braucht aber Zahlen in vergleichbaren Maßstäben, genug Zeilen für seine Gewichte und Einstellungen wie Lernrate und Zahl der Epochen.</p>',
      },
      {
        id: 'vergleich',
        title: 'Gegenübergestellt',
        html: `<div class="table" role="region" aria-labelledby="vergleich-caption" tabindex="0"><table><caption id="vergleich-caption">Gradient Boosting und ein MLP auf Tabellendaten, in Brick DNN</caption><thead><tr><th scope="col">Frage</th><th scope="col">Gradient Boosting</th><th scope="col">MLP</th></tr></thead><tbody>
<tr><th scope="row">Vorbereitung</th><td>Fehlende Werte auffüllen und Kategorien kodieren (im Bereich Daten); keine Skalierung nötig</td><td>Ebenso; das Training standardisiert die Merkmale (Mittelwert 0, Standardabweichung 1)</td></tr>
<tr><th scope="row">Zeilen</th><td>Bis 1.000.000 Trainingszeilen, Rechenzeit etwa linear in der Zeilenzahl</td><td>Beliebig viele, in Batches; profitiert von vielen Zeilen</td></tr>
<tr><th scope="row">Einstellungen</th><td>Lernrate, Blätter je Baum, Zeilen je Blatt; gute Standardwerte</td><td>Schichtgrößen, Lernrate, Batchgröße, Epochen, Dropout</td></tr>
<tr><th scope="row">Training</th><td>Einmal angepasst, in Sekunden bis Minuten</td><td>Epoche für Epoche, mit Kurven und frühem Stoppen</td></tr>
<tr><th scope="row">Seltene Klassen</th><td><code>class_weight</code> <code>balanced</code></td><td>Gewichtete Kreuzentropie oder Focal Loss</td></tr>
<tr><th scope="row">Mehrere Zielgrößen, andere Eingaben</th><td>Eine Zielgröße</td><td>Mehrere Ausgaben; kann ein Zweig eines größeren Netzes sein</td></tr>
<tr><th scope="row">Export</th><td>scikit-learn-Skript und angepasste Pipeline (<code>.joblib</code>)</td><td>PyTorch-Skript und Gewichte (<code>.pt</code>)</td></tr>
</tbody></table></div>`,
      },
      {
        id: 'fair',
        title: 'Auf denselben Zeilen vergleichen',
        html: '<p>Fair ist ein Vergleich nur, wenn beide Modelle aus denselben Trainingszeilen lernen und auf denselben Zeilen bewertet werden, die sie nie gesehen haben. In Brick DNN liefert genau das ein aufbereiteter Datensatz:</p>',
        steps: [
          {
            title: 'Die Tabelle einmal aufbereiten',
            html: 'Importieren Sie die Datei im Bereich <strong>Daten</strong>, wählen Sie die Zielgröße und ergänzen Sie Schritte für fehlende Werte, Ausreißer, Kategorien und Datumsangaben. Jede Zeile wird dabei einmal dem Training, der Validierung oder dem Test zugeordnet, und Schritte, die Werte lernen (einen Median, eine Skalierung), lernen sie nur aus den Trainingszeilen.',
          },
          {
            title: 'Den Trainingsdatensatz erstellen',
            html: '<strong>Trainingsdatensatz erstellen</strong> speichert die aufbereitete Tabelle samt Split. Beide Modelle verwenden diesen Split statt eines eigenen zufälligen.',
          },
          {
            title: 'Gradient Boosting trainieren',
            html: 'Öffnen Sie das <a href="/de/beispiele/gradient-boosting-classifier/">Gradient-Boosting-Beispiel</a> (oder dessen Regressionsbaustein), wählen Sie den Datensatz und trainieren Sie. Der Input passt sich der Zahl der Merkmale an. Eine Hyperparametersuche mit Kreuzvalidierung auf den Trainingszeilen stimmt die Bäume ab, ohne die Validierungszeilen anzurühren.',
          },
          {
            title: 'Ein MLP trainieren',
            html: 'Öffnen Sie das <a href="/de/beispiele/tabular-regression/">MLP-Beispiel für Tabellen</a> oder beginnen Sie mit einer leeren Arbeitsfläche: Wählen Sie dort den Datensatz, legt Brick DNN ein kleines Netz dafür an (Input → Linear → ReLU → Linear). Schalten Sie frühes Stoppen ein und trainieren Sie.',
          },
          {
            title: 'Die Testwerte vergleichen',
            html: 'Vergleichen Sie auf den Testzeilen, nicht auf den Validierungszeilen, die die Epoche oder den Versuch ausgewählt haben: Balanced Accuracy, Makro-F1 oder PR AUC bei Klassen, RMSE und R² bei einer Zahl. Der Verlauf behält beide Läufe, jeden mit genau dem Netz, das trainiert wurde.',
          },
        ],
        image: {
          id: 'data-prep',
          alt: 'Eine Aufbereitung im Bereich Daten: links die Schritte für fehlende Werte, Ausreißer und Kategorien mit dem, was jeder bewirkt hat, rechts die aufbereitete Tabelle mit einem kleinen Verteilungsdiagramm über jeder Spalte.',
          caption: 'Eine Aufbereitung, ein Split, für jedes Modell im Vergleich.',
        },
      },
      {
        id: 'abstimmen',
        title: 'Beide kurz abstimmen',
        html: '<p>Unveränderte Standardwerte bevorzugen Gradient Boosting, weil dessen Standardwerte für die meisten Tabellen gut sind. Gönnen Sie dem MLP eine kurze Zufallssuche über Lernrate und Batchgröße und den Bäumen eine über Lernrate, Blätter je Baum und Zeilen je Blatt. Ordnen Sie beide Suchen nach der Metrik, die Sie berichten wollen, nicht nach dem Verlust.</p>',
        image: {
          id: 'search-results',
          alt: 'Die Versuche einer Zufallssuche für Gradient Boosting: Lernrate, Tiefe, Zeilen je Blatt und Klassengewichte jedes Versuchs mit kreuzvalidiertem Verlust, Genauigkeit und PR AUC; der beste Versuch ist hervorgehoben.',
          caption: 'Eine Suche über die Parameter des Gradient Boosting, geordnet nach PR AUC.',
        },
      },
      {
        id: 'welches',
        title: 'Also welches?',
        html: '<ul class="checks"><li><strong>Gradient Boosting</strong>, wenn die Tabelle klein oder mittelgroß ist, Spalten ganz unterschiedlicher Art und Größe hat, Schwellen und Sprünge enthält oder wenn Sie schnell ein starkes Ergebnis brauchen.</li><li><strong>Ein MLP</strong>, wenn die Tabelle groß und der Zusammenhang glatt ist, wenn mehrere Zielgrößen zugleich vorhergesagt werden sollen, wenn es etwas über den Trainingsbereich hinaus extrapolieren muss oder wenn die Tabelle eine Eingabe eines Netzes ist, das auch Bilder oder Text sieht (siehe das <a href="/de/beispiele/multi-input/">Beispiel mit mehreren Eingaben</a>).</li><li><strong>Beides</strong>: Ein <a href="/de/beispiele/hybrid-mlp-gradient-boosting/">hybrides Netz</a> lässt ein MLP Merkmale für ein Gradient Boosting lernen, und ein <a href="/de/beispiele/voting-ensemble-classifier/">Voting-</a> oder <a href="/de/beispiele/stacking-ensemble-regression/">Stacking-Ensemble</a> kombiniert mehrere klassische Modelle.</li></ul>',
      },
      {
        id: 'erklaeren',
        title: 'Ansehen, was der Gewinner gelernt hat',
        html: '<p><strong>Das Modell erklären</strong> unter einem abgeschlossenen Lauf funktioniert für beide Arten: Die Permutation Importance zeigt, wie stark der Validierungswert sinkt, wenn eine Spalte durchmischt wird, und Shapley-Werte zeigen, wie jedes Merkmal eine einzelne Vorhersage verschoben hat. Stützen sich die beiden Modelle auf ganz verschiedene Spalten, sehen Sie sich die Daten noch einmal an, bevor Sie einem der beiden trauen.</p>',
        image: {
          id: 'explain',
          alt: 'Permutation Importance eines Gradient-Boosting-Modells: wie stark die Validierungsgenauigkeit sinkt, wenn ein Merkmal durchmischt wird, am meisten beim Niederschlag, danach bei Temperatur und Stickstoff.',
          caption: 'Permutation Importance: welche Spalten das Modell wirklich nutzt.',
        },
      },
    ],
    examples: [
      'gradient_boosting_classifier',
      'tabular_regression',
      'hybrid_mlp_gradient_boosting',
    ],
  },
  {
    id: 'segmentation',
    slug: 'bildsegmentierung-u-net',
    title: 'Bildsegmentierung mit U-Net: von den Masken zum trainierten Modell',
    description:
      'Deep Learning zur Bildsegmentierung: ein U-Net trainieren, Bilder und Masken vorbereiten, Batchgröße und Verlust wählen, mittlere IoU und Dice lesen.',
    h1: 'Bildsegmentierung mit U-Net: von den Masken zum trainierten Modell',
    summary:
      'Von einer ZIP-Datei mit Bildern und Masken zum trainierten U-Net, mit mittlerer IoU, Dice und den vorhergesagten Masken.',
    lead: 'Segmentierung gibt jedem Pixel eine Klasse: Hintergrund, Straße, Zelle, Fehlstelle. Diese Anleitung trainiert ein U-Net in Brick DNN, zuerst auf den eingebauten Shapes-Daten, dann auf eigenen Bildern und Masken, und erklärt, wie Sie die Ergebnisse lesen.',
    published: DATE,
    modified: DATE,
    type: 'HowTo',
    totalTime: 'PT20M',
    sections: [
      {
        id: 'idee',
        title: 'Was ein Segmentierungsnetz braucht',
        html: '<p>Ein Klassifikator gibt eine Antwort je Bild, ein Segmentierungsnetz eine je Pixel. Seine Ausgabe ist deshalb so hoch und breit wie sein Input und hat einen Kanal je Klasse: <code>Klassen, Höhe, Breite</code>. Das Netz endet mit rohen Scores, nicht mit Softmax. Ein U-Net ist genau dafür gebaut: ein Encoder, der das Bild verkleinert, um zu erfassen, was darauf ist, ein Decoder, der es wieder auf volle Größe bringt, und Skip-Verbindungen, die die Details hinübertragen (jede Schicht mit ihrer Form zeigt das <a href="/de/beispiele/unet/">U-Net-Beispiel</a>).</p>',
      },
      {
        id: 'erster-lauf',
        title: 'Einmal auf eingebauten Daten trainieren',
        html: '<p>Bevor Sie eigene Daten vorbereiten, prüfen Sie den ganzen Weg mit Daten, die nichts hochladen müssen:</p>',
        steps: [
          {
            title: 'Das U-Net-Beispiel öffnen',
            html: 'Öffnen Sie das <a href="/app?example=unet">U-Net im Editor</a>. Sein Input ist <code>3, 64, 64</code>, seine letzte 1×1-Faltung hat vier Ausgabekanäle.',
          },
          {
            title: 'Shapes wählen',
            html: 'Klicken Sie auf <strong>Trainieren</strong> und wählen Sie <strong>Eingebaut → Shapes</strong>: 500 erzeugte Bilder mit 64 × 64 Pixeln, Quadraten, Kreisen und Dreiecken auf verrauschtem Hintergrund, und zu jedem eine Maske mit vier Klassen. Der Datensatz passt so, wie er ist.',
          },
          {
            title: 'Das Training starten',
            html: 'Behalten Sie die Batchgröße 8 und zehn Epochen bei und starten Sie. Auf einer CPU dauert das wenige CPU-Minuten und erreicht etwa 80 % mittlere IoU, gut innerhalb der 15 CPU-Minuten je Training im kostenlosen Tarif.',
          },
        ],
      },
      {
        id: 'daten',
        title: 'Eigene Bilder und Masken vorbereiten',
        html: '<p>Ihre Daten sind eine ZIP-Datei mit einem Ordner <code>images</code> und einem Ordner <code>masks</code>. Bild und Maske haben denselben Dateinamen; die Maske darf zusätzlich auf <code>_mask</code>, <code>_label</code>, <code>_seg</code> oder <code>_gt</code> enden, und die Endung darf abweichen:</p><pre class="snippet"><code>images/001.jpg        masks/001.png\nimages/002.jpg        masks/002_mask.png\nclasses.txt           (optional: ein Klassenname je Zeile)</code></pre><p>Ordner <code>train/</code> und <code>val/</code> (optional auch <code>test/</code>) mit jeweils eigenen <code>images</code> und <code>masks</code> legen den Split fest; sonst werden die Paare zufällig aufgeteilt. Bilder dürfen bis 4096 × 4096 Pixel groß sein und werden auf den Input skaliert; Masken werden mit nächstem Nachbarn skaliert, damit keine Klassen vermischt werden.</p><p>Wie eine Maske gelesen wird, entscheidet sich einmal für den ganzen Datensatz:</p><div class="table" role="region" aria-labelledby="masken-caption" tabindex="0"><table><caption id="masken-caption">Maskenformate, die Brick DNN liest</caption><thead><tr><th scope="col">Masken</th><th scope="col">Gelesen als</th></tr></thead><tbody><tr><td>Nur Schwarz (0) und Weiß (255)</td><td>Zwei Klassen: Hintergrund und Vordergrund</td></tr><tr><td>Grauwerte oder Palettenindizes unter 32</td><td>Die Klasse jedes Pixels (0, 1, 2 …); 255 markiert Pixel, die nicht zählen</td></tr><tr><td>Höchstens 32 andere Grauwerte, z. B. 0, 128, 255</td><td>Klassen in aufsteigender Reihenfolge</td></tr><tr><td>Farbmasken mit höchstens 32 Farben</td><td>Eine Klasse je Farbe (Reihenfolge aus <code>classes.json</code>, sonst Schwarz zuerst)</td></tr></tbody></table></div><p>Speichern Sie Masken als PNG ohne Glättung. JPEG-Kompression und geglättete Kanten erzeugen Zwischenwerte; solche Masken werden schon beim Hochladen mit Begründung abgelehnt.</p>',
      },
      {
        id: 'netz',
        title: 'Das Netz an die Daten anpassen',
        html: '<p>Wenn Sie Ihren Datensatz wählen, setzt Brick DNN die Form des Inputs und die <code>out_channels</code> der letzten Faltung auf die Zahl der Klassen und sagt, was es geändert hat. Zwei Dinge sollten Sie im Blick behalten:</p><ul class="checks"><li>Jede Pooling-Schicht halbiert Höhe und Breite; bei dreien müssen beide durch 8 teilbar sein. Wird die Ausgabe kleiner als der Input, erklärt die Prüfung den Grund.</li><li>Größere Eingaben kosten mehr: Die CPU-Zeit wächst mit der Zahl der Pixel. Beginnen Sie mit 64 × 64 oder 128 × 128 und vergrößern Sie, sobald das Netz lernt; der Input darf höchstens 512 × 512 groß sein.</li></ul>',
      },
      {
        id: 'einstellungen',
        title: 'Batchgröße und Verlust',
        html: '<p>Jede Schicht hält ihre Ausgabe für jedes Pixel eines Batches im Speicher, daher sind Batches viel kleiner als bei der Klassifikation: 4 bis 16 ist üblich. Ein Batch, der nicht in den Speicher passt, wird schon vor dem Start abgelehnt, zusammen mit der Batchgröße, die passt.</p><p>Der Standardverlust ist die Kreuzentropie jedes Pixels. Bedeckt eine Klasse nur wenige Pixel (ein feiner Riss, eine kleine Zelle), wählen Sie <strong>Kreuzentropie + Dice</strong>, das den Dice-Wert der vorhergesagten Wahrscheinlichkeiten ergänzt, oder die <strong>Gewichtete Kreuzentropie</strong> mit ausgeglichenen Klassengewichten aus den Pixelzahlen. Die Prüfung schlägt sie vor, wenn eine Klasse mehr als 97 % der Pixel ausmacht.</p>',
      },
      {
        id: 'ergebnisse',
        title: 'Die Ergebnisse lesen',
        html: '<p>Nach jeder Epoche misst Brick DNN auf den Validierungsbildern drei Werte:</p><ul class="checks"><li><strong>Mittlere IoU</strong>: je Klasse die Überschneidung von vorhergesagten und echten Pixeln geteilt durch ihre Vereinigung, gemittelt über die Klassen. Das ist die wichtigste Zahl.</li><li><strong>Dice</strong>: die doppelte Überschneidung geteilt durch die Summe beider, ebenso gemittelt; immer mindestens so groß wie die IoU.</li><li><strong>Pixelgenauigkeit</strong>: der Anteil der Pixel mit der richtigen Klasse. Bei viel Hintergrund ist sie auch für ein schwaches Netz hoch, verlassen Sie sich also nicht allein auf sie.</li></ul><p>Die Ansicht des Laufs ergänzt die IoU jeder Klasse für die behaltenen Gewichte und zeigt Vorhersagen für acht Validierungsbilder: das Bild, die echte Maske, die vorhergesagte Maske und eine Überlagerung.</p>',
        image: {
          id: 'segmentation',
          alt: 'Vorhersagen eines auf Shapes trainierten U-Net: zu jedem Validierungsbild das Bild, die echte Maske, die vorhergesagte Maske und eine Überlagerung, mit IoU und Pixelgenauigkeit.',
          caption: 'Vorhergesagte Masken neben den echten, mit der IoU jedes Bildes.',
        },
      },
      {
        id: 'nutzen',
        title: 'Das trainierte Modell nutzen',
        html: `<p>Ein neues Bild sagen Sie auf dem Server über die REST-API oder das MCP-Werkzeug <code>predict</code> vorher: Es wird wie die Trainingsbilder umgewandelt und skaliert, und die Antwort enthält den Anteil jeder Klasse und die Klassenkarte als kleines PNG, dessen Pixelwerte die Klassenindizes sind. Oder Sie laden Gewichte und Skript herunter und rechnen selbst; <code>output.argmax(dim=1)</code> liefert die Klasse jedes Pixels. Einzelheiten stehen in <a href="${DOCS}/training.md#image-segmentation">training.md</a> (englisch).</p>`,
      },
    ],
    examples: ['unet', 'unet_classifier', 'resnet_block'],
  },
  {
    id: 'imbalanced',
    slug: 'unausgewogene-klassen',
    title: 'Unausgewogene Klassen: Gewichte, Focal Loss, Oversampling, Schwelle',
    description:
      'Wenn eine Klasse selten ist: warum die Genauigkeit täuscht und wann Klassengewichte, Focal Loss, Oversampling oder eine andere Entscheidungsschwelle helfen.',
    h1: 'Unausgewogene Klassen: Klassengewichte, Focal Loss, Oversampling und die Entscheidungsschwelle',
    summary:
      'Warum die Genauigkeit täuscht, wenn eine Klasse selten ist, und die vier Stellschrauben dagegen: Gewichte, Focal Loss, Oversampling und die Schwelle.',
    lead: 'Sind 2 % der Zeilen Betrugsfälle, ist ein Modell, das nie Betrug meldet, zu 98 % genau und trotzdem nutzlos. Diese Anleitung zeigt, wie Sie das Problem an den Metriken erkennen und an welcher der vier Stellschrauben Sie zuerst drehen.',
    published: DATE,
    modified: DATE,
    type: 'Article',
    sections: [
      {
        id: 'metriken',
        title: 'Zuerst das Richtige messen',
        html: '<p>Die Genauigkeit zählt jede Zeile gleich, also bestimmt die große Klasse das Ergebnis. Bei unausgewogenen Daten helfen Metriken, die der seltenen Klasse eigenes Gewicht geben:</p><ul class="checks"><li><strong>Balanced Accuracy</strong>, der mittlere Recall über die Klassen: 50 % für ein Modell, das immer die größere von zwei Klassen vorhersagt, egal wie unausgewogen sie sind.</li><li><strong>Recall und Precision</strong> der seltenen Klasse: wie viele ihrer Zeilen gefunden wurden und wie viele Alarme stimmten.</li><li><strong>PR AUC</strong> (Average Precision, bei zwei Klassen), die bei seltenen Positiven mehr sagt als ROC AUC; ein ratendes Modell erreicht den Anteil der Positiven.</li><li><strong>Die Konfusionsmatrix</strong>, die genau zeigt, wo die Fehler liegen.</li></ul><p>Brick DNN meldet sie für jeden abgeschlossenen Klassifikationslauf, auf den Validierungs- und den Testzeilen. Die Bereitschaftsprüfung im Bereich Daten weist auf Klassen hin, die stärker als 3 : 1 unausgewogen sind.</p>',
      },
      {
        id: 'stellschrauben',
        title: 'Vier Stellschrauben',
        html: `<div class="table" role="region" aria-labelledby="stellschrauben-caption" tabindex="0"><table><caption id="stellschrauben-caption">Was jede Stellschraube ändert und wo sie in Brick DNN zu finden ist</caption><thead><tr><th scope="col">Stellschraube</th><th scope="col">Was sie ändert</th><th scope="col">Wo</th></tr></thead><tbody>
<tr><th scope="row">Klassengewichte</th><td>Fehler bei der seltenen Klasse kosten im Training mehr. <code>balanced</code> gewichtet jede Klasse mit <code>n / (k · Anzahl)</code>, sodass jede Klasse insgesamt gleich viel zählt.</td><td>Verlust <em>Gewichtete Kreuzentropie</em> eines Netzes; <code>class_weight</code> von Gradient Boosting und SVMs</td></tr>
<tr><th scope="row">Focal Loss</th><td>Zeilen, die das Netz schon gut einordnet, zählen weniger (<code>focal_gamma</code>, Standard 2); das Training konzentriert sich auf die schweren. Klassengewichte lassen sich als Alpha ergänzen.</td><td>Verlust <em>Focal Loss</em> eines Netzes</td></tr>
<tr><th scope="row">Oversampling</th><td>Wiederholt seltene Zeilen des Trainingsteils (oder lässt häufige weg). Das ändert die Klassenanteile, die das Modell sieht, und verlängert die Epochen.</td><td>Schritt <em>Klassen ausgleichen</em> im Bereich Daten, nur Trainingszeilen</td></tr>
<tr><th scope="row">Entscheidungsschwelle</th><td>Ändert nichts am Training: Sie verschiebt die Wahrscheinlichkeit, ab der eine Zeile als positiv gilt.</td><td>Unter <em>Metriken</em> eines abgeschlossenen Laufs mit zwei Klassen</td></tr>
</tbody></table></div>`,
      },
      {
        id: 'schwelle',
        title: 'Mit der Schwelle beginnen',
        html: '<p>Ein Modell, das die seltenen Zeilen zwar über die häufigen einordnet, ihnen aber selten mehr als 0,5 gibt, braucht nur eine niedrigere Schwelle. Das kostet nichts: kein neues Training, keine kopierten Zeilen. Unter <strong>Metriken</strong> eines abgeschlossenen binären Laufs zeigt ein Schieberegler über die gemessenen Schwellen Precision, Recall, F1 und die Zahl der richtig und falsch Positiven auf den Validierungs- und den Testzeilen. Zwei Schaltflächen springen zur Schwelle mit dem besten F1 und zur höchsten Schwelle mit mindestens 80 % Recall. <strong>Diesen Schwellenwert für Vorhersagen verwenden</strong> speichert sie im Lauf, und seine Vorhersagen verwenden sie von da an.</p><p>Positiv ist die zweite Klasse: bei Textlabels in alphabetischer Reihenfolge, bei Klassenindizes die 1. <code>fraud</code> vor <code>ok</code> macht <code>ok</code> zur positiven Klasse; benennen Sie die Klassen also so, dass die seltene an zweiter Stelle steht, oder verwenden Sie 0 und 1.</p>',
        image: {
          id: 'evaluation',
          alt: 'Die Auswertung eines abgeschlossenen Laufs: eine Konfusionsmatrix mit den echten Klassen in den Zeilen und den vorhergesagten in den Spalten, fast alles auf der Diagonalen.',
          caption: 'Die Konfusionsmatrix eines abgeschlossenen Laufs zeigt, wo die Fehler liegen.',
        },
      },
      {
        id: 'gewichte',
        title: 'Danach Gewichte oder Focal Loss',
        html: '<p>Trennt das Modell die seltene Klasse bei keiner Schwelle gut (eine niedrige PR AUC), ändern Sie, was es lernt. Bei Gradient Boosting und SVMs setzen Sie <code>class_weight</code> auf <code>balanced</code>, wie im <a href="/de/beispiele/gradient-boosting-classifier/">Gradient-Boosting-Beispiel</a>. Bei einem Netz wählen Sie die Gewichtete Kreuzentropie, oder den Focal Loss, wenn es viele leichte und wenige schwere Zeilen gibt. Der Lauf zeigt die Gewichte, die er aus den Klassenzahlen der Trainingszeilen berechnet hat.</p><p>Gewichte und Oversampling schieben beide die vorhergesagten Wahrscheinlichkeiten zur seltenen Klasse; sehen Sie sich danach also die Schwelle noch einmal an. Gewichte und Oversampling zu kombinieren ist selten nötig, denn sie bewirken weitgehend dasselbe.</p>',
      },
      {
        id: 'oversampling',
        title: 'Oversampling, wenn das Modell mehr seltene Zeilen sehen soll',
        html: '<p><em>Klassen ausgleichen</em> im Bereich Daten vervielfacht die seltene Klasse oder dünnt die häufige aus, standardmäßig nur in den Trainingszeilen: Validierung und Test behalten die echten Klassenanteile, ihre Zahlen bleiben also ehrlich. Das passt zu Modellen, deren Training keine Gewichte nimmt, und ist leicht nachzuvollziehen. Der Preis sind längere Epochen und ein Modell, das dieselben seltenen Zeilen viele Male sieht und sie dadurch auswendig lernen kann.</p>',
      },
      {
        id: 'vergleichen',
        title: 'Die Möglichkeiten fair vergleichen',
        html: '<p>Verluste verschiedener Art lassen sich nicht miteinander vergleichen (ein Focal Loss ist kleiner als die Kreuzentropie desselben Modells); vergleichen Sie Läufe daher über ihre Metriken. Eine Hyperparametersuche kann die Verluste für Sie ausprobieren: Geben Sie ihr die Liste <code>cross_entropy</code>, <code>weighted_cross_entropy</code> und <code>focal</code> und ordnen Sie die Versuche nach PR AUC, Balanced Accuracy oder Makro-F1. Eine Suche über den Verlust, die nach dem Verlust selbst geordnet wird, lehnt Brick DNN aus diesem Grund ab.</p>',
        image: {
          id: 'search-results',
          alt: 'Die Versuche einer Suche für ein Gradient-Boosting-Modell, jeder mit Lernrate, Tiefe, Zeilen je Blatt und Klassengewichten sowie kreuzvalidiertem Verlust, Genauigkeit und PR AUC; der beste Versuch ist hervorgehoben.',
          caption:
            'Eine Suche, die auch Klassengewichte ausprobiert, geordnet nach PR AUC statt nach Genauigkeit.',
        },
      },
      {
        id: 'kurz',
        title: 'Kurz gesagt',
        html: '<ol class="numbered"><li>Messen Sie mit Balanced Accuracy, dem Recall der seltenen Klasse, PR AUC und der Konfusionsmatrix.</li><li>Verschieben Sie die Entscheidungsschwelle; oft genügt das.</li><li>Trennen sich die Klassen nicht, nehmen Sie Klassengewichte oder den Focal Loss und stellen die Schwelle danach neu ein.</li><li>Nutzen Sie Oversampling, wenn das Modell keine Gewichte nimmt, oder zum Vergleich mit ihnen, und nur auf den Trainingszeilen.</li></ol>',
      },
    ],
    examples: ['gradient_boosting_classifier', 'svm_classifier', 'voting_ensemble_classifier'],
  },
  {
    id: 'gdpr',
    slug: 'deep-learning-dsgvo-konform',
    title: 'Deep Learning DSGVO-konform: neuronale Netze in Europa trainieren',
    description:
      'KI-Training DSGVO-konform: personenbezogene oder anonyme Daten, wo Brick DNN in der Schweiz speichert und trainiert, Export, Löschung und eigener Server.',
    h1: 'Deep Learning DSGVO-konform: neuronale Netze mit Daten in Europa trainieren',
    summary:
      'Welche Daten die DSGVO betrifft, wo brickdnn.com in der Schweiz speichert und trainiert und was mit Daten geschieht, die personenbezogen bleiben müssen.',
    lead: 'Die DSGVO verbietet nicht, neuronale Netze zu trainieren. Sie fragt, um welche Daten es geht, wohin sie gehen und wer sie verarbeitet. Diese Anleitung beantwortet die Fragen für Brick DNN: welche Daten Sorgfalt brauchen, wo brickdnn.com Daten speichert und in der Schweiz trainiert und wie Sie mit Daten arbeiten, die Ihre Einrichtung nicht verlassen dürfen.',
    published: GDPR_DATE,
    modified: GDPR_DATE,
    type: 'Article',
    sections: [
      {
        id: 'frage',
        title: 'Die erste Frage: Sind es personenbezogene Daten?',
        html: '<p>Die DSGVO gilt für personenbezogene Daten: Angaben über eine Person, die identifiziert ist oder sich identifizieren lässt, direkt oder zusammen mit anderen Angaben. Viel Deep Learning kommt ohne sie aus. Messwerte von Maschinen und Sensoren, Bilder von Werkstoffen, Pflanzen oder Produkten, Simulationsergebnisse, Feldversuche und öffentliche Benchmarks wie MNIST oder CIFAR-10 sagen über niemanden etwas aus. Für solche Daten regelt die DSGVO das Training nicht; offen bleibt nur, welchem Anbieter Sie sie anvertrauen.</p><p>Personenbezogen sind Daten, sobald sich Zeilen oder Bilder Menschen zuordnen lassen: Patientenakten, Umfrageantworten mit Geburtsdatum und Postleitzahl, Fotos mit Gesichtern, Texte mit Namen, Sprachaufnahmen, Logdateien mit Benutzerkennungen. Dann brauchen Sie eine Rechtsgrundlage und einen festgelegten Zweck (Art. 5 und 6 DSGVO), beschränken sich auf das, was der Zweck erfordert, und mit jedem Dienstleister, der die Daten in Ihrem Auftrag verarbeitet, einen Vertrag nach Art. 28. Gesundheits-, genetische und biometrische Daten sind besondere Kategorien mit strengeren Regeln (Art. 9).</p>',
      },
      {
        id: 'anonymisieren',
        title: 'Anonymisiert, pseudonymisiert oder personenbezogen',
        html: `<div class="table" role="region" aria-labelledby="zustaende-caption" tabindex="0"><table><caption id="zustaende-caption">Drei Zustände eines Datensatzes nach der DSGVO</caption><thead><tr><th scope="col">Zustand</th><th scope="col">Was es bedeutet</th><th scope="col">DSGVO</th></tr></thead><tbody>
<tr><th scope="row">Personenbezogen</th><td>Namen, E-Mail-Adressen, Kennungen oder Angaben, die auf eine Person deuten</td><td>Gilt vollständig</td></tr>
<tr><th scope="row">Pseudonymisiert</th><td>Kennungen durch Codes ersetzt; der Schlüssel existiert noch, vielleicht an anderer Stelle</td><td>Weiterhin personenbezogen, gilt vollständig (Erwägungsgrund 26)</td></tr>
<tr><th scope="row">Anonymisiert</th><td>Niemand, auch Sie nicht, kann eine Zeile mit vertretbarem Aufwand einer Person zuordnen</td><td>Gilt nicht (Erwägungsgrund 26)</td></tr>
</tbody></table></div><p>Eine Tabelle zu anonymisieren heißt mehr, als die Namensspalte zu löschen:</p><ul class="checks"><li>Entfernen Sie direkte Kennungen: Namen, Anschriften, E-Mail-Adressen, Telefonnummern, Ausweis- und Aktennummern.</li><li>Vergröbern Sie, was Menschen in Kombination erkennbar macht: das Jahr statt des Geburtsdatums, die Region statt der Postleitzahl, Altersgruppen statt des genauen Alters.</li><li>Lassen Sie Freitextspalten weg; sie enthalten oft Namen.</li><li>Suchen Sie nach seltenen Kombinationen: Eine einzige Zeile mit einem ungewöhnlichen Wert kann genügen, um jemanden zu erkennen.</li><li>Achten Sie bei Bildern auf Gesichter, Namensschilder, eingebrannten Text und Metadaten wie GPS-Positionen in den EXIF-Daten.</li></ul><p>Tun Sie das auf Ihrem eigenen Rechner, bevor Sie etwas hochladen. Sind Sie nicht sicher, ob das Ergebnis anonym ist, behandeln Sie es als personenbezogen und fragen Sie Ihre Datenschutzbeauftragte oder Ihren Datenschutzbeauftragten.</p>`,
      },
      {
        id: 'brickdnn',
        title: 'Was brickdnn.com wo speichert',
        html: `<p>Wenn Sie auf brickdnn.com trainieren, werden Ihre Daten hier gespeichert und verarbeitet. Die <a href="/privacy">Datenschutzerklärung</a> nennt alles mit Zwecken und Speicherfristen.</p><div class="table" role="region" aria-labelledby="wo-caption" tabindex="0"><table><caption id="wo-caption">Wo brickdnn.com Daten speichert und verarbeitet</caption><thead><tr><th scope="col">Daten</th><th scope="col">Wo</th><th scope="col">Anbieter</th></tr></thead><tbody>
<tr><th scope="row">Hochgeladene Dateien, aufbereitete Datensätze, trainierte Gewichte, Trainingsergebnisse</th><td>Server in Genf, Schweiz; dort laufen auch die Trainings</td><td>Infomaniak</td></tr>
<tr><th scope="row">Konten, Netze, Kommentare, die Angaben zu Daten und Trainings (nicht die Dateien)</th><td>Datenbank auf demselben Server in Genf, Schweiz</td><td>Infomaniak</td></tr>
<tr><th scope="row">Technische Fehlerberichte, ohne Benutzername, E-Mail-Adresse, Kontokennung oder hochgeladene Dateien</th><td>EU-Region in Frankfurt am Main</td><td>Sentry</td></tr>
<tr><th scope="row">E-Mails zu Ihrem Konto</th><td>EU</td><td>Brevo</td></tr>
<tr><th scope="row">Zahlung, nur wenn Sie einen Tarif buchen</th><td>Vereinigtes Königreich</td><td>Paddle</td></tr>
</tbody></table></div><p>Für die Schweiz und das Vereinigte Königreich hat die EU-Kommission ein angemessenes Datenschutzniveau festgestellt (Art. 45 DSGVO); Daten dürfen dorthin ohne weitere Garantien übermittelt werden. Für Hosting, Datenbank, E-Mail und Fehlerberichte bestehen Verträge zur Auftragsverarbeitung. Die App setzt keine Cookies, hat keine Analyse und kein Tracking, und Ihre Daten werden nicht verkauft.</p><p>Eine Grenze ist für diese Anleitung wichtig: Nach den Nutzungsbedingungen gehören personenbezogene Daten Dritter und vertrauliche Daten nicht auf brickdnn.com, und die App weist beim Hochladen darauf hin. Der gehostete Dienst ist für Daten ohne Personenbezug und für anonymisierte Daten gedacht. Für alles andere lesen Sie weiter.</p>`,
      },
      {
        id: 'personenbezogen',
        title: 'Wenn die Daten personenbezogen bleiben müssen',
        html: `<p>Brick DNN hilft auch dann, wenn die Daten selbst Ihre Einrichtung nicht verlassen dürfen. Drei Wege, vom geringsten Aufwand zum größten:</p><ol class="numbered"><li><strong>Hier entwerfen, zu Hause trainieren.</strong> Bauen und prüfen Sie das Netz auf brickdnn.com, ohne etwas hochzuladen; dafür brauchen Sie kein Konto. Exportieren Sie das PyTorch-Skript und trainieren Sie es auf den Rechnern Ihrer Einrichtung mit Ihrem eigenen Data Loader; das Skript importiert nichts aus Brick DNN. Für ein Netz mit einem flachen Eingang und einem flachen Ausgang enthält es schon eine Trainingsschleife für CSV-Dateien (<code>python model.py --csv daten.csv</code>).</li><li><strong>Mit einem Stellvertreter arbeiten.</strong> Laden Sie einen anonymisierten Auszug oder synthetische Daten mit denselben Spalten hoch, bereiten Sie sie im Bereich <strong>Daten</strong> auf und trainieren Sie auf dem Server, bis Modell und Einstellungen stimmen. Die Aufbereitung lässt sich als pandas-Skript exportieren, das Sie dann dort auf die echten Daten anwenden, wo sie liegen.</li><li><strong>Brick DNN selbst betreiben.</strong> Brick DNN ist Open Source unter der Apache-Lizenz 2.0. <code>docker compose up -d</code> startet es auf einem Server Ihrer Einrichtung, mit einer lokalen SQLite-Datenbank, solange Sie keine andere einrichten, und ohne Fehlerberichte, solange Sie sie nicht einschalten; für neuronale Netze bauen Sie das Image mit PyTorch (<code>BRICKDNN_INSTALL_TORCH=true</code>). Daten, Trainings und Konten bleiben dann dort, wo dieser Server steht. Die Einrichtung beschreibt der <a href="${REPOSITORY}#quick-start">Quick Start</a> (Englisch).</li></ol>`,
      },
      {
        id: 'kontrolle',
        title: 'Exportieren und löschen',
        html: '<p>Was Sie hochladen, können Sie selbst wieder herausholen und entfernen:</p><ul class="checks"><li><strong>Meine Daten herunterladen</strong> unter <em>Konto</em> liefert eine ZIP-Datei mit Ihren Netzen samt Verlauf, Datenquellen und Aufbereitungen, Datensätzen, Trainings mit Einstellungen und Kennzahlen und den hochgeladenen Dateien selbst (bis 200 MB insgesamt; größere werden aufgelistet). Trainierte Gewichte laden Sie bei jedem abgeschlossenen Training herunter, zusammen mit dem passenden Skript.</li><li>Uploads, Datensätze und Trainings lassen sich jederzeit einzeln löschen. Wenn Sie das Konto löschen, wird alles gelöscht, was dazu gehört, auch die Dateien.</li><li>Uploads, Datensätze und abgeschlossene Trainings, die 180 Tage niemand genutzt hat, werden automatisch gelöscht, Konten nach zwei Jahren ohne Nutzung, mit Erinnerungen per E-Mail vorher.</li><li>Gelöschte Daten können noch eine Weile in Sicherungen liegen, höchstens 5 Wochen, bis sie turnusmäßig gelöscht werden.</li></ul>',
      },
      {
        id: 'checkliste',
        title: 'Eine Checkliste',
        html: '<ol class="numbered"><li>Klären Sie, ob die Daten personenbezogen sind. Wenn nicht, trainieren Sie auf brickdnn.com; die Dateien bleiben auf dem Server in der Schweiz.</li><li>Wenn ja, versuchen Sie, sie vor dem Hochladen auf Ihrem eigenen Rechner zu anonymisieren, und prüfen Sie seltene Kombinationen und Bild-Metadaten.</li><li>Müssen sie personenbezogen bleiben, entwerfen Sie das Modell hier und trainieren den exportierten Code oder Ihre eigene Brick-DNN-Instanz dort, wo die Daten liegen.</li><li>Halten Sie Zweck und Rechtsgrundlage im Verzeichnis der Verarbeitungstätigkeiten Ihrer Einrichtung fest und beziehen Sie den Datenschutz früh ein.</li><li>Laden Sie herunter, was Sie behalten möchten, und löschen Sie Datensätze und Trainings, die Sie nicht mehr brauchen.</li></ol><p>Diese Anleitung beschreibt, wie Brick DNN arbeitet und was die DSGVO allgemein verlangt. Sie ist keine Rechtsberatung für Ihr Vorhaben.</p>',
      },
    ],
    examples: ['tabular_regression', 'cnn_mnist', 'gradient_boosting_classifier'],
  },
  {
    id: 'activations',
    slug: 'aktivierungsfunktionen',
    title: 'Welche Aktivierungsfunktion wann? ReLU, GELU, Sigmoid, Softmax',
    description:
      'Aktivierungsfunktionen im Deep Learning: was ReLU, LeakyReLU, GELU, SiLU, Tanh, Sigmoid und Softmax tun und welche in verdeckte Schichten und an die Ausgabe gehört.',
    h1: 'Welche Aktivierungsfunktion wann? ReLU, GELU, Sigmoid, Softmax und die anderen',
    summary:
      'ReLU für die meisten verdeckten Schichten und an der Ausgabe das, was die Verlustfunktion erwartet, also meist nichts: die Gründe, die Ausnahmen und der klassische Fehler.',
    lead: 'Im Deep Learning entscheidet die Aktivierungsfunktion der verdeckten Schichten selten darüber, ob ein Netz funktioniert: ReLU ist ein solider Standard, und GELU oder SiLU sind in tieferen Netzen einen Versuch wert. An der Ausgabe ist das anders. Dort folgt die Wahl aus der Aufgabe und der Verlustfunktion, und eine falsche kostet Genauigkeit, ohne dass es eine Fehlermeldung gibt.',
    published: BASICS_DATE,
    modified: BASICS_DATE,
    type: 'Article',
    sections: [
      {
        id: 'kurz',
        title: 'Die kurze Antwort',
        html: '<ul class="checks"><li><strong>Verdeckte Schichten:</strong> ReLU. In Transformern GELU, in neueren Faltungsnetzen oft SiLU. Bleiben in einem ReLU-Netz viele Einheiten auf null, LeakyReLU oder ELU.</li><li><strong>Ausgabe einer Regression:</strong> keine Aktivierung. Die letzte Schicht ist eine Linear-Schicht mit einer Ausgabe je vorhergesagter Zahl.</li><li><strong>Ausgabe einer Klassifikation:</strong> ebenfalls keine, solange die Verlustfunktion <code>CrossEntropyLoss</code> oder <code>BCEWithLogitsLoss</code> von PyTorch ist. Beide wenden den Softmax oder das Sigmoid selbst an.</li><li><strong>Sigmoid und Tanh in verdeckten Schichten:</strong> nur in flachen Netzen oder dort, wo eine Schicht begrenzte Werte braucht. In tiefen Stapeln bremsen sie die ersten Schichten aus.</li></ul>',
      },
      {
        id: 'wozu',
        title: 'Wozu eine Aktivierungsfunktion da ist',
        html: '<p>Eine Linear-Schicht bildet gewichtete Summen. Zwei davon hintereinander ergeben wieder nur eine gewichtete Summe. Ohne etwas Nichtlineares dazwischen kann ein Netz beliebiger Tiefe deshalb nicht mehr darstellen als eine einzige Linear-Schicht. Dieses Nichtlineare ist die Aktivierungsfunktion. Sie wird auf jeden Wert einzeln angewendet, hat keine Gewichte zu lernen (PReLU ist die Ausnahme) und lässt die Form des Tensors, wie sie ist.</p><p>Wichtig ist vor allem ihre Steigung, denn mit ihr wird der Gradient auf dem Rückweg durch das Netz multipliziert. Eine Funktion, die über weite Bereiche flach ist, reicht den Schichten davor kaum Gradienten weiter. Die meisten Unterschiede weiter unten gehen darauf zurück.</p>',
      },
      {
        id: 'verdeckt',
        title: 'Verdeckte Schichten: die Kandidaten',
        html: `<div class="table" role="region" aria-labelledby="verdeckt-caption" tabindex="0"><table><caption id="verdeckt-caption">Aktivierungsfunktionen für verdeckte Schichten</caption><thead><tr><th scope="col">Funktion</th><th scope="col">Was sie berechnet</th><th scope="col">Typischer Einsatz</th><th scope="col">Schwachstelle</th></tr></thead><tbody>
<tr><th scope="row">ReLU</th><td><code>max(0, x)</code>: Negative Werte werden zu 0</td><td>Der Standard für MLPs und Faltungsnetze; billig zu berechnen</td><td>Einheiten können für jede Eingabe bei 0 landen (siehe unten)</td></tr>
<tr><th scope="row">LeakyReLU, PReLU</th><td>Wie ReLU, aber negative Werte behalten eine kleine Steigung: bei LeakyReLU standardmäßig 0,01, bei PReLU eine gelernte</td><td>Wenn ReLU-Einheiten absterben</td><td>Eine Einstellung oder ein Parameter mehr; sonst selten besser als ReLU</td></tr>
<tr><th scope="row">ELU, SELU</th><td>Negative Werte laufen glatt gegen eine negative Konstante; SELU ist eine ELU mit zwei festen Skalierungsfaktoren</td><td>ELU als glattere ReLU; SELU für reine vollständig verbundene Netze ohne Normalisierungsschichten</td><td>SELU hält die Aktivierungen nur mit passender Gewichtsinitialisierung und einer eigenen Dropout-Variante stabil</td></tr>
<tr><th scope="row">GELU</th><td><code>x</code> mal dem Anteil einer Standardnormalverteilung, der unter <code>x</code> liegt: glatt, knapp unter 0 leicht negativ</td><td>Der Standard in Transformern (BERT, die GPT-Modelle, Vision Transformer)</td><td>Etwas teurer zu berechnen</td></tr>
<tr><th scope="row">SiLU (Swish)</th><td><code>x · sigmoid(x)</code>, eine Kurve ganz ähnlich der GELU</td><td>Neuere Faltungsnetze wie EfficientNet</td><td>Ebenso</td></tr>
<tr><th scope="row">Tanh</th><td>Staucht auf −1 … 1, um 0 zentriert</td><td>In rekurrenten Schichten (RNN, LSTM, GRU); kleine Netze</td><td>Flach bei großen Eingaben: verschwindende Gradienten in tiefen Stapeln</td></tr>
<tr><th scope="row">Sigmoid</th><td>Staucht auf 0 … 1</td><td>Gates in LSTM und GRU; Ausgaben, die Wahrscheinlichkeiten sind</td><td>Ebenso, und die Steigung liegt nie über 0,25</td></tr>
</tbody></table></div><p>Mish und Softplus gehören zur selben glatten Familie wie GELU und SiLU und werden seltener verwendet. Bei Tabellen und kleinen Bildaufgaben ist der Unterschied zwischen ReLU, GELU und SiLU meist klein gegenüber dem, was die Lernrate oder die Größe des Netzes ausmacht. Stimmen Sie also diese zuerst ab.</p>`,
      },
      {
        id: 'sterbende-relu',
        title: 'Sterbende ReLU-Einheiten (Dying ReLU)',
        html: '<p>Eine ReLU-Einheit mit negativer Eingabe gibt 0 aus und hat die Steigung 0; durch sie fließt also kein Gradient. Werden ihre Gewichte so weit verschoben, dass ihre Eingabe für jede Stichprobe der Trainingsdaten negativ ist (eine zu hohe Lernrate schafft das in einem einzigen Schritt), bekommt sie von da an keinen Gradienten mehr und bleibt für immer bei 0. Einzelne tote Einheiten schaden nicht. Ist ein großer Teil einer Schicht tot, fehlt dem Netz diese Kapazität, und der Verlust verbessert sich schon früh nicht mehr.</p><p>In Brick DNN sehen Sie das unter <strong>Das Modell erklären</strong> bei einem abgeschlossenen Lauf, im Teil <em>Im Inneren des Netzes</em>: Er zeigt für jede Schicht den Anteil der Nullen in der Ausgabe für eine Stichprobe. Eine ReLU setzt für jede einzelne Stichprobe einen guten Teil ihrer Werte auf 0; so arbeitet sie. Ist fast alles 0 und bleibt es bei anderen Stichproben dabei, ist der größte Teil der Schicht tot.</p><p>Was hilft:</p><ul class="checks"><li>eine niedrigere Lernrate, das häufigste Gegenmittel;</li><li>LeakyReLU, ELU oder GELU statt ReLU, die auch für negative Eingaben eine Steigung behalten;</li><li>eine Normalisierungsschicht (BatchNorm oder LayerNorm) vor der Aktivierung, die deren Eingaben um 0 zentriert hält.</li></ul>',
      },
      {
        id: 'verschwindende-gradienten',
        title: 'Verschwindende Gradienten bei Sigmoid und Tanh',
        html: '<p>Die Steigung des Sigmoids beträgt höchstens 0,25, bei x = 0, und fällt nach beiden Seiten gegen 0. Die Backpropagation multipliziert den Gradienten je Schicht mit einer solchen Steigung. Nach vier Sigmoid-Schichten ist das ein Faktor von höchstens 0,25⁴, rund 0,004, noch ohne die Gewichte, und in der Praxis weniger, weil kaum eine Einheit genau bei x = 0 liegt. Die ersten Schichten bekommen dann fast keinen Gradienten und lernen sehr langsam, während die letzten normal trainieren.</p><p>Tanh ist besser: Seine Steigung erreicht bei x = 0 den Wert 1, und seine Ausgabe ist um 0 zentriert. Für große Eingaben ist es aber genauso flach, sodass gesättigte Einheiten ebenfalls fast nichts zurückgeben. Das ist der Hauptgrund, warum ReLU beide in tiefen Netzen abgelöst hat: Ihre Steigung ist für jede positive Eingabe genau 1, wie tief der Stapel auch ist.</p><p>Geblieben sind beide in rekurrenten Schichten. LSTM und GRU verwenden Sigmoid und Tanh im Inneren als Gates und sind so gebaut, dass der Gradient einen Weg an ihnen vorbei hat; diese Aktivierungen fügen Sie nicht selbst hinzu. Wenn Sie Sigmoid oder Tanh zwischen eigenen Schichten möchten, halten Sie den Stapel flach, nehmen Sie eher Tanh, oder setzen Sie Normalisierungsschichten oder Skip-Verbindungen ein.</p>',
      },
      {
        id: 'ausgabe',
        title: 'Die Ausgabeschicht richtet sich nach der Aufgabe',
        html: `<p>Die letzte Schicht ist keine Geschmacksfrage. Sie muss liefern, was die Verlustfunktion erwartet, und die meistgenutzten Verlustfunktionen von PyTorch erwarten rohe Scores (Logits), keine Wahrscheinlichkeiten.</p><div class="table" role="region" aria-labelledby="ausgabe-caption" tabindex="0"><table><caption id="ausgabe-caption">Letzte Schicht, Aktivierung und Verlustfunktion je Aufgabe, in PyTorch</caption><thead><tr><th scope="col">Aufgabe</th><th scope="col">Letzte Schicht</th><th scope="col">Aktivierung danach</th><th scope="col">Verlustfunktion</th></tr></thead><tbody>
<tr><th scope="row">Regression</th><td>Linear, eine Ausgabe je vorhergesagter Zahl</td><td>Keine</td><td><code>MSELoss</code>, <code>L1Loss</code>, <code>HuberLoss</code></td></tr>
<tr><th scope="row">Zwei Klassen</th><td>Linear, 1 Ausgabe</td><td>Keine</td><td><code>BCEWithLogitsLoss</code>, wendet das Sigmoid an</td></tr>
<tr><th scope="row">Mehrere Klassen, genau eine trifft zu</th><td>Linear, eine Ausgabe je Klasse (geht auch bei zwei)</td><td>Keine</td><td><code>CrossEntropyLoss</code>, wendet den Softmax an</td></tr>
<tr><th scope="row">Mehrere Labels können zugleich zutreffen</th><td>Linear, eine Ausgabe je Label</td><td>Keine</td><td><code>BCEWithLogitsLoss</code></td></tr>
<tr><th scope="row">Eine Klasse für jedes Pixel</th><td>Faltung, ein Kanal je Klasse</td><td>Keine</td><td><code>CrossEntropyLoss</code></td></tr>
<tr><th scope="row">Eine Zahl in einem festen Bereich, z. B. 0 … 1</th><td>Linear</td><td>Sigmoid (Tanh für −1 … 1)</td><td><code>MSELoss</code></td></tr>
</tbody></table></div><p>Wahrscheinlichkeiten erhalten Sie von einem trainierten Klassifikator, indem Sie die Funktion bei der Vorhersage auf seine Ausgabe anwenden, außerhalb von Netz und Verlustfunktion: <code>torch.softmax(logits, dim=1)</code> für Klassen, <code>torch.sigmoid(logits)</code> für einen einzelnen Score. Die vorhergesagte Klasse ist in beiden Fällen dieselbe: die mit dem größten Score.</p><p>Brick DNN trainiert jede Klassifikation mit einer Ausgabe je Klasse und einer Verlustfunktion auf rohen Scores (der Kreuzentropie, wenn Sie keine andere wählen), auch bei zwei Klassen. Eine Regression hat eine Ausgabe und keine Aktivierung danach; die Zielgröße wird für das Training standardisiert, eine ReLU oder ein Sigmoid am Ende würde also einen Teil ihres Wertebereichs abschneiden.</p>`,
      },
      {
        id: 'softmax-fehler',
        title: 'Der klassische Fehler: Softmax vor CrossEntropyLoss',
        html: '<p><code>nn.CrossEntropyLoss</code> ist ein Log-Softmax, gefolgt von der negativen Log-Likelihood. Endet das Netz bereits mit <code>nn.Softmax</code>, wird der Softmax zweimal angewendet. Es stürzt nichts ab: Die Formen passen, der Verlust sinkt, die Genauigkeit steigt. Es funktioniert nur schlechter, als es sollte.</p><p>Die Verlustfunktion sieht jetzt Werte zwischen 0 und 1 statt Scores beliebiger Größe, und daraus kann der zweite Softmax keine sichere Vorhersage machen. Bei zehn Klassen gibt selbst eine perfekte Ausgabe, 1 für die richtige Klasse und 0 für alle anderen, der richtigen Klasse nur eine Wahrscheinlichkeit von e / (e + 9), rund 0,23. Der Verlust fällt deshalb nie unter etwa 1,46, während bloßes Raten 2,30 ergibt. Entsprechend klein sind die Gradienten, und das Training ist langsam.</p><p>Dasselbe passiert mit <code>nn.Sigmoid</code> vor <code>BCEWithLogitsLoss</code>. Zusammen gehören diese Paare:</p><ul class="checks"><li>rohe Scores mit <code>CrossEntropyLoss</code> oder <code>BCEWithLogitsLoss</code>, die übliche und numerisch stabilste Wahl;</li><li><code>nn.LogSoftmax</code> mit <code>NLLLoss</code>;</li><li><code>nn.Sigmoid</code> mit <code>BCELoss</code>.</li></ul><p>In Brick DNN erwartet die Verlustfunktion einer Klassifikation immer rohe Scores, das Netz sollte also mit seiner letzten Linear-Schicht enden. Endet es stattdessen mit Softmax oder Sigmoid, weist das Trainingsfenster vor dem Start darauf hin.</p>',
      },
      {
        id: 'architekturen',
        title: 'Was CNNs und Transformer üblicherweise verwenden',
        html: '<p><strong>Faltungsnetze.</strong> Der klassische Block ist eine Faltung mit anschließender ReLU, oft mit einer BatchNorm-Schicht dazwischen. So sind VGG und ResNet gebaut, und ebenso das <a href="/de/beispiele/cnn-mnist/">CNN</a>, der <a href="/de/beispiele/resnet-block/">ResNet-Block</a> und das <a href="/de/beispiele/unet/">U-Net</a> unter den Beispielen hier. Spätere Familien haben das Muster behalten und die Funktion getauscht: EfficientNet verwendet SiLU, ConvNeXt GELU.</p><p><strong>Transformer.</strong> Der Feed-Forward-Teil jedes Blocks hat eine Aktivierung. Der ursprüngliche Transformer verwendete ReLU; BERT, die GPT-Modelle und Vision Transformer verwenden GELU. <code>nn.TransformerEncoderLayer</code> von PyTorch nimmt standardmäßig ReLU, und das erzeugt auch die Schicht <code>TransformerEncoder</code> von Brick DNN. Der Softmax in der Attention gehört zur Schicht und ist nichts, was Sie hinzufügen.</p><p><strong>Rekurrente Netze.</strong> LSTM und GRU bringen Tanh und Sigmoid mit. Danach folgt meist eine Linear-Schicht ohne Aktivierung dazwischen, wie im <a href="/de/beispiele/lstm-sentiment/">LSTM-Beispiel</a>.</p><p><strong>MLPs auf Tabellen.</strong> ReLU. Ist das Netz klein und das Ergebnis enttäuschend, liegt es selten an der Aktivierung; der Vergleich mit <a href="/de/anleitungen/gradient-boosting-oder-mlp/">Gradient Boosting</a> ist der bessere nächste Schritt.</p>',
      },
      {
        id: 'tabelle',
        title: 'Eine kleine Entscheidungstabelle',
        html: `<div class="table" role="region" aria-labelledby="entscheidung-caption" tabindex="0"><table><caption id="entscheidung-caption">Welche Aktivierungsfunktion in welcher Lage</caption><thead><tr><th scope="col">Lage</th><th scope="col">Wahl</th></tr></thead><tbody>
<tr><th scope="row">Verdeckte Schichten eines MLP oder CNN, nichts Besonderes</th><td>ReLU</td></tr>
<tr><th scope="row">Ein Transformer oder ein tiefes Netz, bei dem es auf das letzte bisschen Genauigkeit ankommt</th><td>GELU oder SiLU</td></tr>
<tr><th scope="row">Der Verlust stockt früh, und die meisten Ausgaben einer Schicht sind 0</th><td>Zuerst eine niedrigere Lernrate, dann LeakyReLU oder ELU</td></tr>
<tr><th scope="row">Ein tiefes vollständig verbundenes Netz ohne Normalisierungsschichten</th><td>BatchNorm mit ReLU; SELU nur, wenn Sie ihre Regeln einhalten</td></tr>
<tr><th scope="row">Verdeckte Werte, die zwischen −1 und 1 bleiben müssen</th><td>Tanh, in einem flachen Stapel</td></tr>
<tr><th scope="row">Ausgabe: eine Zahl</th><td>Keine</td></tr>
<tr><th scope="row">Ausgabe: Klassen, mit <code>CrossEntropyLoss</code> oder <code>BCEWithLogitsLoss</code></th><td>Keine</td></tr>
<tr><th scope="row">Ausgabe: ein Wert zwischen 0 und 1, mit <code>MSELoss</code> oder <code>BCELoss</code></th><td>Sigmoid</td></tr>
<tr><th scope="row">Wahrscheinlichkeiten bei der Vorhersage</th><td>Softmax oder Sigmoid auf die Ausgabe anwenden, außerhalb des Netzes</td></tr>
</tbody></table></div>`,
      },
      {
        id: 'ausprobieren',
        title: 'In Brick DNN ausprobieren',
        html: '<p>Die Schicht heißt <code>Activation</code>. Ihre Einstellung <strong>Funktion</strong> bietet ReLU, LeakyReLU (mit der Steigung 0,01), GELU, ELU, SELU, SiLU, Mish, Sigmoid, Tanh, Softplus, Softmax und LogSoftmax. PReLU ist nicht dabei; im exportierten PyTorch-Code ist das eine geänderte Zeile. Ein Vergleich dauert wenige Minuten:</p>',
        steps: [
          {
            title: 'Ein Beispiel öffnen',
            html: 'Öffnen Sie das <a href="/app?example=mlp_mnist">MLP für MNIST im Editor</a>: 784 Eingaben, zwei verdeckte Schichten mit 256 und 128 Einheiten, jeweils gefolgt von ReLU, und eine letzte Linear-Schicht mit 10 Ausgaben ohne Aktivierung danach.',
          },
          {
            title: 'Einmal trainieren',
            html: 'Klicken Sie auf <strong>Trainieren</strong> und wählen Sie <strong>Eingebaut → MNIST</strong>. Legen Sie unter <em>Erweiterte Einstellungen</em> einen <em>Seed</em> fest, damit sich der nächste Lauf nur in der Aktivierung unterscheidet, und trainieren Sie einige Epochen. Das Training braucht ein kostenloses Konto.',
          },
          {
            title: 'Die Funktion ändern',
            html: 'Wählen Sie nacheinander die beiden <code>Activation</code>-Schichten aus und stellen Sie <em>Funktion</em> auf GELU, oder auf Sigmoid, wenn der Unterschied deutlicher sein soll. Das Netz behält seine 235.146 Parameter: Aktivierungen haben keine.',
          },
          {
            title: 'Noch einmal trainieren und vergleichen',
            html: 'Starten Sie einen zweiten Lauf mit denselben Einstellungen. Der Reiter <strong>Läufe</strong> im Trainingsfenster behält beide, jeden mit genau dem Netz, das trainiert wurde, und die Seite <em>Läufe</em> zeigt ihre Kurven in einem Diagramm. Bei zwei verdeckten Schichten sind die Unterschiede klein. Mit der Tiefe werden sie meist größer: Fügen Sie drei oder vier Linear-Schichten mit Sigmoid hinzu und beobachten Sie die ersten Epochen.',
          },
        ],
      },
    ],
    examples: ['mlp_mnist', 'cnn_mnist', 'transformer_classifier'],
  },
  {
    id: 'parameters',
    slug: 'parameter-merkmale-datenpunkte',
    title: 'Parameter, Merkmale, Datenpunkte: Wie groß darf ein Netz sein?',
    description:
      'Wie viele Parameter ein Deep-Learning-Modell hat und woher sie kommen, wie sie mit Merkmalen und Trainingszeilen zusammenhängen und woran Sie Überanpassung erkennen.',
    h1: 'Parameter, Merkmale und Datenpunkte: Wie groß darf ein neuronales Netz sein?',
    summary:
      'Woher die Parameter eines Netzes kommen, was sie mit der Zahl der Merkmale und der Zeilen zu tun haben und was hilft, wenn die Kurven Überanpassung zeigen.',
    lead: 'Ein Netz hat 2.817 Parameter, die Tabelle 300 Zeilen: Ist das zu viel? Eine Formel dafür gibt es nicht, und die Faustregeln, die eine sein wollen, taugen im Deep Learning nur begrenzt. Was hilft: wissen, woher die Parameter kommen, die Kurven lesen und ein paar Gegenmittel kennen.',
    published: BASICS_DATE,
    modified: BASICS_DATE,
    type: 'Article',
    sections: [
      {
        id: 'begriffe',
        title: 'Drei Zahlen, die leicht verwechselt werden',
        html: '<ul class="checks"><li><strong>Merkmale</strong> (Features) sind das, was das Modell je Stichprobe bekommt: die Spalten einer Tabelle ohne die Zielgröße oder die Pixel eines Bildes mal seinen Kanälen. Eine Tabelle mit 8 Spalten für die Vorhersage hat 8 Merkmale, ein Graustufenbild mit 28 × 28 Pixeln 784.</li><li><strong>Datenpunkte</strong> sind die Stichproben: Zeilen, Bilder, Texte. Für das Lernen zählen die Trainingszeilen, also was bleibt, wenn Validierungs- und Testzeilen beiseitegelegt sind.</li><li><strong>Parameter</strong> sind die Gewichte und Bias-Werte, die das Training anpasst. Ihre Zahl folgt aus den Schichten, die Sie gewählt haben.</li></ul><p>Die Merkmale legen die Größe der ersten Schicht fest, die Aufgabe die der letzten. Alles dazwischen entscheiden Sie, und damit auch die Zahl der Parameter. Lernrate, Batchgröße und Ähnliches sind <em>Hyper</em>parameter: Einstellungen des Trainings, kein Teil des Modells.</p>',
      },
      {
        id: 'zaehlen',
        title: 'Woher die Parameter kommen',
        html: `<p>Brick DNN zeigt die Zahl für jede Schicht an. Es lohnt sich trotzdem, sie von Hand nachrechnen zu können, denn die Formeln zeigen, welche Schichten teuer sind.</p><div class="table" role="region" aria-labelledby="zaehlen-caption" tabindex="0"><table><caption id="zaehlen-caption">Parameter gängiger Schichten</caption><thead><tr><th scope="col">Schicht</th><th scope="col">Parameter</th><th scope="col">Beispiel</th></tr></thead><tbody>
<tr><th scope="row">Linear</th><td>Eingaben × Ausgaben, dazu ein Bias-Wert je Ausgabe</td><td>8 → 64: 8 · 64 + 64 = 576</td></tr>
<tr><th scope="row">Conv2d</th><td>Filter × Eingangskanäle × Kernelhöhe × Kernelbreite, dazu ein Bias-Wert je Filter</td><td>32 Filter mit 3 × 3 auf 1 Kanal: 32 · 1 · 9 + 32 = 320</td></tr>
<tr><th scope="row">Embedding</th><td>Vokabulargröße × Embedding-Größe</td><td>10.000 Tokens × 128 = 1.280.000</td></tr>
<tr><th scope="row">BatchNorm</th><td>2 je Kanal: eine Skalierung und eine Verschiebung</td><td>64 Kanäle: 128</td></tr>
<tr><th scope="row">Activation, Pooling, Dropout, Flatten</th><td>0</td><td>–</td></tr>
</tbody></table></div><p>Daraus folgt dreierlei:</p><ul class="checks"><li><strong>Eine Linear-Schicht kostet das Produkt ihrer beiden Seiten.</strong> Das <a href="/de/beispiele/tabular-regression/">Tabellen-Beispiel</a> hat 8 Merkmale und Schichten mit 64, 32 und 1 Einheiten: 576 + 2.080 + 33 Parameter, dazu 128 in seiner BatchNorm-Schicht, zusammen 2.817. Mit verdeckten Schichten von 128 und 64 Einheiten wären es 9.729.</li><li><strong>Einer Faltung ist die Bildgröße gleich.</strong> Ihre Parameterzahl hängt nur von den Kanälen und vom Kernel ab. Teuer ist in einem kleinen CNN die erste Linear-Schicht nach <code>Flatten</code>: Im <a href="/de/beispiele/cnn-mnist/">CNN-Beispiel</a> sind 64 Kanäle mit je 7 × 7 Werten 3.136 Eingaben, und die Linear-Schicht von dort auf 128 Einheiten hat 401.536 Parameter, 95 % der 421.642 des ganzen Netzes.</li><li><strong>Ein Embedding wächst mit dem Vokabular.</strong> Im <a href="/de/beispiele/transformer-classifier/">Transformer-Beispiel</a> enthält die Tabelle aus 10.000 × 128 Werten 1.280.000 der 1.611.268 Parameter, und jede ihrer Zeilen lernt nur aus den Stichproben, in denen ihr Token vorkommt.</li></ul>`,
      },
      {
        id: 'zusammenhang',
        title: 'Wie die drei zusammenhängen',
        html: '<p><strong>Merkmale und Parameter.</strong> Jedes Merkmal bringt ein Gewicht je Einheit der ersten Schicht mit. Bei 784 Pixeln und 256 Einheiten hat allein die erste Schicht des <a href="/de/beispiele/mlp-mnist/">MLP-Beispiels</a> 200.960 seiner 235.146 Parameter. Mehr Merkmale bedeuten auch mehr Möglichkeiten, in den Trainingszeilen ein Muster zu finden, das es in neuen Daten nicht gibt. Eine Spalte ohne Information ist also nicht umsonst.</p><p><strong>Zeilen und Parameter.</strong> Jede Trainingszeile ist ein Hinweis darauf, wie die Parameter aussehen sollten. Ein Modell mit weit mehr einstellbaren Zahlen als Hinweisen kann die Trainingszeilen exakt wiedergeben, samt Rauschen, und zwar auf viele verschiedene Arten, und nichts in diesen Zeilen sagt, welche davon für neue Daten stimmt. Das ist Überanpassung (Overfitting). Hat das Modell weit weniger Parameter, als die Aufgabe braucht, kann es nicht einmal die Trainingszeilen wiedergeben. Das ist Unteranpassung (Underfitting).</p><p><strong>Zeilen und Merkmale.</strong> Je mehr Merkmale es gibt, desto mehr Zeilen braucht es, um ihre Kombinationen abzudecken. Zehn Ja/Nein-Merkmale erlauben schon 1.024 verschiedene Zeilen.</p>',
      },
      {
        id: 'faustregeln',
        title: 'Faustregeln und wo sie enden',
        html: '<p>Die Statistik kennt Regeln wie „zehn Beobachtungen je Koeffizient“. Sie stammen aus der linearen und der logistischen Regression, wo jeder Parameter ein Koeffizient mit eigener Bedeutung ist, und dort sind sie sinnvoll.</p><p>Neuronale Netze brechen sie ständig. Das <a href="/de/beispiele/cnn-mnist/">CNN-Beispiel</a> hat 421.642 Parameter und lernt aus 48.000 MNIST-Bildern, fast neun Parameter je Bild, und erkennt trotzdem etwa 99 % der 10.000 Testbilder richtig. Netze mit mehr Parametern als Trainingsstichproben können gut verallgemeinern, wenn es für die Schwierigkeit der Aufgabe genug Daten gibt, wenn die Architektur zu den Daten passt (eine Faltung verwendet dieselben wenigen Gewichte an jeder Position) und wenn das Training regularisiert ist: durch Dropout, durch Weight Decay oder schlicht durch rechtzeitiges Aufhören. Ein Verhältnis von Parametern zu Zeilen kann Ihnen deshalb nicht sagen, dass ein Netz zu groß ist. Es kann Ihnen sagen, wo Sie genauer hinsehen sollten.</p><p>Was sich ohne falsche Genauigkeit sagen lässt:</p><ul class="checks"><li><strong>Eine Tabelle mit wenigen hundert Zeilen:</strong> Nehmen Sie ein kleines Netz, ein oder zwei verdeckte Schichten mit 8 bis 32 Einheiten, oder ein klassisches Modell. Gradient Boosting, ein Gauß-Prozess oder eine Support Vector Machine ist hier oft die bessere Wahl und braucht weniger Abstimmung; siehe <a href="/de/anleitungen/gradient-boosting-oder-mlp/">Gradient Boosting oder MLP</a>.</li><li><strong>Eine Tabelle mit Tausenden Zeilen oder mehr:</strong> Ein MLP mit einigen tausend bis einigen zehntausend Parametern ist ein vernünftiger Anfang. Vergrößern Sie es nur, solange das Validierungsergebnis besser wird.</li><li><strong>Bilder und Text:</strong> Die Parameterzahl sagt wenig. Wichtiger ist, wie viele Stichproben es je Klasse gibt, wie unterschiedlich sie sind und wie gut die Architektur passt.</li><li><strong>In jedem Fall:</strong> Halten Sie Validierungszeilen zurück, und glauben Sie den Kurven eher als dem Verhältnis.</li></ul><p>Genau so verwendet Brick DNN das Verhältnis. Hat ein Netz mehr als zehnmal so viele Parameter, wie seine Tabelle Trainingszeilen hat, zeigt das Trainingsfenster einen kurzen Tipp. Er ist ein Anlass, auf die Kurven zu sehen und ein Vergleichsmodell zu probieren, kein Fehler.</p>',
      },
      {
        id: 'kurven',
        title: 'Was die Kurven zeigen',
        html: `<p>Nach jeder Epoche misst Brick DNN den Verlust auf den Trainingszeilen und auf den Validierungszeilen, aus denen das Netz nie lernt. Beide Kurven zusammen sagen mehr als jede Zählung.</p><div class="table" role="region" aria-labelledby="kurven-caption" tabindex="0"><table><caption id="kurven-caption">Trainings- und Validierungsverlust lesen</caption><thead><tr><th scope="col">Was Sie sehen</th><th scope="col">Was es bedeutet</th><th scope="col">Was Sie zuerst versuchen</th></tr></thead><tbody>
<tr><th scope="row">Beide fallen und fallen am Ende noch</th><td>Das Training ist nicht fertig</td><td>Mehr Epochen</td></tr>
<tr><th scope="row">Der Trainingsverlust fällt weiter, der Validierungsverlust wird flach oder steigt wieder</th><td>Überanpassung: Das Netz lernt die Trainingszeilen auswendig</td><td>Die beste Epoche behalten, dann weniger Kapazität oder mehr Regularisierung</td></tr>
<tr><th scope="row">Beide bleiben hoch und dicht beieinander</th><td>Unteranpassung oder eine unpassende Lernrate</td><td>Ein größeres Netz, eine andere Lernrate, bessere Merkmale</td></tr>
<tr><th scope="row">Der Validierungsverlust springt von Epoche zu Epoche</th><td>Zu wenige Validierungszeilen oder eine zu hohe Lernrate</td><td>Mehr Validierungszeilen, eine niedrigere Lernrate</td></tr>
</tbody></table></div><p>Ein Abstand zwischen den beiden Kurven ist normal, denn die Trainingszeilen hat das Netz gesehen. Entscheidend ist die Richtung der Validierungskurve. Bei wenigen Zeilen ist Vorsicht angebracht: Bei 40 Validierungszeilen macht eine einzige Zeile 2,5 Prozentpunkte Genauigkeit aus.</p>`,
        image: {
          id: 'training-curves',
          alt: 'Verlust und Genauigkeit je Epoche eines kleinen CNN auf MNIST: Der Trainingsverlust fällt weiter, der Validierungsverlust bleibt ab der dritten Epoche etwa gleich.',
          caption:
            'Ab Epoche 3 fällt der Trainingsverlust noch, der Validierungsverlust nicht mehr: Weitere Epochen würden diesem Netz nicht helfen.',
        },
      },
      {
        id: 'ueberanpassung',
        title: 'Was gegen Überanpassung hilft',
        html: '<p>Ungefähr nach Aufwand geordnet:</p><ol class="numbered"><li><strong>Die beste Epoche behalten und früh stoppen.</strong> Brick DNN behält die Gewichte der Epoche mit dem niedrigsten Validierungsverlust, solange Sie das nicht abschalten; ein Lauf, der erst spät überanpasst, schadet also nicht. <em>Early Stopping</em> in den Trainingseinstellungen beendet einen Lauf, wenn sich der Validierungsverlust so viele Epochen lang nicht verbessert hat, wie Sie angeben.</li><li><strong>Weniger Kapazität.</strong> Entfernen Sie eine Schicht oder halbieren Sie die verdeckten Schichten und trainieren Sie erneut. Bleibt das Validierungsergebnis gleich, ist das kleinere Netz das bessere.</li><li><strong>Dropout.</strong> Eine <code>Dropout</code>-Schicht nach einer verdeckten Aktivierung setzt in jedem Trainingsschritt einen zufälligen Anteil der Werte auf 0. Das hindert Einheiten daran, sich aufeinander zu verlassen. Üblich sind Anteile zwischen 0,1 und 0,5.</li><li><strong>Weight Decay.</strong> Es steht unter <em>Erweiterte Einstellungen</em> und zieht die Gewichte in jedem Schritt ein wenig gegen 0. Probieren Sie Werte zwischen 0,0001 und 0,01, am besten mit dem Optimierer AdamW, oder lassen Sie eine Hyperparametersuche einen Wert wählen.</li><li><strong>Mehr Daten.</strong> Mehr Zeilen helfen mehr als alles davor, wenn sie zu bekommen sind. Weniger und bessere Merkmale helfen ebenfalls: Lassen Sie Spalten weg, die keine Rolle spielen können.</li><li><strong>Ein klassisches Vergleichsmodell.</strong> Trainieren Sie ein <a href="/de/beispiele/gradient-boosting-classifier/">Gradient Boosting</a> auf demselben Split. Ist es auf einer kleinen Tabelle besser, nehmen Sie es.</li></ol><p>Unteranpassung ist der einfachere Fall: mehr Einheiten oder Schichten, mehr Epochen, ein Blick auf die Lernrate und weniger Dropout oder Weight Decay, falls Sie welches eingesetzt hatten.</p>',
      },
      {
        id: 'wo',
        title: 'Wo Brick DNN die Zahlen zeigt',
        html: '<ul class="checks"><li><strong>Parameter:</strong> Jede Schicht auf der Arbeitsfläche zeigt ihre eigene Zahl. Ist keine Schicht ausgewählt, zeigt der Inspektor die <em>Trainierbaren Parameter</em> des ganzen Netzes, und ein Lauf zeigt sie in der Kachel <em>Parameter</em>.</li><li><strong>Merkmale und Zeilen:</strong> Die Liste Ihrer Datensätze im Trainingsfenster nennt beides für jeden Datensatz, und die Prüfung wiederholt es, sobald Sie einen wählen, zum Beispiel „Klassifikation in 2 Klassen aus 18 Merkmalen (900 Zeilen)“. Bei einem Datensatz aus dem Bereich Daten sagt sie außerdem, wie viele Zeilen in Training, Validierung und Test liegen.</li><li><strong>Kurven:</strong> <em>Trainingskurven</em> in der Ansicht eines Laufs, mit den Zahlen jeder Epoche unter <em>Werte pro Epoche</em> und der Epoche, deren Gewichte behalten wurden.</li></ul>',
        image: {
          id: 'training-setup',
          alt: 'Das Trainingsfenster mit einem aufbereiteten Datensatz aus 900 Zeilen und 18 Merkmalen, aufgeteilt in 630 Trainings-, 135 Validierungs- und 135 Testzeilen, und der Prüfung „Ready to train“.',
          caption: 'Der Datensatz und die Prüfung nennen Zeilen, Merkmale und Split.',
        },
      },
    ],
    examples: ['tabular_regression', 'mlp_mnist', 'gradient_boosting_classifier'],
  },
];

export const guidesDe = { index, guides };
