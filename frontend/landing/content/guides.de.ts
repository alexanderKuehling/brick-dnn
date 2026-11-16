/**
 * Die Anleitungen auf Deutsch. Alles, was sie sagen, muss für die App stimmen (docs/training.md,
 * docs/data-preparation.md, docs/explain.md); die Screenshots sind die aus images/. Die
 * Bezeichnungen der Oberfläche sind die der deutschen App (src/i18n/locales/de).
 */
import type { Guide, GuideIndex } from './types.ts';

const DOCS = 'https://github.com/alexanderKuehling/brick-dnn/blob/main/docs';
const DATE = '2026-11-16';

const index: GuideIndex = {
  title: 'Anleitungen: neuronale Netze, Tabellendaten, Segmentierung – Brick DNN',
  description:
    'Praxisnahe Anleitungen: neuronales Netz ohne Code bauen und als PyTorch exportieren, Gradient Boosting oder MLP, U-Net-Segmentierung, unausgewogene Klassen.',
  h1: 'Anleitungen',
  intro:
    'Schritt-für-Schritt-Anleitungen zum Bauen, Trainieren und Bewerten von Modellen mit Brick DNN, und zu den Entscheidungen unterwegs: welches Modell für eine Tabelle, wie man Bilder segmentiert, was zu tun ist, wenn eine Klasse selten ist.',
};

const guides: Guide[] = [
  {
    id: 'no-code',
    slug: 'neuronales-netz-ohne-code',
    title: 'Neuronales Netz ohne Code bauen und als PyTorch-Code exportieren',
    description:
      'Ein neuronales Netz im Browser aus Schichten zusammensetzen, jede Tensorform prüfen lassen und sauberen PyTorch-Code herunterladen, der ohne Brick DNN läuft.',
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
      'Ein U-Net zur Bildsegmentierung trainieren: Bilder und Masken vorbereiten, das Netz prüfen, Batchgröße und Verlust wählen, mittlere IoU und Dice lesen.',
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
];

export const guidesDe = { index, guides };
