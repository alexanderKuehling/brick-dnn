/**
 * Die Texte der Beispielseiten auf Deutsch. Die Fakten (Schichten, Formen, Parameterzahlen,
 * Code) kommen aus examples/examples.json; was hier steht, muss dazu passen (landing.test.ts
 * prüft die Zahlen, die im Text vorkommen).
 */
import type { ExampleCopy, ExampleGallery } from './types.ts';

const gallery: ExampleGallery = {
  title: 'Beispiele für neuronale Netze in PyTorch und scikit-learn – Brick DNN',
  description:
    'Mitgelieferte Beispiele: MLP, CNN, ResNet-Block, U-Net, LSTM, Transformer, Gradient Boosting, SVM, Gauß-Prozesse und Ensembles, jeweils mit Diagramm und Code.',
  h1: 'Beispielnetze mit Diagramm, Tensorformen und Code',
  intro:
    'Jedes Beispiel hier ist ein Netz, das in Brick DNN mitgeliefert wird. Seine Seite zeigt, wie die Schichten verbunden sind, welche Ausgabeform und wie viele Parameter jede Schicht hat, und das vollständige Skript, das der Editor erzeugt: PyTorch für neuronale Netze, scikit-learn für klassische Modelle. Jedes Beispiel lässt sich im Editor öffnen, ändern, prüfen und trainieren.',
  groups: [
    {
      title: 'Bilder',
      intro:
        'Vom mehrschichtigen Perzeptron auf flachgeklopften Ziffern bis zum U-Net, das jedem Pixel eine Klasse gibt.',
      ids: ['mlp_mnist', 'cnn_mnist', 'resnet_block', 'unet', 'unet_classifier'],
    },
    {
      title: 'Text',
      intro:
        'Folgen von Token-IDs, klassifiziert von einem rekurrenten Netz oder einem Transformer-Encoder.',
      ids: ['lstm_sentiment', 'transformer_classifier'],
    },
    {
      title: 'Tabellen und gemischte Eingaben',
      intro:
        'Neuronale Netze für Zeilen aus Zahlen, ein Netz mit zwei Eingaben und eine Mischung aus MLP und Gradient Boosting.',
      ids: ['tabular_regression', 'multi_input', 'hybrid_mlp_gradient_boosting'],
    },
    {
      title: 'Klassische Modelle',
      intro:
        'Modelle von scikit-learn für Tabellendaten, aus Bausteinen gebaut wie die Netze: Gradient Boosting, eine Support Vector Machine und Gauß-Prozesse mit Kerneln.',
      ids: ['gradient_boosting_classifier', 'svm_classifier', 'gp_regression', 'periodic_gp'],
    },
    {
      title: 'Ensembles',
      intro: 'Mehrere klassische Modelle, kombiniert durch Abstimmung (Voting) oder Stacking.',
      ids: ['voting_ensemble_classifier', 'stacking_ensemble_regression'],
    },
  ],
  note: 'Diagramme, Formen, Parameterzahlen und Code auf diesen Seiten entstehen aus den Beispielen selbst, mit derselben Formprüfung und demselben Code-Generator wie im Editor.',
};

const examples: ExampleCopy[] = [
  {
    id: 'mlp_mnist',
    name: 'MLP für MNIST',
    title: 'MLP für MNIST in PyTorch – Beispiel für ein mehrschichtiges Perzeptron',
    description:
      'Ein mehrschichtiges Perzeptron für MNIST-Ziffern in PyTorch: 784 Eingaben, zwei verdeckte Schichten, 235.146 Parameter, mit Diagramm, Formen und Code.',
    h1: 'MLP für MNIST in PyTorch',
    task: 'classification',
    summary:
      'Ein mehrschichtiges Perzeptron: 784 Pixelwerte, zwei verdeckte Schichten, zehn Klassen-Scores.',
    lead: 'Das klassische erste Netz für handgeschriebene Ziffern: Jedes 28×28-Bild wird zu 784 Zahlen ausgerollt, zwei vollständig verbundene Schichten mit ReLU lernen daraus, und eine letzte Schicht liefert für jede der zehn Ziffern einen Score.',
    what: [
      'Der <code>Input</code> nimmt einen flachen Vektor aus 784 Werten, einen je Pixel. <code>Linear</code>-Schichten verbinden jeden Wert mit jeder Einheit: Von 784 Eingaben auf 256 Einheiten sind das 784 × 256 Gewichte plus 256 Bias-Werte, also 200.960 Parameter und damit fast alle der 235.146 des Netzes. ReLU nach jeder verdeckten Schicht macht das Netz nichtlinear, und <code>Dropout</code> mit p = 0,2 schaltet beim Training zufällig ein Fünftel der Einheiten der ersten Schicht ab, was Überanpassung bremst.',
      'Die letzte <code>Linear</code>-Schicht gibt zehn rohe Scores (Logits) aus, einen je Ziffer. Ein Softmax am Ende fehlt mit Absicht: Die Kreuzentropie, mit der trainiert wird, erwartet rohe Scores und wendet den Softmax selbst an.',
    ],
    when: [
      'Um zu verstehen, wie sich ein dichtes Netz, seine Parameterzahl und sein Training verhalten, bevor es um Faltungen geht.',
      'Als Vergleichsbasis für Bilddaten: Ein CNN wie das <a href="/de/beispiele/cnn-mnist/">kleine CNN für MNIST</a> sollte mit weniger Parametern besser abschneiden, weil es die Nachbarschaft der Pixel nutzt.',
      'Für jede Eingabe, die schon ein flacher Merkmalsvektor ist: Form des Inputs und Größe der letzten Schicht anpassen, fertig.',
    ],
    train:
      'Klicken Sie auf <strong>Trainieren</strong> und wählen Sie <strong>MNIST</strong>: Der eingebaute Datensatz (70.000 Bilder, einmalig auf den Server geladen) passt zu diesem Input, weil MNIST sowohl als <code>784</code> flache Werte als auch als <code>1, 28, 28</code> angenommen wird. Brick DNN hält 20 % der Trainingsbilder zur Validierung zurück, testet auf den offiziellen 10.000 Testbildern und zeigt nach jeder Epoche Verlust und Genauigkeit. Weil das Netz einen flachen Vektor nimmt und einen ausgibt, enthält das exportierte Skript zusätzlich eine Trainingsschleife für CSV-Dateien.',
    related: ['cnn_mnist', 'tabular_regression', 'multi_input'],
  },
  {
    id: 'cnn_mnist',
    name: 'Kleines CNN für MNIST',
    title: 'CNN für MNIST in PyTorch – Beispiel für ein kleines Faltungsnetz',
    description:
      'Ein kleines Faltungsnetz (CNN) für MNIST in PyTorch: zwei Conv2d-Blöcke mit Max-Pooling und ein dichter Kopf, etwa 99 % Testgenauigkeit, mit Code und Formen.',
    h1: 'CNN für MNIST in PyTorch',
    task: 'classification',
    summary:
      'Zwei Faltungsblöcke und ein Klassifikationskopf: etwa 99 % Testgenauigkeit auf MNIST.',
    lead: 'Ein kleines Faltungsnetz, das auf handgeschriebenen Ziffern etwa 99 % Testgenauigkeit erreicht: Zwei Blöcke aus Faltung, ReLU und Max-Pooling erkennen Striche und Formen, ein dichter Kopf macht daraus zehn Klassen-Scores.',
    what: [
      'Jede <code>Conv2d</code>-Schicht schiebt 3×3-Filter über das Bild. Die erste hat 32 Filter, aus dem Bild 1 × 28 × 28 werden also 32 Merkmalskarten 32 × 28 × 28; <code>padding=same</code> erhält Höhe und Breite. <code>MaxPool2d</code> mit einem 2×2-Fenster halbiert sie auf 32 × 14 × 14. Der zweite Block wiederholt das mit 64 Filtern und endet bei 64 × 7 × 7.',
      '<code>Flatten</code> macht aus diesen 3.136 Werten einen Vektor für den Kopf: eine <code>Linear</code>-Schicht mit 128 Einheiten, ReLU, <code>Dropout</code> mit p = 0,25 und die letzte <code>Linear</code>-Schicht mit zehn Ausgaben. Die erste dichte Schicht enthält 401.536 der 421.642 Parameter, die beiden Faltungen zusammen nur 18.816, weil ein Filter für das ganze Bild gilt.',
    ],
    when: [
      'Für kleine Graustufen- oder Farbbilder mit je einem Label: Ziffern, Kleidung (Fashion-MNIST), Symbole, kleine Produktfotos.',
      'Als erstes CNN in einer Lehrveranstaltung: Es zeigt, wie Faltung und Pooling die Form verändern und wie wenige Parameter eine Faltung im Vergleich zu einer dichten Schicht braucht.',
      'Als Ausgangspunkt für eigene Bilder: Für Farbbilder setzen Sie den Input auf <code>3, 64, 64</code>, bei größeren Bildern ergänzen Sie einen dritten Block.',
    ],
    train:
      'Wählen Sie im Trainingsbereich <strong>MNIST</strong> oder <strong>Fashion-MNIST</strong>; beide passen zum Input <code>1, 28, 28</code> und haben zehn Klassen. Auch ein Ordner mit eigenen Bildern funktioniert (eine ZIP-Datei mit einem Ordner je Klasse): Die Bilder werden in die Kanäle des Inputs umgewandelt und auf seine Höhe und Breite skaliert. Die Ansicht des Laufs zeigt Verlust und Genauigkeit je Epoche, am Ende die Konfusionsmatrix sowie Precision und Recall je Klasse auf den Testbildern.',
    related: ['mlp_mnist', 'resnet_block', 'unet_classifier'],
  },
  {
    id: 'resnet_block',
    name: 'Residual-CNN für CIFAR-10',
    title: 'ResNet-Block für CIFAR-10 in PyTorch – Beispiel mit Skip-Verbindung',
    description:
      'Ein Residualblock mit Batch-Normalisierung und Skip-Verbindung (Add) für die Farbbilder von CIFAR-10, in PyTorch: 76.490 Parameter, Diagramm, Formen und Code.',
    h1: 'ResNet-Block für CIFAR-10 in PyTorch',
    task: 'classification',
    summary:
      'Ein Block im Stil von ResNet mit Skip-Verbindung über Add, danach globales Average Pooling.',
    lead: 'Der Grundbaustein von ResNet in einem kleinen Netz für 32×32-Farbfotos: Zwei Faltungen lernen eine Korrektur, die eine <code>Add</code>-Schicht auf ihre eigene Eingabe addiert. So nehmen Signal und Gradient auch den kurzen Weg außen herum.',
    what: [
      'Ein Stamm aus <code>Conv2d</code> (64 Filter 3×3, ohne Bias), <code>BatchNorm2d</code> und ReLU macht aus dem Bild 3 × 32 × 32 die Merkmalskarten 64 × 32 × 32. Der Residualblock folgt mit zwei weiteren Faltungen samt Batch-Normalisierung. Die Faltungen haben keinen Bias, weil die Batch-Normalisierung direkt danach ohnehin eine eigene Verschiebung lernt.',
      '<code>Add</code> addiert die Ausgabe des Blocks und die des Stamms elementweise; deshalb müssen beide dieselbe Form haben, 64 × 32 × 32, was Brick DNN schon beim Bauen prüft. Nach einer ReLU mittelt <code>AdaptiveAvgPool2d</code> jede Merkmalskarte zu einem Wert (64 × 1 × 1), und eine <code>Linear</code>-Schicht mit 650 Parametern liefert die zehn Klassen-Scores. Weil globales Pooling statt einer großen dichten Schicht verwendet wird, bleibt das ganze Netz bei 76.490 Parametern.',
    ],
    when: [
      'Wenn ein einfacher Stapel von Faltungen mit weiteren Schichten nicht mehr besser wird: Skip-Verbindungen machen tiefe Netze erst trainierbar.',
      'Um zu sehen, wie eine Merge-Schicht arbeitet: Verbinden Sie Tensoren verschiedener Form mit <code>Add</code> und lesen Sie die Meldung.',
      'Als Grundlage für ein tieferes Netz: Block kopieren, mehrere hintereinander schalten und dazwischen eine Faltung mit Stride 2 einsetzen, die die Auflösung halbiert.',
    ],
    train:
      'Wählen Sie <strong>CIFAR-10</strong> (60.000 Farbfotos in zehn Klassen, einmalig auf den Server geladen); der Datensatz passt so, wie er ist, zum Input <code>3, 32, 32</code>. Batch-Normalisierung verhält sich im Training anders als in der Auswertung; im exportierten Modul regeln das <code>model.train()</code> und <code>model.eval()</code> von PyTorch.',
    related: ['cnn_mnist', 'unet', 'unet_classifier'],
  },
  {
    id: 'unet',
    name: 'U-Net zur Segmentierung',
    title: 'U-Net zur Bildsegmentierung in PyTorch – Beispiel mit Code',
    description:
      'Ein U-Net zur semantischen Segmentierung in PyTorch: Encoder, Decoder mit ConvTranspose2d und Skip-Verbindungen, ein Klassen-Score je Pixel, mit Code.',
    h1: 'U-Net zur Bildsegmentierung in PyTorch',
    task: 'segmentation',
    summary:
      'Encoder, Decoder und Skip-Verbindungen: eine Klasse für jedes Pixel, trainiert auf dem eingebauten Datensatz Shapes.',
    lead: 'Ein U-Net gibt jedem Pixel eines Bildes eine Klasse. Sein Encoder halbiert das Bild dreimal, um zu erfassen, was darauf zu sehen ist, sein Decoder vergrößert es wieder auf die volle Größe, und Skip-Verbindungen reichen die feinen Details jeder Ebene an den Decoder weiter.',
    what: [
      'Der Encoder hat drei Ebenen aus je zwei 3×3-Faltungen mit ReLU, mit 32, 64 und 128 Filtern und jeweils einem <code>MaxPool2d</code> danach: Aus dem Bild 3 × 64 × 64 wird 128 × 8 × 8. Ein Engpass aus zwei Faltungen mit 256 Filtern arbeitet auf dieser groben Ansicht, in der jeder Wert einen großen Teil des Bildes sieht.',
      'Jede Ebene des Decoders beginnt mit <code>ConvTranspose2d</code>, einer gelernten Vergrößerung, die Höhe und Breite verdoppelt. <code>Concat</code> hängt das Ergebnis entlang der Kanäle an die gleich große Ausgabe des Encoders (zum Beispiel 128 + 128 Kanäle bei 16 × 16), sodass der Decoder die Details zurückbekommt, die das Pooling verworfen hat. Eine letzte 1×1-Faltung liefert vier Kanäle, also einen Score je Klasse für jedes der 64 × 64 Pixel.',
      'Alle Faltungen verwenden <code>padding=same</code>; nur Pooling und Vergrößerung ändern die Größe. Höhe und Breite des Inputs müssen deshalb durch 8 teilbar sein. Brick DNN prüft das und sagt, was zu ändern ist.',
    ],
    when: [
      'Für semantische Segmentierung, also eine Klasse je Pixel: Gewebe in Mikroskopbildern, Straßen und Gebäude in Luftbildern, Fehler auf Oberflächen, der Hintergrund hinter einem Objekt.',
      'Bei wenigen Trainingsbildern: Das U-Net wurde für kleine medizinische Datensätze entworfen, und dank der Skip-Verbindungen lernt es aus ein paar hundert Masken scharfe Umrisse.',
      'Für zwei Klassen (Vordergrund und Hintergrund) bekommt die letzte Faltung einen Kanal, bei mehr Klassen einen Kanal je Klasse.',
    ],
    train:
      'Klicken Sie auf <strong>Trainieren</strong>, wählen Sie <strong>Eingebaut → Shapes</strong> und starten Sie: Der erzeugte Datensatz (400 Trainings- und 100 Testbilder mit Quadraten, Kreisen und Dreiecken samt Maske) passt so, wie er ist, und zehn Epochen erreichen auf einer CPU etwa 80 % mittlere IoU. Eigene Daten laden Sie als ZIP-Datei mit den Ordnern <code>images/</code> und <code>masks/</code> hoch. Jede Epoche meldet mittlere IoU, Dice und Pixelgenauigkeit, und der Lauf zeigt die vorhergesagten Masken neben den echten. Die <a href="/de/anleitungen/bildsegmentierung-u-net/">Anleitung zur Segmentierung</a> führt Schritt für Schritt hindurch.',
    related: ['unet_classifier', 'resnet_block', 'cnn_mnist'],
  },
  {
    id: 'unet_classifier',
    name: 'U-Net als Klassifikator',
    title: 'U-Net als Klassifikator für Fashion-MNIST in PyTorch – Beispiel',
    description:
      'Ein U-Net als Bildklassifikator: Encoder, Decoder und Skip-Verbindungen, danach globales Pooling und eine Linear-Schicht für Fashion-MNIST, in PyTorch.',
    h1: 'U-Net als Bildklassifikator in PyTorch',
    task: 'classification',
    summary:
      'Die Architektur des U-Net mit einem Klassifikationskopf: globales Pooling und eine Linear-Schicht ergeben zehn Klassen-Scores.',
    lead: 'Encoder, Decoder und Skip-Verbindungen wie beim U-Net zur Segmentierung, aber am Ende steht eine Antwort je Bild: Globales Average Pooling und eine <code>Linear</code>-Schicht machen aus den Merkmalskarten in voller Größe zehn Klassen-Scores für Fashion-MNIST.',
    what: [
      'Die 28×28-Bilder von Fashion-MNIST werden beim Laden auf 32×32 vergrößert, damit drei Halbierungen aufgehen (32 → 16 → 8 → 4). Die Filterzahlen sind halb so groß wie im Segmentierungsbeispiel, 16, 32 und 64 mit einem Engpass von 128, damit das Netz auf einem CPU-Server trainiert.',
      'Nach dem Decoder mittelt <code>AdaptiveAvgPool2d</code> jede der 16 Merkmalskarten von 32 × 32 zu einem Wert, <code>Flatten</code> macht daraus einen Vektor der Länge 16, und die letzte <code>Linear</code>-Schicht mit nur 170 Parametern liefert die zehn Klassen-Scores. Insgesamt hat das Netz 481.898 Parameter.',
    ],
    when: [
      'Um zu sehen, wie aus einer Segmentierungsarchitektur ein Klassifikator wird und dass sich nur der Kopf unterscheidet.',
      'Nicht als beste Wahl für diese Aufgabe: Für kleine Bilder mit einem Label ist das <a href="/de/beispiele/cnn-mnist/">kleine CNN</a> kleiner und schneller.',
      'Als Ausgangspunkt für Aufgaben, die beides brauchen: Derselbe Encoder und Decoder kann eine Maske und eine Klasse liefern.',
    ],
    train:
      'Wählen Sie <strong>Fashion-MNIST</strong>: Die 28×28-Graustufenbilder werden automatisch auf den Input <code>1, 32, 32</code> skaliert. Der Lauf meldet die Genauigkeit je Epoche und am Ende die Konfusionsmatrix der zehn Kleidungsklassen, an der Sie sehen, welche das Netz verwechselt (typischerweise Hemden und T-Shirts).',
    related: ['unet', 'cnn_mnist', 'resnet_block'],
  },
  {
    id: 'lstm_sentiment',
    name: 'LSTM-Textklassifikator',
    title: 'LSTM für Sentiment-Analyse in PyTorch – Beispiel mit Code',
    description:
      'Ein bidirektionales LSTM als Textklassifikator in PyTorch: Token-IDs, Embeddings, LSTM und eine Linear-Schicht für positive oder negative Stimmung, mit Code.',
    h1: 'LSTM für Sentiment-Analyse in PyTorch',
    task: 'sequence',
    summary: 'Token-IDs → Embeddings → bidirektionales LSTM → positiv oder negativ.',
    lead: 'Ein rekurrentes Netz liest einen Text als Folge von 128 Token-IDs und entscheidet, ob er positiv oder negativ ist: Ein Embedding gibt jedem Token einen Vektor, ein bidirektionales LSTM liest die Folge vorwärts und rückwärts, und eine <code>Linear</code>-Schicht entscheidet.',
    what: [
      'Der <code>Input</code> enthält 128 ganze Zahlen (Datentyp <code>long</code>): die IDs der Tokens eines Textes aus einem Vokabular von 20.000. <code>Embedding</code> macht aus jeder ID einen Vektor aus 128 Zahlen; mit 20.000 × 128 = 2.560.000 Gewichten ist es mit Abstand der größte Teil der 2.824.706 Parameter.',
      'Das <code>LSTM</code> mit 128 verdeckten Einheiten läuft in beide Richtungen über die 128 Vektoren. Weitergegeben werden nur die letzten verdeckten Zustände (<code>return_sequences</code> aus), die der beiden Richtungen aneinandergehängt: ein Vektor aus 256 Zahlen, der den Text zusammenfasst. Nach <code>Dropout</code> mit p = 0,3 liefert eine <code>Linear</code>-Schicht zwei Scores, negativ und positiv.',
    ],
    when: [
      'Für kurze Texte mit einem Label, bei denen die Wortfolge zählt: Bewertungen, Support-Tickets, Kurznachrichten.',
      'Um zu verstehen, wie Sequenzmodelle mit Formen umgehen: Das Embedding fügt eine Dimension hinzu, das LSTM nimmt sie wieder weg.',
      'Für lange Texte oder große Datenmengen eignet sich oft ein <a href="/de/beispiele/transformer-classifier/">Transformer-Encoder</a> besser, vortrainierte Sprachmodelle noch mehr.',
    ],
    train:
      'Textmodelle trainieren in Brick DNN bisher auf Demodaten: zufällige Token-Folgen mit einer verborgenen Regel. Das zeigt, dass das Netz lernt. Für echte Texte exportieren Sie das Skript: Es enthält das <code>nn.Module</code> und einen Test mit einem zufälligen Batch von Token-IDs. Zerlegen Sie Ihre Texte in IDs unter 20.000, füllen Sie sie auf 128 auf und trainieren Sie mit einer eigenen Schleife.',
    related: ['transformer_classifier', 'multi_input', 'mlp_mnist'],
  },
  {
    id: 'transformer_classifier',
    name: 'Transformer-Textklassifikator',
    title: 'Transformer-Textklassifikator in PyTorch – Beispiel mit Encoder',
    description:
      'Ein kleiner Transformer-Encoder zur Textklassifikation in PyTorch: Embeddings, gelernte Positionen, zwei Encoder-Schichten und Mean Pooling, mit Code.',
    h1: 'Transformer-Textklassifikator in PyTorch',
    task: 'sequence',
    summary:
      'Ein kleiner Transformer-Encoder mit gelernten Positions-Embeddings und Mean Pooling, vier Klassen.',
    lead: 'Ein kompakter Transformer, der Texte in vier Klassen einsortiert: Token-Embeddings plus gelernte Positionen laufen durch zwei Encoder-Schichten mit Self-Attention, und der Mittelwert über alle Positionen geht an den Klassifikator.',
    what: [
      'Der <code>Input</code> nimmt 64 Token-IDs. <code>Embedding</code> bildet jede auf 128 Zahlen ab (10.000 × 128 = 1.280.000 Parameter), und <code>PositionalEmbedding</code> addiert für jede Position einen gelernten Vektor, weil Self-Attention für sich genommen die Reihenfolge der Tokens nicht kennt.',
      'Der <code>TransformerEncoder</code> hat zwei Schichten, jede mit Self-Attention aus vier Köpfen und einem Feed-Forward-Teil mit 256 Einheiten: zusammen 264.960 Parameter. In jeder Schicht kann jede Position jede andere betrachten. <code>SequencePool</code> bildet den Mittelwert über die 64 Positionen, <code>LayerNorm</code> normalisiert den Vektor der Länge 128, und eine <code>Linear</code>-Schicht liefert die vier Klassen-Scores. Das Netz hat 1.611.268 Parameter.',
    ],
    when: [
      'Für die Einordnung kurzer und mittellanger Texte nach Thema oder Absicht, wenn weit auseinanderliegende Wörter voneinander abhängen.',
      'Um die Teile eines Transformer-Encoders mit echten Formen kennenzulernen, bevor Sie mit großen vortrainierten Modellen arbeiten.',
      'Für längere Texte erhöhen Sie <code>max_len</code> des Positions-Embeddings und die Länge des Inputs; der Aufwand der Attention wächst mit dem Quadrat der Länge.',
    ],
    train:
      'Textmodelle trainieren in Brick DNN bisher auf Demodaten, was prüft, dass das Netz lernt. Das exportierte PyTorch-Modul ist bereit für Ihr eigenes Training auf zerlegten Texten: 64 Token-IDs je Text aus einem Vokabular von 10.000.',
    related: ['lstm_sentiment', 'multi_input', 'mlp_mnist'],
  },
  {
    id: 'tabular_regression',
    name: 'Regression auf Tabellendaten',
    title: 'Regression mit neuronalem Netz auf Tabellendaten in PyTorch',
    description:
      'Ein kleines MLP, das aus 8 numerischen Merkmalen eine Zahl vorhersagt, in PyTorch, mit Batch-Normalisierung und einer Trainingsschleife für CSV-Dateien.',
    h1: 'Regression mit neuronalem Netz auf Tabellendaten',
    task: 'regression',
    summary:
      'Sagt aus 8 numerischen Merkmalen eine Zahl vorher; das Skript trainiert auf einer CSV-Datei.',
    lead: 'Ein kleines mehrschichtiges Perzeptron für Tabellen: Acht numerische Spalten gehen hinein, eine Zahl kommt heraus, etwa ein Preis, ein Ertrag oder eine Temperatur. Das erzeugte Skript trainiert es selbständig auf einer CSV-Datei.',
    what: [
      'Zwei verdeckte <code>Linear</code>-Schichten mit 64 und 32 Einheiten und ReLU lernen den Zusammenhang zwischen den acht Merkmalen und der Zielgröße. <code>BatchNorm1d</code> nach der ersten verdeckten Schicht normalisiert deren Ausgaben über jeden Batch, was das Training weniger empfindlich für die Lernrate macht. Die letzte Schicht hat eine einzige Ausgabe ohne Aktivierung und kann so jede Zahl vorhersagen. Das Netz hat 2.817 Parameter.',
      'Bei einer Regression standardisiert Brick DNN Merkmale und Zielgröße (Mittelwert 0, Standardabweichung 1, aus den Trainingszeilen) und trainiert mit der mittleren quadratischen Abweichung. Die Ergebnisse erscheinen zusätzlich als RMSE in der Einheit der Zielgröße und als R².',
    ],
    when: [
      'Für Regression auf Tabellen mit wenigen bis einigen hundert numerischen Spalten, sobald die Daten sauber sind (keine fehlenden Werte, Kategorien kodiert).',
      'Zum Vergleich mit Gradient Boosting, das auf Tabellen oft die stärkere Basis ist: siehe die <a href="/de/anleitungen/gradient-boosting-oder-mlp/">Anleitung Gradient Boosting oder MLP</a>.',
      'Bei mehr Merkmalen setzen Sie den Input auf deren Anzahl, bei mehreren Zielgrößen bekommt die letzte Schicht eine Ausgabe je Zielgröße.',
    ],
    train:
      'Laden Sie eine CSV-Datei hoch, deren letzte Spalte die Zielgröße ist und deren übrige Spalten numerische Merkmale sind, oder bereiten Sie eine Tabelle im Bereich <strong>Daten</strong> auf (fehlende Werte, Kategorien, Datumsangaben) und klicken Sie auf <strong>Trainingsdatensatz erstellen</strong>; der Datensatz erscheint dann im Trainingsbereich. Der Input folgt der Zahl der Merkmale des Datensatzes. Weil das Netz einen flachen Vektor nimmt und einen ausgibt, enthält das exportierte Skript eine Trainingsschleife: <code>python tabular_regression.py --csv daten.csv</code>.',
    related: ['gp_regression', 'stacking_ensemble_regression', 'hybrid_mlp_gradient_boosting'],
  },
  {
    id: 'multi_input',
    name: 'Modell mit mehreren Eingaben',
    title: 'Neuronales Netz mit mehreren Eingaben in PyTorch – Bild und Metadaten',
    description:
      'Ein neuronales Netz mit zwei Eingaben in PyTorch: ein Zweig für ein Bild, einer für Metadaten, verbunden durch Concat, etwa für eine Preisprognose. Mit Code.',
    h1: 'Neuronales Netz mit mehreren Eingaben in PyTorch',
    task: 'regression',
    summary:
      'Ein Bild und ein Vektor mit Metadaten, verbunden durch Concat, sagen eine Zahl vorher.',
    lead: 'Zwei Arten von Daten in einem Netz: Ein 28×28-Bild läuft durch einen Zweig, zehn Zahlen mit Metadaten durch einen anderen, und <code>Concat</code> führt zusammen, was beide gelernt haben, bevor eine letzte Schicht einen einzelnen Wert vorhersagt, etwa den Preis eines Artikels aus Foto und Eigenschaften.',
    what: [
      'Der Bildzweig rollt das Bild 1 × 28 × 28 aus und verdichtet es mit einer <code>Linear</code>-Schicht und ReLU auf 64 Merkmale (50.240 Parameter). Der Metadatenzweig bildet zehn Werte auf 16 Merkmale ab (176 Parameter). <code>Concat</code> hängt beide Vektoren zu einem aus 80 Werten aneinander, und der Kopf, eine <code>Linear</code>-Schicht mit einer Ausgabe, sagt die Zielgröße vorher.',
      'Die erzeugte <code>forward</code>-Methode nimmt zwei Tensoren, einen je Input-Schicht: <code>x_1</code> das Bild, <code>x_2</code> die Metadaten. Wenn das Bild wichtiger ist, ersetzen Sie den Bildzweig durch Faltungen wie im <a href="/de/beispiele/cnn-mnist/">kleinen CNN</a>.',
    ],
    when: [
      'Wenn ein Datensatz Daten verschiedener Art enthält: ein Foto und eine Tabellenzeile, ein Signal und das Alter einer Person, ein Text und seine Metadaten.',
      'Um zu lernen, wie Zweige und Zusammenführungen in einem Graphen funktionieren und wie der Code eines Netzes mit mehreren Eingaben aussieht.',
      'Statt zweier getrennter Modelle, deren Vorhersagen gemittelt werden: Das Netz lernt, wie die Eingaben zusammenwirken.',
    ],
    train:
      'Netze mit mehreren Eingaben trainieren in Brick DNN bisher auf Demodaten, was prüft, dass das Netz aus beiden Eingaben lernt. Das exportierte Modul nimmt beide Tensoren in <code>forward(x_1, x_2)</code> und ist bereit für einen eigenen Data Loader.',
    related: ['mlp_mnist', 'tabular_regression', 'cnn_mnist'],
  },
  {
    id: 'gp_regression',
    name: 'Gauß-Prozess-Regression',
    title: 'Gauß-Prozess-Regression mit scikit-learn – Beispiel mit RBF-Kernel',
    description:
      'Gauß-Prozess-Regression mit skaliertem RBF-Kernel plus weißem Rauschen in scikit-learn: Vorhersagen mit Unsicherheit aus 8 Merkmalen, Kernel aus Bausteinen.',
    h1: 'Gauß-Prozess-Regression mit scikit-learn',
    task: 'regression',
    summary:
      'Sagt eine Zahl samt Unsicherheit vorher; der Kernel ist ein skalierter RBF plus weißes Rauschen.',
    lead: 'Ein Gauß-Prozess sagt eine Zahl vorher und dazu, wie sicher er sich ist. Sein Kernel, hier ein skalierter RBF plus weißes Rauschen, wird auf der Arbeitsfläche aus Bausteinen zusammengesetzt, und beim Anpassen werden seine Hyperparameter an die Daten angepasst.',
    what: [
      'Der Kernel sagt, wie ähnlich sich zwei Zeilen sind und wie viel die Zielgröße der einen damit über die der anderen verrät. <code>ConstantKernel</code> × <code>RBF</code> ist eine glatte Ähnlichkeit, die mit dem Abstand der Zeilen abnimmt, skaliert mit einer Konstanten; <code>WhiteKernel</code> ergänzt Rauschen, denn Messungen sind nie exakt. <code>KernelProduct</code> und <code>KernelSum</code> kombinieren sie zu <code>ConstantKernel(constant_value=1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)</code>.',
      'Die drei Werte sind Startwerte: Das Anpassen maximiert die Marginal Likelihood und findet so Skalierung, Längenskala und Rauschniveau, die zu den Daten passen. Trainierbare Gewichte wie ein Netz hat das Modell nicht; Brick DNN zählt stattdessen die drei Hyperparameter des Kernels. Das exportierte Skript ist eine scikit-learn-<code>Pipeline</code> aus einem <code>StandardScaler</code> und dem <code>GaussianProcessRegressor</code>.',
    ],
    when: [
      'Für kleine bis mittlere Tabellen (bis zu einigen tausend Zeilen), wenn Sie zu jeder Vorhersage eine Unsicherheit brauchen, etwa bei Versuchsreihen, Kalibrierung oder Bayes’scher Optimierung.',
      'Wenn der Zusammenhang glatt ist und es wenige Daten gibt: Ein Gauß-Prozess macht aus wenigen Punkten viel.',
      'Nicht für große Tabellen: Die Rechenzeit wächst mit der dritten Potenz der Zeilenzahl. Brick DNN passt ihn an höchstens 5.000 Trainingszeilen an.',
    ],
    train:
      'Laden Sie eine CSV-Datei mit acht numerischen Merkmalsspalten und einer numerischen Zielgröße (letzte Spalte) hoch oder verwenden Sie einen Datensatz aus dem Bereich <strong>Daten</strong>. Das Modell wird einmal angepasst, ohne Epochen; der Lauf zeigt den angepassten Kernel, seine Log Marginal Likelihood, RMSE und R². Endet ein Hyperparameter an seiner Grenze, weist Brick DNN darauf hin und bietet eine Korrektur an.',
    related: ['periodic_gp', 'stacking_ensemble_regression', 'tabular_regression'],
  },
  {
    id: 'svm_classifier',
    name: 'SVM-Klassifikator',
    title: 'SVM-Klassifikator mit RBF-Kernel in scikit-learn – Beispiel mit Code',
    description:
      'Eine Support Vector Machine (SVC) mit RBF-Kernel klassifiziert Zeilen aus 4 numerischen Merkmalen, mit Klassenwahrscheinlichkeiten per Platt-Skalierung.',
    h1: 'SVM-Klassifikator mit RBF-Kernel in scikit-learn',
    task: 'classification',
    summary:
      'Eine Support Vector Machine mit RBF-Kernel klassifiziert Zeilen aus 4 Merkmalen, mit Wahrscheinlichkeiten.',
    lead: 'Eine Support Vector Machine mit RBF-Kernel für Zeilen aus vier numerischen Merkmalen, etwa den Messwerten von Schwertlilien (Iris). Sie trennt die Klassen mit dem größtmöglichen Abstand und liefert zusätzlich Klassenwahrscheinlichkeiten.',
    what: [
      'Der Baustein <code>SVC</code> bekommt die Merkmale direkt vom <code>Input</code>. Mit dem RBF-Kernel darf die Grenze zwischen den Klassen gekrümmt sein; <code>C = 1.0</code> legt fest, wie viel eine falsch eingeordnete Trainingszeile kostet, und <code>gamma = scale</code> wählt die Breite des Kernels aus der Streuung der Merkmale.',
      'Eine SVM liefert Scores, keine Wahrscheinlichkeiten. Das erzeugte Skript legt deshalb <code>CalibratedClassifierCV</code> darum, das die Scores per Platt-Skalierung mit 5-facher Kreuzvalidierung in Wahrscheinlichkeiten umrechnet, und setzt einen <code>StandardScaler</code> davor, weil eine SVM vom Maßstab der Merkmale abhängt.',
    ],
    when: [
      'Für kleine und mittlere Tabellen mit numerischen Merkmalen und einer klaren, aber nichtlinearen Grenze zwischen den Klassen.',
      'Bis zu einigen zehntausend Zeilen: Der Aufwand wächst zwischen dem Quadrat und der dritten Potenz der Zeilenzahl. Brick DNN passt sie an höchstens 50.000 Trainingszeilen an; für mehr nehmen Sie <a href="/de/beispiele/gradient-boosting-classifier/">Gradient Boosting</a>.',
      'Bei seltenen Klassen setzen Sie <code>class_weight</code> auf <code>balanced</code>.',
    ],
    train:
      'Laden Sie eine CSV-Datei mit vier numerischen Merkmalen und einer Klassenspalte hoch (Textlabels wie <code>setosa</code> sind in Ordnung) oder einen aufbereiteten Datensatz aus dem Bereich <strong>Daten</strong>. Der Lauf zeigt Genauigkeit, Konfusionsmatrix und die Zahl der Stützvektoren; eine Hyperparametersuche über <code>C</code> und <code>gamma</code> mit Kreuzvalidierung ist nur einen Schalter entfernt.',
    related: [
      'gradient_boosting_classifier',
      'voting_ensemble_classifier',
      'hybrid_mlp_gradient_boosting',
    ],
  },
  {
    id: 'periodic_gp',
    name: 'Periodischer Gauß-Prozess',
    title: 'Periodischer Gauß-Prozess für saisonale Daten – scikit-learn-Beispiel',
    description:
      'Ein Gauß-Prozess für saisonale Zeitreihen in scikit-learn: periodischer Kernel aus RBF × ExpSineSquared plus glatter Trend und Rauschen, aus Bausteinen.',
    h1: 'Periodischer Gauß-Prozess für saisonale Daten',
    task: 'regression',
    summary:
      'Ein saisonales Muster (RBF × ExpSineSquared) plus glatter Trend und Rauschen, für einen Wert über die Zeit.',
    lead: 'Ein Gauß-Prozess für einen Messwert über die Zeit, der sich mit den Jahreszeiten wiederholt und langsam driftet: Sein Kernel addiert einen periodischen Anteil, einen glatten Trend und Rauschen, jeder davon ein Baustein auf der Arbeitsfläche.',
    what: [
      'Der Input ist ein einziges Merkmal, die Zeit. <code>ExpSineSquared</code> ist ein periodischer Kernel: Punkte, die eine Periode auseinanderliegen, gelten als ähnlich. Multipliziert mit einem <code>RBF</code> mit langer Längenskala (10) darf sich das saisonale Muster von Jahr zu Jahr langsam ändern. Ein zweiter <code>RBF</code> mit Längenskala 5 beschreibt den Trend, ein <code>WhiteKernel</code> das Rauschen. <code>KernelSum</code> addiert die drei Teile.',
      'Beim Anpassen werden alle fünf Hyperparameter optimiert, auch die Periodenlänge. Setzen Sie deren Startwert auf die Länge einer Saison in standardisierten Einheiten des Zeitmerkmals, denn die Pipeline standardisiert die Merkmale zuerst.',
    ],
    when: [
      'Für saisonale Messreihen mit wenigen Punkten: Monatsumsätze, Temperaturen, Energieverbrauch, Sensorwerte mit Tagesgang.',
      'Wenn die Unsicherheit einer Prognose wachsen soll, je weiter sie sich von den Daten entfernt.',
      'Um zu lernen, wie zusammengesetzte Kernel Struktur beschreiben: Entfernen Sie den Baustein für den Trend und sehen Sie, was die Anpassung aus den Daten macht.',
    ],
    train:
      'Verwenden Sie eine CSV-Datei mit der Zeit als Zahl (etwa Tage seit der ersten Messung) als einziger Merkmalsspalte und dem Messwert als Zielgröße in der letzten Spalte. Der Lauf zeigt den angepassten Kernel mit allen Hyperparametern und die Log Marginal Likelihood.',
    related: ['gp_regression', 'stacking_ensemble_regression', 'tabular_regression'],
  },
  {
    id: 'gradient_boosting_classifier',
    name: 'Gradient-Boosting-Klassifikator',
    title: 'Gradient-Boosting-Klassifikator in scikit-learn – HistGradientBoosting',
    description:
      'Ein histogrammbasierter Gradient-Boosting-Klassifikator für 30 numerische Merkmale in scikit-learn, mit ausgeglichenen Klassengewichten für seltene Klassen.',
    h1: 'Gradient-Boosting-Klassifikator in scikit-learn',
    task: 'classification',
    summary:
      'Geboostete Entscheidungsbäume klassifizieren Zeilen aus 30 Merkmalen; ausgeglichene Klassengewichte für seltene Klassen.',
    lead: 'Geboostete Entscheidungsbäume sind die starke Standardwahl für Tabellendaten. Dieses Beispiel klassifiziert Zeilen aus 30 numerischen Merkmalen und gewichtet die Klassen so, dass eine seltene Klasse, etwa wenige Prozent Betrugsfälle, nicht untergeht.',
    what: [
      'Der Baustein <code>HistGradientBoostingClassifier</code> fügt nacheinander bis zu 200 Entscheidungsbäume hinzu (<code>max_iter</code>); jeder korrigiert die Fehler der vorherigen, skaliert mit der Lernrate 0,1. Ein Baum hat höchstens 31 Blätter mit je mindestens 20 Zeilen. Vorab werden die Merkmale in Histogramme eingeteilt, was das Verfahren auch bei Hunderttausenden Zeilen schnell macht.',
      '<code>class_weight = balanced</code> gewichtet jede Klasse umgekehrt zu ihrer Häufigkeit. Ohne das erreicht ein Modell eine hohe Genauigkeit, indem es die seltene Klasse kaum je vorhersagt. Das frühe Stoppen (<code>auto</code>) schaltet sich ab 10.000 Trainingszeilen ein und hört auf, Bäume hinzuzufügen, sobald ein zurückgehaltenes Zehntel davon nicht mehr besser wird.',
    ],
    when: [
      'Als erstes Modell für fast jede Tabelle: Es kommt ohne Feinabstimmung mit unterschiedlichen Maßstäben und nichtlinearen Zusammenhängen zurecht.',
      'Für große Tabellen: Die Rechenzeit wächst etwa linear mit der Zeilenzahl. Brick DNN passt es an bis zu 1.000.000 Trainingszeilen an.',
      'Für unausgewogene Klassen, mit ausgeglichenen Gewichten und einem Blick auf Balanced Accuracy, Recall und PR AUC statt auf die Genauigkeit: siehe die <a href="/de/anleitungen/unausgewogene-klassen/">Anleitung zu unausgewogenen Klassen</a>.',
    ],
    train:
      'Laden Sie eine CSV-Datei hoch oder bereiten Sie eine Tabelle im Bereich <strong>Daten</strong> auf; fehlende Werte müssen wie bei den anderen klassischen Modellen vorher ergänzt werden. Der Lauf meldet Genauigkeit, Balanced Accuracy, F1, die Konfusionsmatrix und bei zwei Klassen ROC- und PR-Kurven mit verschiebbarer Entscheidungsschwelle. Eine Suche über Lernrate, Blätter und Zeilen je Blatt mit Kreuzvalidierung findet bessere Einstellungen.',
    related: ['svm_classifier', 'voting_ensemble_classifier', 'hybrid_mlp_gradient_boosting'],
  },
  {
    id: 'voting_ensemble_classifier',
    name: 'Voting-Ensemble',
    title: 'Voting-Ensemble in scikit-learn – Beispiel mit Soft Voting',
    description:
      'Ein Ensemble aus Gradient Boosting, einer SVM und einem Gauß-Prozess-Klassifikator mit Soft Voting in scikit-learn: Die Klassenwahrscheinlichkeiten werden gemittelt.',
    h1: 'Voting-Ensemble zur Klassifikation in scikit-learn',
    task: 'classification',
    summary:
      'Gradient Boosting, eine SVM und ein Gauß-Prozess; Soft Voting mittelt ihre Wahrscheinlichkeiten.',
    lead: 'Drei unterschiedliche Klassifikatoren betrachten dieselben vier Merkmale, und das Ensemble mittelt ihre Klassenwahrscheinlichkeiten. Modelle, die verschiedene Fehler machen, gleichen einen Teil davon gegenseitig aus.',
    what: [
      'Der <code>Input</code> versorgt alle drei Modelle: <code>HistGradientBoostingClassifier</code> (Bäume), <code>SVC</code> (eine Kernel-Grenze) und <code>GaussianProcessClassifier</code> (mit seinem Standardkernel, einem skalierten RBF). Die gepunkteten Verbindungen zum <code>VotingClassifier</code> tragen die Modelle selbst, nicht ihre Vorhersagen; das Ensemble passt jedes davon an die Trainingszeilen an.',
      'Mit <code>voting = soft</code> mittelt das Ensemble die vorhergesagten Wahrscheinlichkeiten und wählt die wahrscheinlichste Klasse. Hard Voting würde stattdessen die vorhergesagten Klassen auszählen. Optionale Gewichte lassen ein Modell stärker zählen; sie folgen der Reihenfolge der Verbindungen.',
    ],
    when: [
      'Wenn mehrere Modelle etwa gleich gut sind und bei verschiedenen Zeilen danebenliegen: Der Mittelwert ist oft besser als das beste Einzelmodell.',
      'Für kleine und mittlere Tabellen: Das langsamste Modell setzt die Grenze, hier der Gauß-Prozess-Klassifikator (höchstens 2.000 Trainingszeilen in Brick DNN).',
      'Um zu prüfen, ob das Kombinieren hilft: Der Lauf zeigt den Score jedes Modells neben dem des Ensembles.',
    ],
    train:
      'Laden Sie eine CSV-Datei mit vier numerischen Merkmalen und einer Klassenspalte hoch oder verwenden Sie einen Datensatz aus dem Bereich <strong>Daten</strong>. Das Ensemble trainiert wie ein einzelnes klassisches Modell, mit derselben Auswertung (Konfusionsmatrix, Schwellenanalyse bei zwei Klassen), denselben Erklärungen und demselben Download.',
    related: ['stacking_ensemble_regression', 'gradient_boosting_classifier', 'svm_classifier'],
  },
  {
    id: 'stacking_ensemble_regression',
    name: 'Stacking-Ensemble zur Regression',
    title: 'Stacking-Ensemble für Regression in scikit-learn – Beispiel mit Code',
    description:
      'Ein Stacking-Ensemble in scikit-learn: Gradient Boosting, eine SVR und ein Gauß-Prozess sagen eine Zahl vorher, eine Ridge-Regression lernt, wie sie zu gewichten sind.',
    h1: 'Stacking-Ensemble zur Regression in scikit-learn',
    task: 'regression',
    summary:
      'Gradient Boosting, eine SVR und ein Gauß-Prozess, kombiniert durch eine Ridge-Regression.',
    lead: 'Beim Stacking lernt ein Modell, wie sehr es anderen Modellen trauen kann: Gradient Boosting, eine Support Vector Regression und ein Gauß-Prozess sagen aus acht Merkmalen eine Zahl vorher, und eine Ridge-Regression lernt aus ihren Vorhersagen, wie sie zu kombinieren sind.',
    what: [
      'Die drei Modelle erhalten die Merkmale vom <code>Input</code>. Der Gauß-Prozess hat einen eigenen Kernel, einen <code>RBF</code> plus <code>WhiteKernel</code>, verbunden durch <code>KernelSum</code> (die gestrichelten Verbindungen tragen den Kernel). Die gepunkteten Verbindungen zum <code>StackingRegressor</code> tragen die drei Modelle.',
      'Mit <code>cv = 5</code> wird jedes Modell fünfmal angepasst, jedes Mal ohne ein Fünftel der Trainingszeilen, und sagt dieses Fünftel vorher. Die Ridge-Regression lernt aus diesen Vorhersagen für Zeilen, die die Modelle nicht gesehen haben, und belohnt deshalb kein Modell, das die Daten nur auswendig gelernt hat. Danach wird jedes Modell noch einmal an alle Zeilen angepasst. Das dauert insgesamt etwa sechsmal so lange wie ein einzelnes Anpassen der drei Modelle.',
    ],
    when: [
      'Wenn verschiedene Arten von Modellen verschiedene Teile des Zusammenhangs erfassen, etwa bei Immobilienpreisen mit glatten und sprunghaften Effekten.',
      'Wenn ein einfacher Mittelwert nicht reicht, weil ein Modell in manchen Bereichen klar besser ist.',
      'Hier für Tabellen bis zu einigen tausend Zeilen, wegen des Gauß-Prozesses (höchstens 5.000 Trainingszeilen in Brick DNN).',
    ],
    train:
      'Laden Sie eine CSV-Datei mit acht numerischen Merkmalen und einer numerischen Zielgröße hoch oder verwenden Sie einen Datensatz aus dem Bereich <strong>Daten</strong>. Der Lauf meldet RMSE und R² des Ensembles und jedes einzelnen Modells auf den Validierungszeilen; so sehen Sie, ob das Stacking sein bestes Modell schlägt.',
    related: ['voting_ensemble_classifier', 'gp_regression', 'tabular_regression'],
  },
  {
    id: 'hybrid_mlp_gradient_boosting',
    name: 'MLP-Merkmale für Gradient Boosting',
    title: 'MLP-Merkmale für Gradient Boosting – hybrides PyTorch + scikit-learn',
    description:
      'Ein hybrides Netz: Ein MLP in PyTorch lernt 32 Merkmale aus 8 Spalten, danach werden geboostete Bäume in scikit-learn darauf angepasst. Mit Diagramm und Code.',
    h1: 'MLP-Merkmale für Gradient Boosting',
    task: 'classification',
    summary:
      'Ein MLP lernt 32 Merkmale, danach werden geboostete Bäume darauf angepasst und sagen die Klasse vorher.',
    lead: 'Eine Mischung aus Deep Learning und Bäumen: Zwei <code>Linear</code>-Schichten lernen aus acht Spalten 32 neue Merkmale, und ein Gradient-Boosting-Klassifikator wird auf diese Merkmale angepasst statt auf die Rohspalten.',
    what: [
      'Das Training hat zwei Phasen in einem Lauf. Zuerst lernen die neuronalen Schichten mit einem vorübergehenden linearen Kopf auf die Zielgröße, wie ein kleines eigenes Netz: 8 → 64 → ReLU → 32, mit 2.656 trainierbaren Parametern. Dann fällt der Kopf weg, die Schichten berechnen die 32 Merkmale jeder Zeile, und der <code>HistGradientBoostingClassifier</code> wird auf die Merkmale der Trainingszeilen angepasst.',
      'Das erzeugte Skript enthält beide Teile: die Schichten als PyTorch-Modul (<code>FeatureExtractor</code>), den Klassifikator als scikit-learn-Pipeline und den Code, der beide Phasen auf einer CSV-Datei trainiert und beides speichert.',
    ],
    when: [
      'Um zu prüfen, ob gelernte Merkmale den Bäumen auf Ihrer Tabelle helfen: Vergleichen Sie auf demselben Split mit dem <a href="/de/beispiele/gradient-boosting-classifier/">reinen Gradient Boosting</a>.',
      'Mit <em>eingefrorenen Zufallsmerkmalen</em> in den Trainingseinstellungen als schnelle Basis dafür, was das Lernen der Merkmale bringt.',
      'Als Muster für Bilder: ein kleines CNN, das mit <code>Flatten</code> oder <code>Linear</code> endet, vor einer SVM oder einem Gradient Boosting.',
    ],
    train:
      'Setzen Sie den Input auf die Zahl der Merkmalsspalten Ihrer Daten und trainieren Sie dann auf einer CSV-Datei, einem Datensatz aus dem Bereich <strong>Daten</strong> oder auf Demodaten. Der Trainingsbereich zeigt für die erste Phase die Einstellungen eines Netzes (Epochen, Lernrate …); die Kurven gehören zum Lernen der Merkmale, die Auswertung zum angepassten Klassifikator. Der Download ist eine ZIP-Datei mit beiden Teilen und dem Skript.',
    related: ['gradient_boosting_classifier', 'tabular_regression', 'mlp_mnist'],
  },
];

export const examplesDe = { gallery, examples };
