import type { training as en } from '../en/training';
import type { Messages } from '../types';

export const training: Messages<typeof en> = {
  drawer: {
    title: 'Dieses Netz trainieren',
    titleRuns: 'Trainings dieses Netzes',
    sidePanel: 'Zurück zur Seitenleiste',
    fullWidth: 'Volle Breite',
    unavailable: 'Training ist nicht verfügbar',
    loginTitle: 'Zum Trainieren anmelden',
    loginText:
      'Das Training läuft auf dem Brick-DNN-Server und braucht wie das Speichern ein kostenloses Konto. Ihr Netz bleibt auf der Arbeitsfläche, während Sie sich anmelden.',
    newTraining: 'Neues Training',
    runs: 'Läufe',
    viewerNote:
      'Sie können dieses Netz ansehen: Sie sehen, wie es trainiert wurde, können darin aber keine Trainings starten. Um es selbst zu trainieren, kopieren Sie es.',
    cannotLoad: 'Das Training kann nicht geladen werden',
    allRuns: 'Alle Läufe',
    lostConnection: 'Verbindung zum Server verloren ({{error}}). Neuer Versuch …',
    noSklearn: 'scikit-learn ist auf dem Server nicht installiert.',
    hybridNoSklearn:
      'Dieses Netz ist hybrid: Seine neuronalen Schichten lernen die Merkmale mit PyTorch, das installiert ist, und sein klassisches Modell wird darauf mit scikit-learn angepasst, das auf dem Rechner des Brick-DNN-Backends fehlt. Im Ordner <code>backend</code> bei aktivierter virtueller Umgebung:',
    classicalNoSklearn:
      'Klassische Modelle (Gauß-Prozesse, SVMs, Gradient Boosting) werden auf dem Rechner des Brick-DNN-Backends mit scikit-learn angepasst; sie brauchen kein PyTorch. Im Ordner <code>backend</code> bei aktivierter virtueller Umgebung:',
    dockerSklearn:
      'Das Docker-Image enthält scikit-learn. Starten Sie das Backend neu und prüfen Sie erneut.',
    checkAgain: 'Erneut prüfen',
    disabled: 'Training ist auf diesem Server deaktiviert.',
    noTorch: 'PyTorch ist auf dem Server nicht installiert.',
    needsTorch:
      'Das Training läuft auf dem Rechner des Brick-DNN-Backends und braucht dort PyTorch. So installieren Sie es:',
    local: 'Lokale Installation',
    inBackend: 'Im Ordner <code>backend</code> bei aktivierter virtueller Umgebung:',
    hybridAlso:
      'Dieses hybride Netz braucht für sein klassisches Modell außerdem scikit-learn: <code>pip install -r requirements.txt</code> im Ordner <code>backend</code> (das Docker-Image enthält es).',
    restart: 'Starten Sie dann das Backend neu und prüfen Sie erneut.',
  },
  source: {
    label: 'Datenquelle',
    tabs: { random: 'Demodaten', builtin: 'Eingebaut', upload: 'Ihre Daten' },
    random:
      'Zufällige Eingaben mit einer versteckten Regel zum Lernen. Funktioniert mit jedem Netz und zeigt, ob es überhaupt lernen kann. Es wird nichts heruntergeladen.',
    dataset: 'Datensatz',
    downloadedOrGenerated: 'Wird beim ersten Mal auf den Server geladen oder dort erzeugt.',
    downloaded: 'Wird beim ersten Mal auf den Server geladen.',
    needsTorchvision: 'Heruntergeladene Datensätze brauchen torchvision auf dem Server.',
    sizes: '{{train}} Trainings- und {{test}} Testbilder',
    segmentationDetails_one:
      '{{sizes}} mit Masken · {{count}} Klasse ({{classes}}) · kein Download',
    segmentationDetails_other:
      '{{sizes}} mit Masken · {{count}} Klassen ({{classes}}) · kein Download',
    details_one: '{{sizes}} · {{count}} Klasse · {{mb}} MB Download',
    details_other: '{{sizes}} · {{count}} Klassen · {{mb}} MB Download',
    dataTab: 'Daten-Tab',
    messyCsv: 'Unaufgeräumte CSV-Datei? Bereinigen, kodieren und analysieren Sie sie im',
  },
  setup: {
    budget: {
      budget_exceeded: 'Nicht genug Rechenbudget',
      estimate_too_high: 'Zu viel Rechenzeit für das Budget',
      storage_quota_exceeded: 'Der Speicher Ihres Kontos ist voll',
      other: 'Nicht genug Budget',
    },
    deleteTitle: '„{{name}}“ löschen?',
    deleteText:
      'Die hochgeladenen Daten werden vom Server entfernt. Abgeschlossene Trainings behalten ihre Ergebnisse.',
    deleteFailed: 'Die Daten können nicht gelöscht werden',
    empty:
      'Die Arbeitsfläche ist leer. Bauen Sie ein Netz (oder laden Sie ein Beispiel), um es zu trainieren.',
    fixFirst: 'Beheben Sie zuerst die rot markierten Schichten und kommen Sie dann hierher zurück.',
    showProblems: 'Probleme anzeigen',
    running: 'Ein Training läuft',
    stillTraining:
      '„{{name}}“ trainiert noch. Sie können ein neues starten, wenn es abgeschlossen ist.',
    stillTrainingBy:
      '„{{name}}“ (gestartet von {{user}}) trainiert noch. Sie können ein neues starten, wenn es abgeschlossen ist.',
    showIt: 'Anzeigen',
    data: 'Daten',
    settings: 'Einstellungen',
    tooMany: 'Zu viele Trainings warten',
    another: 'Ein anderes Training läuft',
    cannotStart: 'Das Training kann nicht gestartet werden',
    openIt: 'Öffnen',
    visible: 'Für alle in diesem Netz sichtbar',
    startSearch: 'Suche starten',
    start: 'Training starten',
    noWorker: 'Kein Trainings-Worker läuft',
    noWorkerText:
      'Dieser Server kann gerade nicht trainieren. Ein gestartetes Training wartet in der Warteschlange, bis wieder ein Worker läuft.',
    afterYours: 'startet nach Ihrem laufenden Training',
    busy: 'Der Server ist ausgelastet: Dieses Training wartet in der Warteschlange (Position {{position}}; geschätzt).',
    busyStart:
      'Der Server ist ausgelastet: Dieses Training wartet in der Warteschlange (Position {{position}}, {{start}}; geschätzt).',
    theData: 'die Daten',
    cannotCheck: 'Die Daten können nicht geprüft werden',
    checking: 'Das Netz wird mit den Daten abgeglichen …',
    noFit: 'Passt nicht zu den Daten',
    ready: 'Bereit zum Training',
    fitExactly: 'Damit es genau zu „{{name}}“ passt, ändern Sie:',
    fit: 'Damit es zu „{{name}}“ passt, ändern Sie:',
    adapt: 'Netz an „{{name}}“ anpassen',
  },
  settings: {
    searched: 'Von der Suche festgelegt',
    notTrained: 'Nicht verwendet: eingefrorene Merkmale werden nicht trainiert',
    automatic: 'Automatisch',
    automaticDevice: 'Automatisch: {{device}}',
    name: 'Name',
    valSplit: 'Validierungsanteil',
    noValClassical:
      'Keine Validierungsdaten: Das Modell wird nur auf den Testzeilen gemessen (ohne sie auf seinen Trainingszeilen).',
    noVal:
      'Keine Validierungsdaten: Die letzte Epoche wird behalten, kein Early Stopping und keine Suche; Bewertungen nur auf Test- (oder Trainings-)Daten.',
    officialTest: 'Auf den offiziellen Testbildern testen',
    holdOut: 'Testdaten zurückhalten',
    ownSplitImages:
      'Dieser Datensatz bringt seine eigene Aufteilung mit, die Ordner train, val und test: {{split}} Bilder für Training, Validierung und Test. Das Training verwendet sie unverändert.',
    ownSplitRows:
      'Dieser Datensatz bringt seine eigene Aufteilung aus dem Daten-Tab mit: {{split}} Zeilen für Training, Validierung und Test. Das Training verwendet sie unverändert; ändern Sie sie im Daten-Tab.',
    builtinTest:
      'Bei eingebauten Datensätzen immer an: Auf den offiziellen Testbildern wird nie trainiert, und sie entscheiden nicht über die Gewichte; die behaltenen Gewichte werden nach dem Training einmal darauf gemessen. Der Validierungsanteil wird aus den Trainingsbildern genommen.',
    testHint:
      'Wird nie trainiert und nicht zum Vergleich von Einstellungen verwendet: Das endgültige Modell wird nach dem Training einmal darauf gemessen.',
    testSplit: 'Testanteil',
    maxSamples: 'Max. Stichproben',
    maxRowsHint:
      '{{model}} wird auf diesem Server an höchstens {{rows}} Trainingszeilen angepasst; größere Trainingsmengen werden ausgedünnt.',
    maxSamplesHint: 'Trainingszeilen für die Anpassung (leer: alle).',
    allUpTo: 'Alle, bis zu {{rows}}',
    all: 'Alle',
    seed: 'Seed',
    random: 'Zufällig',
    classicalNote:
      '{{model}} wird einmal mit scikit-learn auf der CPU angepasst: Es gelten nur die Aufteilung, die Zeilenzahl und der Seed.',
    classicalNoteKernel:
      '{{model}} wird einmal mit scikit-learn auf der CPU angepasst: Es gelten nur die Aufteilung, die Zeilenzahl und der Seed; die Anpassung optimiert die Kernel-Hyperparameter selbst.',
    search: 'Hyperparameter-Suche',
    searchClassical:
      'Passt das Modell mehrmals mit anderen Werten seiner Parameter an und behält das beste.',
    searchHybrid:
      'Passt {{model}} mehrmals mit anderen Werten seiner Parameter an die Merkmale an und kann die Merkmale in jedem Versuch mit anderen Trainingseinstellungen neu lernen; behält den besten Versuch.',
    searchNeural:
      'Trainiert das Netz mehrmals mit verschiedenen Einstellungen und behält das beste (standardmäßig das mit dem niedrigsten Validierungsverlust).',
    hybridNote:
      'Zwei Phasen: Die neuronalen Schichten lernen {{features}} mit einem temporären linearen Kopf auf der Zielspalte (die Einstellungen unten), dann wird {{model}} an die Merkmale der Trainingszeilen angepasst.',
    hybridNoteMax:
      'Zwei Phasen: Die neuronalen Schichten lernen {{features}} mit einem temporären linearen Kopf auf der Zielspalte (die Einstellungen unten), dann wird {{model}} an die Merkmale der Trainingszeilen angepasst (auf diesem Server höchstens {{rows}}).',
    nFeatures_one: '{{count}} Merkmal',
    nFeatures_other: '{{count}} Merkmale',
    theFeatures: 'die Merkmale',
    features: 'Merkmale',
    frozenHint:
      'Die neuronalen Schichten behalten ihre zufälligen Anfangsgewichte: Nichts wird trainiert, daher werden die Trainingseinstellungen (Epochen, Batchgröße, Lernrate, Optimierer, Weight Decay, Early Stopping) nicht verwendet.',
    epochs: 'Epochen',
    epochsFrozen: 'Nicht verwendet: die Merkmale bleiben zufällig',
    perTrial: 'Pro Versuch der Suche',
    epochsHint: 'Durchläufe durch die Trainingsdaten',
    batchSize: 'Batchgröße',
    imagesPerStepMax: 'Bilder pro Optimierungsschritt; bis zu {{max}} passen in den Speicher',
    imagesPerStep: 'Bilder pro Optimierungsschritt (4 bis 16 sind bei Segmentierung üblich)',
    samplesPerStep: 'Stichproben pro Optimierungsschritt',
    learningRate: 'Lernrate',
    optimizer: 'Optimierer',
    device: 'Gerät',
    loss: 'Verlustfunktion',
    lossHint: 'Der Dice-Term hilft Klassen, die nur wenige Pixel einnehmen',
    advanced: 'Erweiterte Einstellungen',
    weightDecay: 'Weight Decay',
    earlyStopping: 'Early Stopping',
    earlyStoppingHint:
      'Stoppen, wenn sich der Validierungsverlust so viele Epochen lang nicht verbessert hat',
    restoreBest: 'Gewichte der besten Epoche behalten',
    restoreBestHint:
      'Die Gewichte mit dem niedrigsten Validierungsverlust testen und speichern, nicht die der letzten Epoche.',
    restoreBestNoVal:
      'Ohne Validierungsdaten nicht verwendet: Die Gewichte der letzten Epoche werden behalten.',
    options: {
      featureTraining: {
        pretrain: 'Lernen (temporärer Kopf auf der Zielspalte)',
        frozen: 'Eingefrorene Zufallsmerkmale (kein Training)',
      },
      loss: {
        cross_entropy: 'Kreuzentropie',
        cross_entropy_dice: 'Kreuzentropie + Dice',
      },
    },
  },
};
