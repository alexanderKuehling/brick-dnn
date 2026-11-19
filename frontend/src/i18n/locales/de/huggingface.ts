import type { huggingface as en } from '../en/huggingface';
import type { Messages } from '../types';

export const huggingface: Messages<typeof en> = {
  publish: {
    button: 'Auf Hugging Face veröffentlichen',
    title: 'Auf Hugging Face veröffentlichen',
    intro:
      'Lädt das trainierte Modell, sein Skript, das Netz und eine Modellkarte in ein Modell-Repository Ihres Hugging-Face-Kontos hoch. Gesendet wird erst, wenn Sie auf „Veröffentlichen“ klicken.',
    usa: 'Hugging Face, Inc. sitzt in den USA; dort gilt dessen eigene Datenschutzerklärung.',
    token: 'Hugging-Face-Zugriffstoken',
    tokenHelp:
      'Ein Token mit Schreibrecht (huggingface.co → Settings → Access Tokens). Es wird nur für dieses Hochladen verwendet und nie gespeichert.',
    createToken: 'Token erstellen',
    repo: 'Name des Repositorys',
    repoHelp:
      'Ihr Hugging-Face-Benutzername wird vorangestellt. Für eine Organisation: „organisation/name“.',
    repoInvalid:
      'Nur Buchstaben, Ziffern, „-“, „_“ und „.“ (höchstens 96), optional nach „besitzer/“.',
    visibility: 'Sichtbarkeit',
    private: 'Privat: nur Sie (und Ihre Organisation) sehen es',
    public: 'Öffentlich: alle können es sehen und herunterladen',
    license: 'Lizenz',
    licenseOther: 'Andere (in der Modellkarte beschreiben)',
    updateExisting: 'Repository aktualisieren, falls es schon existiert',
    updateExistingHelp:
      'Gleichnamige Dateien darin werden ersetzt; seine Sichtbarkeit bleibt, wie sie ist.',
    files: 'Dateien ({{size}})',
    approximate: 'Die Größe der Modelldatei ist ungefähr.',
    showCard: 'Modellkarte zeigen (README.md)',
    hideCard: 'Modellkarte ausblenden',
    cardLabel: 'Modellkarte (README.md)',
    tooLarge:
      'Diese Dateien sind größer als {{max}} MB, die Grenze für Veröffentlichungen von diesem Server.',
    notAllowed:
      'Nur wer den Lauf gestartet hat und die Bearbeiter seines Netzes können ihn veröffentlichen.',
    previewFailed: 'Kann nicht zeigen, was veröffentlicht würde',
    loading: 'Dateien werden zusammengestellt …',
    cancel: 'Abbrechen',
    submit: 'Veröffentlichen',
    uploading: 'Wird zu Hugging Face hochgeladen …',
    failed: 'Veröffentlichen fehlgeschlagen',
    done: 'Auf Hugging Face veröffentlicht',
    doneText_one: '„{{repo}}“ ist mit {{count}} Datei auf Hugging Face.',
    doneText_other: '„{{repo}}“ ist mit {{count}} Dateien auf Hugging Face.',
    updated:
      'Ein vorhandenes Repository wurde aktualisiert; seine Sichtbarkeit wurde nicht geändert.',
    open: 'Auf Hugging Face öffnen',
    close: 'Schließen',
    finish: 'Fertig',
  },
  errors: {
    token_invalid:
      'Hugging Face hat das Zugriffstoken nicht angenommen. Prüfen Sie, ob Sie es vollständig kopiert haben und ob es noch gültig ist.',
    token_read_only:
      'Dieses Zugriffstoken darf nur lesen. Erstellen Sie eines mit der Rolle „Write“ (oder ein Fine-grained-Token, das in Ihre Repositorys schreiben darf).',
    forbidden:
      'Hugging Face hat das mit diesem Zugriffstoken abgelehnt: Es darf dort nicht schreiben, oder das Repository gehört jemand anderem.',
    name_taken:
      'Ein Repository mit diesem Namen gibt es schon. Wählen Sie einen anderen Namen oder erlauben Sie, das vorhandene zu aktualisieren.',
    too_large: 'Die Dateien sind zu groß, um sie von hier zu veröffentlichen.',
    busy: 'Hugging Face ist gerade ausgelastet. Versuchen Sie es in ein paar Minuten noch einmal.',
    unreachable: 'Hugging Face ist gerade nicht erreichbar. Versuchen Sie es später noch einmal.',
    timeout: 'Das Veröffentlichen hat zu lange gedauert und wurde abgebrochen.',
    not_found: 'Auf Hugging Face wurde nichts gefunden: Prüfen Sie den Namen.',
    repoCreated:
      'Das Repository „{{repo}}“ wurde angelegt und ist noch leer: Erlauben Sie, das vorhandene Repository zu aktualisieren, um dorthin zu veröffentlichen, oder löschen Sie es auf Hugging Face.',
  },
  import: {
    tab: 'Hugging Face',
    openInTab: 'Ein Datensatz auf Hugging Face: seine Dateien im Tab „Hugging Face“ zeigen',
    dataset: 'Datensatz auf Hugging Face',
    datasetHelp:
      'hf://datasets/besitzer/name, besitzer/name oder der Link seiner Seite auf huggingface.co. CSV-, TSV-, JSON-, JSON-Lines- und Parquet-Dateien lassen sich importieren (höchstens {{maxUploadMb}} MB).',
    token: 'Zugriffstoken (nur für zugangsbeschränkte oder private Datensätze)',
    tokenHelp: 'Nur für diese Anfrage verwendet, nie gespeichert.',
    list: 'Dateien zeigen',
    listing: 'Hugging Face wird nach den Dateien gefragt …',
    listFailed: 'Dateien können nicht aufgelistet werden',
    config: 'Konfiguration',
    split: 'Aufteilung',
    all: 'Alle',
    file: 'Datei',
    noFiles:
      'Dieser Datensatz hat keine CSV-, JSON- oder Parquet-Dateien, die sich importieren lassen.',
    noMatch: 'Keine Datei dieser Konfiguration und Aufteilung.',
    converted:
      'Hugging Face hat diesen Datensatz nach Parquet umgewandelt: Das sind die umgewandelten Dateien.',
    gated:
      'Dieser Datensatz ist zugangsbeschränkt: Akzeptieren Sie seine Bedingungen auf seiner Hugging-Face-Seite und geben Sie dann ein Zugriffstoken ein.',
    private: 'Ein privater Datensatz: Das Zugriffstoken wird auch für den Import gebraucht.',
    truncated_one: 'Nur die erste Datei wird gezeigt.',
    truncated_other: 'Nur die ersten {{count}} Dateien werden gezeigt.',
    choose: 'Wählen Sie eine Datei des Datensatzes.',
    example: 'Kein Datensatz zur Hand? Probieren Sie <0>scikit-learn/iris</0>.',
    privacy:
      'Der Server fragt Hugging Face (USA) nach der Liste der Dateien und lädt die gewählte herunter; gespeichert wird nur diese Datei.',
    open: 'Auf Hugging Face öffnen',
    unknownSize: 'Größe unbekannt',
    splitNone: 'keine Aufteilung',
    fileDetails: '{{config}} · {{split}} · {{size}}',
  },
};
