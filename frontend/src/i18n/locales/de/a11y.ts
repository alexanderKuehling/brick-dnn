import type { a11y as en } from '../en/a11y';
import type { Messages } from '../types';

export const a11y: Messages<typeof en> = {
  skipToContent: 'Zum Inhalt springen',
  mainNav: 'Hauptnavigation',
  canvas: {
    label: 'Arbeitsfläche des Netzes',
    nodeDescription:
      'Schicht auf der Arbeitsfläche. Eingabe oder Leertaste wählt sie aus; danach verschieben die Pfeiltasten sie (mit Umschalt schneller), Entf löscht sie, Escape hebt die Auswahl auf. Ihre Einstellungen und Verbindungen stehen im Bereich „Eigenschaften“.',
    nodeDescriptionReadOnly:
      'Schicht auf der Arbeitsfläche. Eingabe oder Leertaste wählt sie aus und zeigt ihre Einstellungen.',
    edgeDescription:
      'Verbindung auf der Arbeitsfläche. Eingabe oder Leertaste wählt sie aus; danach löscht Entf sie, Escape hebt die Auswahl auf.',
    moved: 'Nach {{direction}} verschoben. Position: x {{x}}, y {{y}}.',
    directions: { left: 'links', right: 'rechts', up: 'oben', down: 'unten' },
    controls: 'Steuerung der Arbeitsfläche',
    zoomIn: 'Vergrößern',
    zoomOut: 'Verkleinern',
    fitView: 'Ganzes Netz anzeigen',
    interactive: 'Verschieben der Schichten sperren oder freigeben',
    minimap: 'Übersicht des Netzes',
    handle: 'Verbindungspunkt',
    node: '{{name}}, {{type}}',
    nodeWithErrors: '{{name}}, {{type}}, hat Probleme',
    edge: 'Verbindung von {{source}} zu {{target}}',
  },
  live: {
    added: '{{name}} hinzugefügt.',
    addedAfter: '{{name}} nach {{after}} hinzugefügt.',
    addedInto: '{{name}} hinzugefügt und mit {{into}} verbunden.',
    connected: '{{source}} mit {{target}} verbunden.',
    disconnected: 'Verbindung von {{source}} zu {{target}} entfernt.',
    deleted: '{{name}} gelöscht.',
    deletedMany_one: '{{count}} Schicht gelöscht.',
    deletedMany_other: '{{count}} Schichten gelöscht.',
    selected: '{{name}} ausgewählt.',
  },
  connections: {
    title: 'Verbindungen',
    inputs: 'Eingänge von',
    outputs: 'Ausgänge zu',
    none: 'Keine',
    addInput: 'Eingang hinzufügen von …',
    addOutput: 'Ausgang hinzufügen zu …',
    addInputHint: 'Schichten, deren Ausgabe diese Schicht aufnehmen kann.',
    addOutputHint: 'Schichten, die die Ausgabe dieser Schicht aufnehmen können.',
    noCandidates: 'Keine andere Schicht passt hier.',
    remove: 'Verbindung von {{source}} zu {{target}} entfernen',
    select: '{{name}} auswählen',
  },
  layers: {
    title: 'Schichten',
    label: 'Schichten des Netzes, in der Reihenfolge, in der sie hinzugefügt wurden',
    inputsFrom: 'Eingang von {{names}}',
    noInputs: 'keine Eingänge',
  },
  titles: {
    format: '{{page}} · Brick DNN',
    authorize: 'App verbinden',
    shared: 'Geteiltes Netz',
  },
  charts: {
    noValue: 'kein Wert',
    epoch: 'Epoche',
  },
};
