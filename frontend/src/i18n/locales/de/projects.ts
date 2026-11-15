import type { projects as en } from '../en/projects';
import type { Messages } from '../types';

export const projects: Messages<typeof en> = {
  renameTitle: 'Netz umbenennen',
  renameFailed: 'Das Netz konnte nicht umbenannt werden',
  created: '„{{name}}“ erstellt.',
  savedToMine: '„{{name}}“ in Ihren eigenen Netzen („Meine“) gespeichert.',
  deleteTitle: '„{{name}}“ löschen?',
  deleteText: 'Das lässt sich nicht rückgängig machen.',
  deleteShared:
    'Alle, mit denen Sie es geteilt haben, verlieren ebenfalls den Zugriff. Das lässt sich nicht rückgängig machen.',
  deleteFailed: 'Das Netz konnte nicht gelöscht werden',
  leaveTitle: '„{{name}}“ verlassen?',
  leaveText: 'Sie verlieren den Zugriff auf dieses Netz. Nur {{owner}} kann Sie erneut einladen.',
  leave: 'Verlassen',
  left: 'Sie haben „{{name}}“ verlassen.',
  leaveFailed: 'Das Netz konnte nicht verlassen werden',
  emptyMine:
    'Sie haben noch keine Netze gespeichert. Bauen Sie eines im Editor und klicken Sie auf „Speichern“.',
  emptyShared:
    'Noch hat niemand ein Netz mit Ihnen geteilt. Wenn Sie jemand einlädt, erscheint es hier.',
  mine: 'Meine',
  sharedWithMe: 'Mit mir geteilt',
  actions: 'Aktionen für das Netz',
  sharedData: 'Geteilte Daten …',
  by: 'von {{user}}',
  roles: { owner: 'Eigentümer', editor: 'Darf bearbeiten', viewer: 'Darf ansehen' },
  layers_one: '{{count}} Schicht',
  layers_other: '{{count}} Schichten',
  shared: 'Geteilt',
  updated: 'Aktualisiert {{date}}',
  updatedBy: 'Aktualisiert {{date}} von {{user}}',
  openInBuilder: 'Im Editor öffnen',
};
