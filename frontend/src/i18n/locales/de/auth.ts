import type { auth as enAuth, legal as enLegal } from '../en/auth';
import type { Messages } from '../types';

export const auth: Messages<typeof enAuth> = {
  username: 'Benutzername',
  password: 'Passwort',
  email: 'E-Mail',
  logIn: 'Anmelden',
  backToLogin: 'Zurück zur Anmeldung',
  passwordHint: 'Mindestens 8 Zeichen, nicht zu gebräuchlich.',
  login: {
    title: 'Willkommen zurück',
    noAccount: 'Noch kein Konto? <anchor>Jetzt registrieren</anchor>',
    forgot: 'Passwort vergessen?',
    failed: 'Die Anmeldung ist fehlgeschlagen.',
    twoFactorTitle: 'Zwei-Faktor-Authentifizierung',
    loggingInAs: 'Anmeldung als {{username}}',
    expired:
      'Die Anmeldung ist abgelaufen oder wurde abgebrochen. Geben Sie Ihr Passwort erneut ein.',
    enterRecovery: 'Geben Sie einen Ihrer Wiederherstellungscodes ein. Jeder Code gilt einmal.',
    enterCode: 'Geben Sie den 6-stelligen Code aus Ihrer Authenticator-App ein.',
    recoveryCode: 'Wiederherstellungscode',
    code: 'Code',
    verify: 'Bestätigen',
    useApp: 'Authenticator-App verwenden',
    useRecovery: 'Wiederherstellungscode verwenden',
    codeRejected: 'Der Code wurde nicht akzeptiert.',
  },
  register: {
    title: 'Konto erstellen',
    already: 'Schon registriert? <anchor>Anmelden</anchor>',
    intro:
      'Mit einem Konto können Sie Netze auf diesem Server speichern. Netze bauen und Code exportieren geht auch ohne.',
    emailHint: 'Damit können andere Sie zu ihren Netzen einladen.',
    acceptTerms: 'Ich stimme den <terms>Nutzungsbedingungen</terms> zu.',
    acceptTermsAndPrivacy:
      'Ich stimme den <terms>Nutzungsbedingungen</terms> zu und habe die <privacy>Datenschutzerklärung</privacy> gelesen.',
    privacyNote:
      'Wie dieser Server mit Ihren Daten umgeht: <privacy>Datenschutzerklärung</privacy>.',
    submit: 'Konto erstellen',
    failed: 'Die Registrierung ist fehlgeschlagen.',
  },
  forgot: {
    title: 'Passwort vergessen?',
    unavailable:
      'Auf diesem Server kann das Passwort nicht zurückgesetzt werden: Er verschickt keine E-Mails. Wenden Sie sich an den Betreiber.',
    checkMail: 'Sehen Sie in Ihr Postfach',
    nothingArrived:
      'Nichts angekommen? Sehen Sie im Spam-Ordner nach. Konten, deren E-Mail-Adresse nie bestätigt wurde, können keinen Link erhalten; wenden Sie sich an den Betreiber dieser Seite.',
    intro:
      'Geben Sie Ihren Benutzernamen oder Ihre E-Mail-Adresse ein. Wir senden einen Link zum Festlegen eines neuen Passworts an die bestätigte E-Mail-Adresse Ihres Kontos.',
    login: 'Benutzername oder E-Mail-Adresse',
    submit: 'Link senden',
  },
  reset: {
    title: 'Neues Passwort festlegen',
    mismatch: 'Die beiden Passwörter stimmen nicht überein.',
    changedTitle: 'Passwort geändert',
    changed:
      'Ihr Passwort wurde geändert und Sie wurden überall abgemeldet. Melden Sie sich mit dem neuen Passwort an.',
    invalidTitle: 'Dieser Link funktioniert nicht',
    invalid:
      'Der Link ist ungültig, abgelaufen (er gilt eine Stunde) oder wurde bereits verwendet.',
    newLink: 'Neuen Link anfordern',
    chooseFor: 'Legen Sie ein neues Passwort für <b>{{username}}</b> fest.',
    newPassword: 'Neues Passwort',
    repeat: 'Neues Passwort wiederholen',
    submit: 'Passwort ändern',
  },
  verify: {
    title: 'E-Mail-Adresse bestätigen',
    failed: 'Die Adresse konnte nicht bestätigt werden.',
    thanks: 'Vielen Dank',
    confirmedAddress:
      '{{email}} ist bestätigt. Falls Sie Ihr Passwort einmal vergessen, können Sie es über einen Link an diese Adresse zurücksetzen.',
    confirmed:
      'Ihre E-Mail-Adresse ist bestätigt. Falls Sie Ihr Passwort einmal vergessen, können Sie es über einen Link an diese Adresse zurücksetzen.',
    toAccount: 'Zu Ihrem Konto',
    notConfirmed: 'Nicht bestätigt',
    logOutOf: '{{username}} abmelden',
    thisAccount: 'dieses Konto',
    sendNew: 'Neuen Link über Ihr Konto senden',
  },
  notFound: {
    title: 'Seite nicht gefunden',
    text: 'Diese Seite gibt es nicht.',
    back: 'Zurück zum Editor',
  },
  requireLogin: 'Melden Sie sich an, um diese Seite zu sehen.',
};

export const legal: Messages<typeof enLegal> = {
  labels: {
    imprint: 'Impressum',
    privacy: 'Datenschutzerklärung',
    terms: 'Nutzungsbedingungen',
    refund: 'Erstattung',
  },
  short: {
    imprint: 'Impressum',
    privacy: 'Datenschutz',
    terms: 'AGB',
    refund: 'Erstattung',
  },
  contact: 'Kontakt',
  nav: 'Rechtliches',
  missing: 'Diese Seite ist auf diesem Server nicht eingerichtet.',
  lastChanged: 'Zuletzt geändert: {{date}}',
  allLanguages: 'Alle Sprachen anzeigen',
};
