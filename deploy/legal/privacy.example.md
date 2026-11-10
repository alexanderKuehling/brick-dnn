# Datenschutzerklärung

> **Vorlage, keine Rechtsberatung.** Diese Vorlage beschreibt, was Brick DNN technisch speichert (Stand des Quellcodes, siehe docs/privacy.md). Ersetzen Sie alle Angaben in [eckigen Klammern], streichen Sie, was auf Ihren Server nicht zutrifft (z. B. Supabase), lassen Sie den Text prüfen (Datenschutz-Generator oder anwaltlich) und löschen Sie diesen Hinweis, bevor Sie die Datei als `privacy.md` auf den Server legen.

Stand: [TT.MM.JJJJ]

## 1. Verantwortlicher

[Vorname Nachname]
[Straße Hausnummer]
[PLZ Ort]
E-Mail: [datenschutz@example.com]

Ein Datenschutzbeauftragter ist nicht benannt (als Privatperson nicht vorgeschrieben).

## 2. Überblick

Brick DNN ist ein Werkzeug, mit dem Sie neuronale Netze aus Bausteinen zusammensetzen, PyTorch-Code erzeugen, Daten aufbereiten und Netze auf diesem Server trainieren können. Die Grundfunktionen sind kostenlos. Für Netze, Daten und Trainings brauchen Sie ein Konto.

- Es gibt **kein Tracking**, keine Analyse- oder Werbedienste, keine Cookies für die App und keine eingebundenen Inhalte anderer Anbieter (Schriften, Skripte usw. kommen von diesem Server). Einzige Ausnahme: der Bezahlvorgang des Zahlungsanbieters Paddle, wenn Sie ein kostenpflichtiges Abo abschließen (Abschnitt 10).
- [Ohne E-Mail-Versand:] Die App versendet **keine E-Mails**. [Mit E-Mail-Versand (`BRICKDNN_EMAIL_HOST`):] E-Mails versendet die App nur zu Ihrem Konto (z. B. zum Zurücksetzen des Passworts), keine Werbung und keine Newsletter (Abschnitt 5).
- Ihre Daten werden **nicht verkauft** und nur an die unten genannten Dienstleister weitergegeben, die den Server bzw. die Datenbank für mich betreiben.

## 3. Hosting

**Server:** Die App läuft auf einem Server der Hetzner Online GmbH, Industriestr. 25, 91710 Gunzenhausen, Deutschland, im Rechenzentrum [Standort, z. B. Nürnberg/Falkenstein]. Dort liegen auch alle hochgeladenen Dateien und Trainingsergebnisse. Mit Hetzner besteht ein Vertrag zur Auftragsverarbeitung nach Art. 28 DSGVO.

[Nur wenn Sie Supabase nutzen:] **Datenbank:** Konten, Netze, Kommentare und die Angaben zu Daten und Trainings (nicht die Dateien selbst) liegen in einer PostgreSQL-Datenbank bei Supabase ([Anbieter und Anschrift laut Supabase-DPA]), Region [z. B. EU Central (Frankfurt)]. Mit Supabase besteht ein Vertrag zur Auftragsverarbeitung (DPA). [Prüfen und beschreiben: Ob Daten in ein Drittland (z. B. USA) übermittelt werden können und auf welcher Grundlage, z. B. EU-Standardvertragsklauseln oder EU-US Data Privacy Framework.]

[Ohne Supabase:] Die Datenbank liegt auf demselben Server bei Hetzner.

Rechtsgrundlage: Art. 6 Abs. 1 lit. b DSGVO (Bereitstellung des Dienstes) und lit. f (berechtigtes Interesse an einem sicheren, zuverlässigen Betrieb).

## 4. Aufruf der Website und Server-Logdateien

Beim Aufruf der Seite verarbeitet der Server technisch notwendig Ihre IP-Adresse. In den Protokollen (Logdateien) des Webservers werden gespeichert: die **gekürzte** IP-Adresse (bei IPv4 ist der letzte Block 0, bei IPv6 bleiben nur die ersten beiden Blöcke), Datum und Uhrzeit, die aufgerufene Seite ohne Parameter, HTTP-Status, übertragene Datenmenge, Antwortzeit und die Browserkennung (User-Agent). Fehlerprotokolle können vollständige IP-Adressen enthalten. [Falls `BRICKDNN_ADMIN_EMAIL` gesetzt ist:] Wenn bei einer Anfrage ein Fehler im Server auftritt, erhält der Betreiber per E-Mail einen technischen Fehlerbericht (aufgerufene Adresse, Benutzername, Programmablauf; keine Passwörter oder Token), um den Fehler zu beheben; höchstens eine E-Mail je Fehlerart und Stunde.

Die Protokolle werden nach Größe automatisch überschrieben (höchstens etwa 30 MB je Dienst); bei wenig Verkehr können sie [einige Wochen] alt werden. [Falls Sie Backups des Servers machen: Backups werden nach [X] Tagen gelöscht.]

Zum Schutz vor Missbrauch zählt der Server Anfragen je IP-Adresse bzw. je Konto (Ratenbegrenzung). Diese Zähler liegen höchstens eine Stunde im Zwischenspeicher.

Zweck: Betrieb, Sicherheit, Fehlersuche, Abwehr von Missbrauch. Rechtsgrundlage: Art. 6 Abs. 1 lit. f DSGVO.

## 5. Konto

Wenn Sie ein Konto anlegen, speichere ich:

- Benutzername, E-Mail-Adresse (und ob und wann Sie sie bestätigt haben) und Ihr Passwort (nur als sicherer Hash, nie im Klartext),
- Zeitpunkt der Registrierung, des letzten Logins und der letzten Nutzung,
- ob und wann Sie die Nutzungsbedingungen akzeptiert haben (mit deren Version).

Die E-Mail-Adresse dient dazu, dass andere Sie zu ihren Netzen einladen können; sie wird anderen Nutzern nicht angezeigt. Ihr Benutzername ist für Personen sichtbar, mit denen Sie zusammenarbeiten.

[Nur mit E-Mail-Versand:]
**E-Mails.** Wenn auf diesem Server der E-Mail-Versand eingerichtet ist, sendet die App E-Mails nur zu Ihrem Konto: einen Link zur Bestätigung Ihrer E-Mail-Adresse (bei der Registrierung und nach einer Änderung), auf Ihren Wunsch einen Link zum Zurücksetzen des Passworts, kurze Sicherheitshinweise (Passwort geändert, E-Mail-Adresse geändert – dieser Hinweis geht an die bisherige Adresse –, Zwei-Faktor-Anmeldung ein- oder ausgeschaltet, neues API-Token, neu verbundene App) und eine Erinnerung, bevor ein inaktives Konto gelöscht wird (Abschnitt 11). Keine Werbung, keine Newsletter. Gespeichert wird, ob und wann Ihre Adresse bestätigt wurde und wann zuletzt an die Löschung erinnert wurde; die E-Mails selbst werden auf dem Server nicht gespeichert. Für den Versand setze ich den E-Mail-Versanddienstleister **[Name, Anschrift und Sitzland des E-Mail-Versanddienstleisters, z. B. Sendinblue SAS (Brevo), 106 boulevard Haussmann, 75008 Paris, Frankreich]** ein. Er erhält Ihre E-Mail-Adresse, Ihren Benutzernamen und den Inhalt der E-Mail; mit ihm besteht ein Vertrag zur Auftragsverarbeitung nach Art. 28 DSGVO. [Falls der Dienstleister Daten außerhalb der EU verarbeitet: Grundlage der Übermittlung, z. B. EU-Standardvertragsklauseln oder Angemessenheitsbeschluss.] Rechtsgrundlage: Art. 6 Abs. 1 lit. b DSGVO, für Sicherheitshinweise und Erinnerungen auch lit. f (Schutz Ihres Kontos).

**Zwei-Faktor-Anmeldung.** Wenn Sie sie unter „Account“ einschalten, speichere ich den geheimen Schlüssel für Ihre Authenticator-App (verschlüsselt), wann Sie sie eingerichtet haben, welcher Code zuletzt verwendet wurde (damit jeder Code nur einmal gilt) und Ihre Wiederherstellungscodes (nur als Hash). Die Codes erzeugt die App auf Ihrem Gerät selbst; dabei werden keine Daten an mich übertragen. Die Angaben werden gelöscht, wenn Sie die Zwei-Faktor-Anmeldung ausschalten oder das Konto löschen. Rechtsgrundlage: Art. 6 Abs. 1 lit. b DSGVO.

Zweck: Bereitstellung des Kontos und der Funktionen. Rechtsgrundlage: Art. 6 Abs. 1 lit. b DSGVO.

## 6. Ihre Inhalte

Gespeichert werden die Inhalte, die Sie anlegen:

- **Netze** mit Name, Beschreibung, Aufbau und Versionsverlauf; **Kommentare**;
- **Freigaben:** Personen, die Sie einladen, sehen das Netz, seinen Verlauf und die Kommentare. Mit einem Freigabe-Link kann jede Person mit dem Link das Netz ansehen;
- **hochgeladene Daten:** Dateien des Daten-Tabs (CSV, Excel, JSON, Parquet u. a.) und Trainingsdaten (CSV oder ZIP mit Bildern) als Dateien auf dem Server, dazu Aufbereitungsschritte und Einstellungen. Beim Import über einen Link ruft der Server die Datei bei dem Anbieter ab, auf den der Link zeigt (z. B. GitHub, Google Drive, Dropbox); dieser sieht dabei die IP-Adresse des Servers, nicht Ihre. Gespeichert wird der Link ohne seinen Query-Teil (alles ab „?“);
- **Trainings:** Netz, Einstellungen, Verlauf der Kennzahlen, Protokoll und trainierte Gewichte. Trainings laufen auf diesem Server; die Daten verlassen ihn dafür nicht.
- **Nutzung:** die Rechenzeit (CPU-Zeit) jedes Trainings und jeder Vorhersage je Kalendermonat, der Anteil über dem Kontingent Ihres Tarifs, welches API-Token oder welche App sie ausgelöst hat (mit deren Namen) und der Name des Trainings; dazu Ihr Tarif, Ihre Budgets und Grenzen. Damit werden die Grenzen Ihres Tarifs und Ihre Budgets durchgesetzt und Ihnen Ihr Verbrauch angezeigt.

**Bitte laden Sie keine personenbezogenen oder vertraulichen Daten hoch.** Die App weist beim Hochladen darauf hin.

Rechtsgrundlage: Art. 6 Abs. 1 lit. b DSGVO.

## 7. Gemeinsames Bearbeiten in Echtzeit

Wenn Sie ein geteiltes Netz geöffnet haben, hält Ihr Browser eine WebSocket-Verbindung zum Server. Darüber sehen die anderen Personen im Netz Ihren Benutzernamen und was Sie gerade auswählen oder verschieben. Diese Nachrichten werden nur im Arbeitsspeicher weitergereicht (Redis auf demselben Server) und nicht gespeichert; Änderungen am Netz werden wie gewohnt gespeichert. Rechtsgrundlage: Art. 6 Abs. 1 lit. b DSGVO.

## 8. Speicher im Browser (keine Cookies)

Die App setzt keine Cookies. Sie speichert im lokalen Speicher Ihres Browsers (localStorage):

- Ihre Anmeldung (Zugangs- und Erneuerungs-Token; ein Login gilt höchstens 7 Tage ohne Nutzung),
- den gerade bearbeiteten Entwurf eines Netzes, zuletzt gewählte Trainings- und Daten-Einstellungen und das Farbschema (hell/dunkel).

Diese Einträge sind für die von Ihnen gewünschten Funktionen unbedingt erforderlich (§ 25 Abs. 2 Nr. 2 TDDDG). Beim Abmelden wird die Anmeldung gelöscht; die übrigen Einträge können Sie über die Browser-Einstellungen löschen.

[Nur für den Betreiber selbst: Der Verwaltungsbereich `/admin/` verwendet ein Sitzungs- und ein CSRF-Cookie.]

## 9. API-Token und verbundene Apps (KI-Assistenten)

Sie können API-Token anlegen oder Apps (z. B. KI-Assistenten über MCP) mit Ihrem Konto verbinden. Gespeichert werden Name, Anfang des Tokens, Zeitpunkte von Erstellung und letzter Nutzung, der Name der App und ein Monatsbudget, falls Sie eines festlegen – vom Token selbst nur ein Hash.

Verbinden Sie eine App, ruft diese auf Ihre Veranlassung Daten Ihres Kontos ab und verarbeitet sie bei ihrem Anbieter (z. B. Anthropic bei Claude). Dafür gelten die Datenschutzbestimmungen dieses Anbieters. Sie können die Verbindung jederzeit unter „API tokens“ trennen.

**Protokoll der Agentenaktionen:** Was ein Token oder eine verbundene App in Ihrem Konto tut (jeder Aufruf über MCP oder die API), wird protokolliert und Ihnen unter „Agent activity“ angezeigt: Name des Tokens bzw. der App, Anfang des Tokens, Kennung des Programms (User-Agent), Aktion, Ergebnis mit Fehlermeldung, Dauer, betroffene Netze, Trainings und Datensätze sowie die Argumente in zusammengefasster Form (Größen und Anzahlen statt Dateiinhalten und Eingabedaten, nie Passwörter oder Token). Nur Sie sehen Ihr Protokoll. Es wird nach [90] Tagen automatisch gelöscht (höchstens 20.000 Einträge je Konto), mit dem Konto gelöscht und ist im Datenexport enthalten. Zweck: Nachvollziehbarkeit und Kontrolle dessen, was Assistenten in Ihrem Namen tun. Rechtsgrundlage: Art. 6 Abs. 1 lit. b DSGVO.

## 10. Kostenpflichtige Tarife: Zahlung über Paddle

[Nur wenn Sie kostenpflichtige Tarife über Paddle anbieten (docs/billing.md); sonst diesen Abschnitt und die Verweise darauf in Abschnitt 2 streichen. Anschrift von Paddle und Link zur Datenschutzerklärung mit den aktuellen Angaben von Paddle abgleichen.]

Die Grundfunktionen von Brick DNN sind kostenlos. Dieser Abschnitt betrifft Sie nur, wenn Sie ein kostenpflichtiges Abo abschließen (soweit auf der Seite „Pricing“ Tarife angeboten werden).

Verkäufer der Tarife ist nicht der Betreiber dieser Seite, sondern **Paddle.com Market Ltd** mit Sitz in London, Vereinigtes Königreich („Paddle“). Paddle verkauft die Tarife als Wiederverkäufer (Merchant of Record), wickelt die Zahlung ab, stellt die Rechnung aus und führt die Umsatzsteuer ab. Für diese Verarbeitung ist Paddle selbst verantwortlich; es gilt die Datenschutzerklärung von Paddle (https://www.paddle.com/legal).

- **Was an Paddle übermittelt wird:** Wenn Sie den Bezahlvorgang öffnen, legt der Server für Ihr Konto einen Kundeneintrag bei Paddle an und übermittelt dafür Ihre E-Mail-Adresse und die interne Nummer Ihres Kontos, damit der Kauf Ihrem Konto zugeordnet werden kann. Ändern Sie später Ihre E-Mail-Adresse, wird sie auch bei Paddle geändert.
- **Was Sie bei Paddle eingeben:** Zahlungsdaten (z. B. Karte, PayPal), Land und Postleitzahl bzw. Rechnungsanschrift und gegebenenfalls eine Umsatzsteuer-ID geben Sie direkt bei Paddle ein. Zahlungsdaten wie Kartennummern erreichen diesen Server nicht. Im Händlerkonto bei Paddle sind die Angaben zu Ihrem Kauf einsehbar (z. B. E-Mail-Adresse, Land, Tarif, Beträge, Zahlungsstatus).
- **Inhalte von Paddle im Browser:** Erst wenn Sie den Bezahlvorgang öffnen (oder einem Zahlungslink von Paddle folgen), lädt Ihr Browser ein Skript und das Bezahlformular von Servern von Paddle. Paddle erhält dabei Ihre IP-Adresse und Angaben zu Ihrem Browser und kann eigene Cookies setzen, etwa zur Betrugsprävention. Vorher wird nichts von Paddle geladen, auch nicht auf der Preisseite. Das Kundenportal für Zahlungsmittel und Rechnungen ist eine Seite von Paddle.
- **Was dieser Server speichert:** Ihre Kunden- und Abonummer bei Paddle, Tarif, Status und Zeitraum des Abos, zu jeder Mitteilung von Paddle über Ihr Abo Art, Zeitpunkt und Abonummer (nicht ihren Inhalt) sowie abgerechnete zusätzliche Rechenzeit. Diese Angaben stehen im Daten-Export und werden mit dem Konto gelöscht.
- **E-Mails:** Bestätigungen, Rechnungen und Zahlungshinweise versendet Paddle an Ihre E-Mail-Adresse; die App selbst versendet dazu keine E-Mails.
- **Speicherdauer bei Paddle:** Rechnungen und Zahlungsdaten bewahrt Paddle nach den für Paddle geltenden gesetzlichen Fristen auf, auch nachdem Ihr Konto hier gelöscht wurde. Wenn Sie Ihr Konto löschen, wird ein laufendes Abo sofort beendet.
- **Drittland:** Für das Vereinigte Königreich besteht ein Angemessenheitsbeschluss der EU-Kommission (Art. 45 DSGVO). Welche weiteren Dienstleister Paddle einsetzt, auch außerhalb der EU, steht in der Datenschutzerklärung von Paddle.

Zweck: Abschluss und Durchführung des Abos, Zuordnung des Tarifs zu Ihrem Konto. Rechtsgrundlage: Art. 6 Abs. 1 lit. b DSGVO.

## 11. Speicherdauer und Löschung

- **Konto und Inhalte:** bis Sie sie löschen. Unter „Account“ können Sie Ihr Konto jederzeit selbst löschen; dabei werden alle Ihre Netze, Kommentare, hochgeladenen Dateien, Aufbereitungen, Trainings, API-Token, verbundenen Apps, das Protokoll der Agentenaktionen, Nutzungsdaten und die Angaben zur Zwei-Faktor-Anmeldung gelöscht. Netze anderer, an denen Sie mitgearbeitet haben, bleiben bestehen (ohne Hinweis auf Sie als Bearbeiter).
- [Falls eingestellt:] **Ungenutzte Uploads und Trainings** werden nach [X] Tagen ohne Nutzung automatisch gelöscht.
- [Falls eingestellt:] **Inaktive Konten** werden nach [Y] Tagen ohne Login mit allen Inhalten automatisch gelöscht. [Mit E-Mail-Versand:] Die App erinnert Sie [14 und 3] Tage vorher an Ihre E-Mail-Adresse; ein Login genügt, um das Konto zu behalten. Ein Konto mit E-Mail-Adresse wird erst gelöscht, nachdem Sie erinnert wurden. [Ohne E-Mail-Versand:] Eine Erinnerung vorher gibt es nicht.
- **Nutzungsdaten** bleiben gespeichert, solange Ihr Konto besteht, auch wenn ein Training, Token oder eine App gelöscht wird (sie gehören zum Verbrauch des Monats).
- **Logins** (Erneuerungs-Token) verfallen nach 7 Tagen ohne Nutzung und werden danach gelöscht.
- **Protokoll der Agentenaktionen:** nach [90] Tagen (`BRICKDNN_AGENT_LOG_DAYS`).
- **Server-Logs:** siehe Abschnitt 4.
- [Backups: werden nach [Z] Tagen überschrieben; gelöschte Daten können bis dahin noch in Backups enthalten sein.]

## 12. Ihre Rechte

Sie haben das Recht auf Auskunft (Art. 15 DSGVO), Berichtigung (Art. 16), Löschung (Art. 17), Einschränkung der Verarbeitung (Art. 18), Datenübertragbarkeit (Art. 20) und Widerspruch gegen Verarbeitungen auf Grundlage berechtigter Interessen (Art. 21).

Vieles können Sie selbst unter **Account** erledigen: E-Mail-Adresse ändern, **alle Ihre Daten als ZIP-Datei herunterladen** (maschinenlesbar, JSON und Ihre Dateien) und das **Konto löschen**. Für alles andere schreiben Sie an [datenschutz@example.com].

Sie haben außerdem das Recht, sich bei einer Datenschutz-Aufsichtsbehörde zu beschweren (Art. 77 DSGVO), z. B. bei der für mich zuständigen Behörde: [Name und Anschrift der Landesdatenschutzbehörde Ihres Bundeslandes].

## 13. Pflicht zur Bereitstellung, automatisierte Entscheidungen

Für ein Konto sind Benutzername, E-Mail-Adresse und Passwort nötig; ohne sie kann kein Konto angelegt werden. Netze bauen und Code erzeugen geht auch ohne Konto. Es gibt keine automatisierten Entscheidungen oder Profiling im Sinne von Art. 22 DSGVO.

## 14. Änderungen

Ich passe diese Erklärung an, wenn sich die App oder die Rechtslage ändert. Es gilt die hier veröffentlichte Fassung.
