# Datenschutzerklärung

Stand: 12. November 2026

## 1. Verantwortlicher

Alexander Kühling
E-Mail: alexander.kuehling@gmx.net

Anschrift: siehe [Impressum](/imprint).

Ein Datenschutzbeauftragter ist nicht benannt (gesetzlich nicht vorgeschrieben).

## 2. Überblick

Brick DNN ist ein Werkzeug, mit dem Sie neuronale Netze aus Bausteinen zusammensetzen, PyTorch-Code erzeugen, Daten aufbereiten und Netze auf diesem Server trainieren können. Die Grundfunktionen sind kostenlos. Für Netze, Daten und Trainings brauchen Sie ein Konto.

- Es gibt **kein Tracking**, keine Analyse- oder Werbedienste, keine Cookies für die App und keine eingebundenen Inhalte anderer Anbieter (Schriften, Skripte usw. kommen von diesem Server). Einzige Ausnahme: der Bezahlvorgang des Zahlungsanbieters Paddle, wenn Sie ein kostenpflichtiges Abo abschließen (Abschnitt 10).
- E-Mails versendet die App nur zu Ihrem Konto (z. B. zum Bestätigen Ihrer Adresse oder Zurücksetzen des Passworts), keine Werbung und keine Newsletter (Abschnitt 5).
- Ihre Daten werden **nicht verkauft** und nur an die unten genannten Dienstleister weitergegeben, die den Server, die Datenbank bzw. den E-Mail-Versand für mich betreiben, sowie an Paddle, wenn Sie ein kostenpflichtiges Abo abschließen (Abschnitt 10).
- Links zu anderen Seiten (z. B. GitHub oder „Buy me a coffee“) werden erst geladen, wenn Sie sie anklicken; dort gelten die Datenschutzbestimmungen des jeweiligen Anbieters.

## 3. Hosting

**Server:** Die App läuft auf einem Server der Infomaniak Network SA, Rue Eugène-Marziano 25, 1227 Les Acacias (Genf), Schweiz, in einem Rechenzentrum in der Schweiz. Dort liegen auch alle hochgeladenen Dateien und Trainingsergebnisse. Mit Infomaniak besteht ein Vertrag zur Auftragsverarbeitung nach Art. 28 DSGVO. Für die Schweiz hat die EU-Kommission festgestellt, dass dort ein angemessenes Datenschutzniveau besteht (Angemessenheitsbeschluss, Art. 45 DSGVO).

**Datenbank:** Konten, Netze, Kommentare und die Angaben zu Daten und Trainings (nicht die Dateien selbst) liegen in einer PostgreSQL-Datenbank des Anbieters Supabase Inc., in der Region London (Vereinigtes Königreich). Mit Supabase besteht ein Vertrag zur Auftragsverarbeitung (Data Processing Addendum) nach Art. 28 DSGVO. Für das Vereinigte Königreich hat die EU-Kommission festgestellt, dass dort ein angemessenes Datenschutzniveau besteht (Angemessenheitsbeschluss, Art. 45 DSGVO). Soweit Supabase oder seine Unterauftragnehmer Daten in andere Drittländer, etwa die USA, übermitteln, geschieht das auf Grundlage der EU-Standardvertragsklauseln (Art. 46 Abs. 2 lit. c DSGVO), die Teil dieses Vertrags sind.

Rechtsgrundlage: Art. 6 Abs. 1 lit. b DSGVO (Bereitstellung des Dienstes) und lit. f (berechtigtes Interesse an einem sicheren, zuverlässigen Betrieb).

## 4. Aufruf der Website und Server-Logdateien

Beim Aufruf der Seite verarbeitet der Server technisch notwendig Ihre IP-Adresse. In den Protokollen (Logdateien) des Webservers werden gespeichert: die **gekürzte** IP-Adresse (bei IPv4 ist der letzte Block 0, bei IPv6 bleiben nur die ersten beiden Blöcke), Datum und Uhrzeit, die aufgerufene Seite ohne Parameter, HTTP-Status, übertragene Datenmenge, Antwortzeit und die Browserkennung (User-Agent). Fehlerprotokolle können vollständige IP-Adressen enthalten. Wenn bei einer Anfrage ein Fehler im Server auftritt, erhalte ich per E-Mail einen technischen Fehlerbericht (aufgerufene Adresse, Benutzername, Programmablauf; keine Passwörter oder Token), um den Fehler zu beheben; höchstens eine E-Mail je Fehlerart und Stunde.

Die Protokolle werden nach Größe automatisch überschrieben (höchstens etwa 30 MB je Dienst); bei wenig Verkehr können sie einige Wochen alt werden.

Zum Schutz vor Missbrauch zählt der Server Anfragen je IP-Adresse bzw. je Konto (Ratenbegrenzung). Diese Zähler liegen höchstens eine Stunde im Zwischenspeicher.

Zweck: Betrieb, Sicherheit, Fehlersuche, Abwehr von Missbrauch. Rechtsgrundlage: Art. 6 Abs. 1 lit. f DSGVO.

## 5. Konto

Wenn Sie ein Konto anlegen, speichere ich:

- Benutzername, E-Mail-Adresse (und ob und wann Sie sie bestätigt haben) und Ihr Passwort (nur als sicherer Hash, nie im Klartext),
- Zeitpunkt der Registrierung, des letzten Logins und der letzten Nutzung,
- ob und wann Sie die Nutzungsbedingungen akzeptiert haben (mit deren Version).

Die E-Mail-Adresse dient dazu, dass andere Sie zu ihren Netzen einladen können; sie wird anderen Nutzern nicht angezeigt. Ihr Benutzername ist für Personen sichtbar, mit denen Sie zusammenarbeiten.

**E-Mails.** Die App sendet E-Mails nur zu Ihrem Konto: einen Link zur Bestätigung Ihrer E-Mail-Adresse (bei der Registrierung und nach einer Änderung), auf Ihren Wunsch einen Link zum Zurücksetzen des Passworts, kurze Sicherheitshinweise (Passwort geändert, E-Mail-Adresse geändert – dieser Hinweis geht an die bisherige Adresse –, Zwei-Faktor-Anmeldung ein- oder ausgeschaltet, neues API-Token, neu verbundene App) und eine Erinnerung, bevor ein inaktives Konto gelöscht wird (Abschnitt 11). Keine Werbung, keine Newsletter. Gespeichert wird, ob und wann Ihre Adresse bestätigt wurde und wann zuletzt an die Löschung erinnert wurde; die E-Mails selbst werden auf dem Server nicht gespeichert. Für den Versand setze ich den E-Mail-Versanddienstleister **Brevo** (Sendinblue SAS, 106 boulevard Haussmann, 75008 Paris, Frankreich) ein. Er erhält Ihre E-Mail-Adresse, Ihren Benutzernamen und den Inhalt der E-Mail und protokolliert Zustellung und Zustellfehler; mit ihm besteht ein Vertrag zur Auftragsverarbeitung nach Art. 28 DSGVO. Die Daten werden in der EU verarbeitet; soweit Unterauftragnehmer von Brevo Daten in Drittländer übermitteln, geschieht das auf Grundlage der EU-Standardvertragsklauseln (Art. 46 Abs. 2 lit. c DSGVO). Rechtsgrundlage: Art. 6 Abs. 1 lit. b DSGVO, für Sicherheitshinweise und Erinnerungen auch lit. f (Schutz Ihres Kontos).

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

Nur der Verwaltungsbereich `/admin/`, den ausschließlich der Betreiber nutzt, verwendet ein Sitzungs- und ein CSRF-Cookie.

## 9. API-Token und verbundene Apps (KI-Assistenten)

Sie können API-Token anlegen oder Apps (z. B. KI-Assistenten über MCP) mit Ihrem Konto verbinden. Gespeichert werden Name, Anfang des Tokens, Zeitpunkte von Erstellung und letzter Nutzung, der Name der App und ein Monatsbudget, falls Sie eines festlegen – vom Token selbst nur ein Hash.

Verbinden Sie eine App, ruft diese auf Ihre Veranlassung Daten Ihres Kontos ab und verarbeitet sie bei ihrem Anbieter (z. B. Anthropic bei Claude). Dafür gelten die Datenschutzbestimmungen dieses Anbieters. Sie können die Verbindung jederzeit unter „API tokens“ trennen.

**Protokoll der Agentenaktionen:** Was ein Token oder eine verbundene App in Ihrem Konto tut (jeder Aufruf über MCP oder die API), wird protokolliert und Ihnen unter „Agent activity“ angezeigt: Name des Tokens bzw. der App, Anfang des Tokens, Kennung des Programms (User-Agent), Aktion, Ergebnis mit Fehlermeldung, Dauer, betroffene Netze, Trainings und Datensätze sowie die Argumente in zusammengefasster Form (Größen und Anzahlen statt Dateiinhalten und Eingabedaten, nie Passwörter oder Token). Nur Sie sehen Ihr Protokoll. Es wird nach 90 Tagen automatisch gelöscht (höchstens 20.000 Einträge je Konto), mit dem Konto gelöscht und ist im Datenexport enthalten. Zweck: Nachvollziehbarkeit und Kontrolle dessen, was Assistenten in Ihrem Namen tun. Rechtsgrundlage: Art. 6 Abs. 1 lit. b DSGVO.

## 10. Kostenpflichtige Tarife: Zahlung über Paddle

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

- **Konto und Inhalte:** bis Sie sie löschen. Unter „Account“ können Sie Ihr Konto jederzeit selbst löschen; dabei werden alle Ihre Netze, Kommentare, hochgeladenen Dateien, Aufbereitungen, Trainings, API-Token, verbundenen Apps, das Protokoll der Agentenaktionen, Nutzungsdaten, die Angaben zur Zwei-Faktor-Anmeldung und zu einem Abo (Abschnitt 10) gelöscht. Netze anderer, an denen Sie mitgearbeitet haben, bleiben bestehen (ohne Hinweis auf Sie als Bearbeiter).
- **Ungenutzte Uploads und Trainings** (hochgeladene Dateien, aufbereitete Datensätze und abgeschlossene Trainings) werden nach 180 Tagen ohne Nutzung automatisch gelöscht.
- **Inaktive Konten** werden nach zwei Jahren (730 Tagen) ohne Login und ohne Nutzung mit allen Inhalten automatisch gelöscht. Die App erinnert Sie 14 und 3 Tage vorher an Ihre E-Mail-Adresse; ein Login genügt, um das Konto zu behalten. Ein Konto mit E-Mail-Adresse wird erst gelöscht, nachdem Sie erinnert wurden.
- **Nutzungsdaten** bleiben gespeichert, solange Ihr Konto besteht, auch wenn ein Training, Token oder eine App gelöscht wird (sie gehören zum Verbrauch des Monats).
- **Logins** (Erneuerungs-Token) verfallen nach 7 Tagen ohne Nutzung und werden danach gelöscht.
- **Protokoll der Agentenaktionen** (Abschnitt 9): nach 90 Tagen.
- **Server-Logs:** siehe Abschnitt 4.
- **Sicherungen:** Gelöschte Daten können noch in Sicherungskopien der Datenbank und der Dateien enthalten sein, bis diese turnusmäßig gelöscht werden: die täglichen Sicherungen auf dem Server nach 7 Tagen, die wöchentlichen nach spätestens 5 Wochen; Sicherungen des Hosters bzw. der Datenbank nach dessen Fristen (in der Regel 7 Tage).

## 12. Ihre Rechte

Sie haben das Recht auf Auskunft (Art. 15 DSGVO), Berichtigung (Art. 16), Löschung (Art. 17), Einschränkung der Verarbeitung (Art. 18), Datenübertragbarkeit (Art. 20) und Widerspruch gegen Verarbeitungen auf Grundlage berechtigter Interessen (Art. 21).

Vieles können Sie selbst unter **Account** erledigen: E-Mail-Adresse ändern, **alle Ihre Daten als ZIP-Datei herunterladen** (maschinenlesbar, JSON und Ihre Dateien) und das **Konto löschen**. Für alles andere schreiben Sie an alexander.kuehling@gmx.net.

Sie haben außerdem das Recht, sich bei einer Datenschutz-Aufsichtsbehörde zu beschweren (Art. 77 DSGVO), insbesondere in dem Mitgliedstaat Ihres Aufenthaltsorts, Ihres Arbeitsplatzes oder des Orts des mutmaßlichen Verstoßes.

## 13. Pflicht zur Bereitstellung, automatisierte Entscheidungen

Für ein Konto sind Benutzername, E-Mail-Adresse und Passwort nötig; ohne sie kann kein Konto angelegt werden. Netze bauen und Code erzeugen geht auch ohne Konto. Es gibt keine automatisierten Entscheidungen oder Profiling im Sinne von Art. 22 DSGVO.

## 14. Änderungen

Ich passe diese Erklärung an, wenn sich die App oder die Rechtslage ändert. Es gilt die hier veröffentlichte Fassung.
