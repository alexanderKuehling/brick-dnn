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

Brick DNN ist ein kostenloses Werkzeug, mit dem Sie neuronale Netze aus Bausteinen zusammensetzen, PyTorch-Code erzeugen, Daten aufbereiten und Netze auf diesem Server trainieren können. Für Netze, Daten und Trainings brauchen Sie ein Konto.

- Es gibt **kein Tracking**, keine Analyse- oder Werbedienste, keine Cookies für die App und keine eingebundenen Inhalte anderer Anbieter (Schriften, Skripte usw. kommen von diesem Server).
- Die App versendet **keine E-Mails**.
- Ihre Daten werden **nicht verkauft** und nur an die unten genannten Dienstleister weitergegeben, die den Server bzw. die Datenbank für mich betreiben.

## 3. Hosting

**Server:** Die App läuft auf einem Server der Hetzner Online GmbH, Industriestr. 25, 91710 Gunzenhausen, Deutschland, im Rechenzentrum [Standort, z. B. Nürnberg/Falkenstein]. Dort liegen auch alle hochgeladenen Dateien und Trainingsergebnisse. Mit Hetzner besteht ein Vertrag zur Auftragsverarbeitung nach Art. 28 DSGVO.

[Nur wenn Sie Supabase nutzen:] **Datenbank:** Konten, Netze, Kommentare und die Angaben zu Daten und Trainings (nicht die Dateien selbst) liegen in einer PostgreSQL-Datenbank bei Supabase ([Anbieter und Anschrift laut Supabase-DPA]), Region [z. B. EU Central (Frankfurt)]. Mit Supabase besteht ein Vertrag zur Auftragsverarbeitung (DPA). [Prüfen und beschreiben: Ob Daten in ein Drittland (z. B. USA) übermittelt werden können und auf welcher Grundlage, z. B. EU-Standardvertragsklauseln oder EU-US Data Privacy Framework.]

[Ohne Supabase:] Die Datenbank liegt auf demselben Server bei Hetzner.

Rechtsgrundlage: Art. 6 Abs. 1 lit. b DSGVO (Bereitstellung des Dienstes) und lit. f (berechtigtes Interesse an einem sicheren, zuverlässigen Betrieb).

## 4. Aufruf der Website und Server-Logdateien

Beim Aufruf der Seite verarbeitet der Server technisch notwendig Ihre IP-Adresse. In den Protokollen (Logdateien) des Webservers werden gespeichert: die **gekürzte** IP-Adresse (bei IPv4 ist der letzte Block 0, bei IPv6 bleiben nur die ersten beiden Blöcke), Datum und Uhrzeit, die aufgerufene Seite ohne Parameter, HTTP-Status, übertragene Datenmenge, Antwortzeit und die Browserkennung (User-Agent). Fehlerprotokolle können vollständige IP-Adressen enthalten.

Die Protokolle werden nach Größe automatisch überschrieben (höchstens etwa 30 MB je Dienst); bei wenig Verkehr können sie [einige Wochen] alt werden. [Falls Sie Backups des Servers machen: Backups werden nach [X] Tagen gelöscht.]

Zum Schutz vor Missbrauch zählt der Server Anfragen je IP-Adresse bzw. je Konto (Ratenbegrenzung). Diese Zähler liegen höchstens eine Stunde im Zwischenspeicher.

Zweck: Betrieb, Sicherheit, Fehlersuche, Abwehr von Missbrauch. Rechtsgrundlage: Art. 6 Abs. 1 lit. f DSGVO.

## 5. Konto

Wenn Sie ein Konto anlegen, speichere ich:

- Benutzername, E-Mail-Adresse und Ihr Passwort (nur als sicherer Hash, nie im Klartext),
- Zeitpunkt der Registrierung, des letzten Logins und der letzten Nutzung,
- ob und wann Sie die Nutzungsbedingungen akzeptiert haben (mit deren Version).

Die E-Mail-Adresse dient dazu, dass andere Sie zu ihren Netzen einladen können; sie wird anderen Nutzern nicht angezeigt. Ihr Benutzername ist für Personen sichtbar, mit denen Sie zusammenarbeiten.

Zweck: Bereitstellung des Kontos und der Funktionen. Rechtsgrundlage: Art. 6 Abs. 1 lit. b DSGVO.

## 6. Ihre Inhalte

Gespeichert werden die Inhalte, die Sie anlegen:

- **Netze** mit Name, Beschreibung, Aufbau und Versionsverlauf; **Kommentare**;
- **Freigaben:** Personen, die Sie einladen, sehen das Netz, seinen Verlauf und die Kommentare. Mit einem Freigabe-Link kann jede Person mit dem Link das Netz ansehen;
- **hochgeladene Daten:** Dateien des Daten-Tabs (CSV, Excel, JSON, Parquet u. a.) und Trainingsdaten (CSV oder ZIP mit Bildern) als Dateien auf dem Server, dazu Aufbereitungsschritte und Einstellungen. Beim Import über einen Link ruft der Server die Datei bei dem Anbieter ab, auf den der Link zeigt (z. B. GitHub, Google Drive, Dropbox); dieser sieht dabei die IP-Adresse des Servers, nicht Ihre. Gespeichert wird der Link ohne seinen Query-Teil (alles ab „?“);
- **Trainings:** Netz, Einstellungen, Verlauf der Kennzahlen, Protokoll und trainierte Gewichte. Trainings laufen auf diesem Server; die Daten verlassen ihn dafür nicht.

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

Sie können API-Token anlegen oder Apps (z. B. KI-Assistenten über MCP) mit Ihrem Konto verbinden. Gespeichert werden Name, Anfang des Tokens, Zeitpunkte von Erstellung und letzter Nutzung sowie der Name der App – vom Token selbst nur ein Hash.

Verbinden Sie eine App, ruft diese auf Ihre Veranlassung Daten Ihres Kontos ab und verarbeitet sie bei ihrem Anbieter (z. B. Anthropic bei Claude). Dafür gelten die Datenschutzbestimmungen dieses Anbieters. Sie können die Verbindung jederzeit unter „API tokens“ trennen.

## 10. Speicherdauer und Löschung

- **Konto und Inhalte:** bis Sie sie löschen. Unter „Account“ können Sie Ihr Konto jederzeit selbst löschen; dabei werden alle Ihre Netze, Kommentare, hochgeladenen Dateien, Aufbereitungen, Trainings, API-Token und verbundenen Apps gelöscht. Netze anderer, an denen Sie mitgearbeitet haben, bleiben bestehen (ohne Hinweis auf Sie als Bearbeiter).
- [Falls eingestellt:] **Ungenutzte Uploads und Trainings** werden nach [X] Tagen ohne Nutzung automatisch gelöscht.
- [Falls eingestellt:] **Inaktive Konten** werden nach [Y] Tagen ohne Login mit allen Inhalten automatisch gelöscht.
- **Logins** (Erneuerungs-Token) verfallen nach 7 Tagen ohne Nutzung und werden danach gelöscht.
- **Server-Logs:** siehe Abschnitt 4.
- [Backups: werden nach [Z] Tagen überschrieben; gelöschte Daten können bis dahin noch in Backups enthalten sein.]

## 11. Ihre Rechte

Sie haben das Recht auf Auskunft (Art. 15 DSGVO), Berichtigung (Art. 16), Löschung (Art. 17), Einschränkung der Verarbeitung (Art. 18), Datenübertragbarkeit (Art. 20) und Widerspruch gegen Verarbeitungen auf Grundlage berechtigter Interessen (Art. 21).

Vieles können Sie selbst unter **Account** erledigen: E-Mail-Adresse ändern, **alle Ihre Daten als ZIP-Datei herunterladen** (maschinenlesbar, JSON und Ihre Dateien) und das **Konto löschen**. Für alles andere schreiben Sie an [datenschutz@example.com].

Sie haben außerdem das Recht, sich bei einer Datenschutz-Aufsichtsbehörde zu beschweren (Art. 77 DSGVO), z. B. bei der für mich zuständigen Behörde: [Name und Anschrift der Landesdatenschutzbehörde Ihres Bundeslandes].

## 12. Pflicht zur Bereitstellung, automatisierte Entscheidungen

Für ein Konto sind Benutzername, E-Mail-Adresse und Passwort nötig; ohne sie kann kein Konto angelegt werden. Netze bauen und Code erzeugen geht auch ohne Konto. Es gibt keine automatisierten Entscheidungen oder Profiling im Sinne von Art. 22 DSGVO.

## 13. Änderungen

Ich passe diese Erklärung an, wenn sich die App oder die Rechtslage ändert. Es gilt die hier veröffentlichte Fassung.
