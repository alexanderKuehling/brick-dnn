# Datenschutzerklärung

Stand: 9. November 2026

## 1. Verantwortlicher

Alexander Kühling
E-Mail: alexander.kuehling@gmx.net

Ein Datenschutzbeauftragter ist nicht benannt (als Privatperson nicht vorgeschrieben).

## 2. Überblick

Brick DNN ist ein kostenloses Werkzeug, mit dem Sie neuronale Netze aus Bausteinen zusammensetzen, PyTorch-Code erzeugen, Daten aufbereiten und Netze auf diesem Server trainieren können. Für Netze, Daten und Trainings brauchen Sie ein Konto.

- Es gibt **kein Tracking**, keine Analyse- oder Werbedienste, keine Cookies für die App und keine eingebundenen Inhalte anderer Anbieter (Schriften, Skripte usw. kommen von diesem Server).
- Die App versendet **keine E-Mails**.
- Ihre Daten werden **nicht verkauft** und nur an die unten genannten Dienstleister weitergegeben, die den Server bzw. die Datenbank für mich betreiben.
- Links zu anderen Seiten (z. B. GitHub oder „Buy me a coffee“) werden erst geladen, wenn Sie sie anklicken; dort gelten die Datenschutzbestimmungen des jeweiligen Anbieters.

## 3. Hosting

**Server:** Die App läuft auf einem Server der Hetzner Online GmbH, Industriestr. 25, 91710 Gunzenhausen, Deutschland, im Rechenzentrum Nürnberg. Dort liegen auch alle hochgeladenen Dateien und Trainingsergebnisse. Mit Hetzner besteht ein Vertrag zur Auftragsverarbeitung nach Art. 28 DSGVO.

**Datenbank:** Konten, Netze, Kommentare und die Angaben zu Daten und Trainings (nicht die Dateien selbst) liegen in einer PostgreSQL-Datenbank des Anbieters Supabase Inc., in der Region London (Vereinigtes Königreich). Mit Supabase besteht ein Vertrag zur Auftragsverarbeitung (Data Processing Addendum) nach Art. 28 DSGVO. Für das Vereinigte Königreich hat die EU-Kommission festgestellt, dass dort ein angemessenes Datenschutzniveau besteht (Angemessenheitsbeschluss, Art. 45 DSGVO). Soweit Supabase oder seine Unterauftragnehmer Daten in andere Drittländer, etwa die USA, übermitteln, geschieht das auf Grundlage der EU-Standardvertragsklauseln (Art. 46 Abs. 2 lit. c DSGVO), die Teil dieses Vertrags sind.

Rechtsgrundlage: Art. 6 Abs. 1 lit. b DSGVO (Bereitstellung des Dienstes) und lit. f (berechtigtes Interesse an einem sicheren, zuverlässigen Betrieb).

## 4. Aufruf der Website und Server-Logdateien

Beim Aufruf der Seite verarbeitet der Server technisch notwendig Ihre IP-Adresse. In den Protokollen (Logdateien) des Webservers werden gespeichert: die **gekürzte** IP-Adresse (bei IPv4 ist der letzte Block 0, bei IPv6 bleiben nur die ersten beiden Blöcke), Datum und Uhrzeit, die aufgerufene Seite ohne Parameter, HTTP-Status, übertragene Datenmenge, Antwortzeit und die Browserkennung (User-Agent). Fehlerprotokolle können vollständige IP-Adressen enthalten.

Die Protokolle werden nach Größe automatisch überschrieben (höchstens etwa 30 MB je Dienst); bei wenig Verkehr können sie einige Wochen alt werden.

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

Nur der Verwaltungsbereich `/admin/`, den ausschließlich der Betreiber nutzt, verwendet ein Sitzungs- und ein CSRF-Cookie.

## 9. API-Token und verbundene Apps (KI-Assistenten)

Sie können API-Token anlegen oder Apps (z. B. KI-Assistenten über MCP) mit Ihrem Konto verbinden. Gespeichert werden Name, Anfang des Tokens, Zeitpunkte von Erstellung und letzter Nutzung sowie der Name der App – vom Token selbst nur ein Hash.

Verbinden Sie eine App, ruft diese auf Ihre Veranlassung Daten Ihres Kontos ab und verarbeitet sie bei ihrem Anbieter (z. B. Anthropic bei Claude). Dafür gelten die Datenschutzbestimmungen dieses Anbieters. Sie können die Verbindung jederzeit unter „API tokens“ trennen.

**Protokoll der Agentenaktionen:** Was ein Token oder eine verbundene App in Ihrem Konto tut (jeder Aufruf über MCP oder die API), wird protokolliert und Ihnen unter „Agent activity“ angezeigt: Name des Tokens bzw. der App, Anfang des Tokens, Kennung des Programms (User-Agent), Aktion, Ergebnis mit Fehlermeldung, Dauer, betroffene Netze, Trainings und Datensätze sowie die Argumente in zusammengefasster Form (Größen und Anzahlen statt Dateiinhalten und Eingabedaten, nie Passwörter oder Token). Nur Sie sehen Ihr Protokoll. Es wird nach 90 Tagen automatisch gelöscht (höchstens 20.000 Einträge je Konto), mit dem Konto gelöscht und ist im Datenexport enthalten. Zweck: Nachvollziehbarkeit und Kontrolle dessen, was Assistenten in Ihrem Namen tun. Rechtsgrundlage: Art. 6 Abs. 1 lit. b DSGVO.

## 10. Speicherdauer und Löschung

- **Konto und Inhalte:** bis Sie sie löschen. Unter „Account“ können Sie Ihr Konto jederzeit selbst löschen; dabei werden alle Ihre Netze, Kommentare, hochgeladenen Dateien, Aufbereitungen, Trainings, API-Token, verbundenen Apps und das Protokoll der Agentenaktionen gelöscht. Netze anderer, an denen Sie mitgearbeitet haben, bleiben bestehen (ohne Hinweis auf Sie als Bearbeiter).
- **Ungenutzte Uploads und Trainings** (hochgeladene Dateien, aufbereitete Datensätze und abgeschlossene Trainings) werden nach 180 Tagen ohne Nutzung automatisch gelöscht.
- **Inaktive Konten** werden nach zwei Jahren (730 Tagen) ohne Login und ohne Nutzung mit allen Inhalten automatisch gelöscht. Die App versendet keine E-Mails, eine Erinnerung vorher gibt es daher nicht.
- **Logins** (Erneuerungs-Token) verfallen nach 7 Tagen ohne Nutzung und werden danach gelöscht.
- **Protokoll der Agentenaktionen** (Abschnitt 9): nach 90 Tagen.
- **Server-Logs:** siehe Abschnitt 4.
- **Sicherungen:** Gelöschte Daten können noch in Sicherungskopien der Datenbank bzw. des Servers enthalten sein, bis diese turnusmäßig überschrieben werden (in der Regel nach wenigen Tagen).

## 11. Ihre Rechte

Sie haben das Recht auf Auskunft (Art. 15 DSGVO), Berichtigung (Art. 16), Löschung (Art. 17), Einschränkung der Verarbeitung (Art. 18), Datenübertragbarkeit (Art. 20) und Widerspruch gegen Verarbeitungen auf Grundlage berechtigter Interessen (Art. 21).

Vieles können Sie selbst unter **Account** erledigen: E-Mail-Adresse ändern, **alle Ihre Daten als ZIP-Datei herunterladen** (maschinenlesbar, JSON und Ihre Dateien) und das **Konto löschen**. Für alles andere schreiben Sie an alexander.kuehling@gmx.net.

Sie haben außerdem das Recht, sich bei einer Datenschutz-Aufsichtsbehörde zu beschweren (Art. 77 DSGVO), insbesondere in dem Mitgliedstaat Ihres Aufenthaltsorts, Ihres Arbeitsplatzes oder des Orts des mutmaßlichen Verstoßes.

## 12. Pflicht zur Bereitstellung, automatisierte Entscheidungen

Für ein Konto sind Benutzername, E-Mail-Adresse und Passwort nötig; ohne sie kann kein Konto angelegt werden. Netze bauen und Code erzeugen geht auch ohne Konto. Es gibt keine automatisierten Entscheidungen oder Profiling im Sinne von Art. 22 DSGVO.

## 13. Änderungen

Ich passe diese Erklärung an, wenn sich die App oder die Rechtslage ändert. Es gilt die hier veröffentlichte Fassung.
