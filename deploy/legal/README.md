# Rechtstexte: Vorlagen und Checkliste

Brick DNN zeigt ein Impressum (`/imprint`), eine Datenschutzerklärung (`/privacy`) und
Nutzungsbedingungen (`/terms`) an, sobald die passenden Markdown-Dateien auf dem Server liegen.
Die Dateien hier sind **Vorlagen**, keine Rechtsberatung: Sie beschreiben, was die Software
technisch tut, ersetzen aber keinen Generator und keine anwaltliche Prüfung. Die technischen
Hintergründe (auf Englisch) stehen in [docs/privacy.md](../../docs/privacy.md).

| Vorlage                | Datei auf dem Server | Seite      |
| ---------------------- | -------------------- | ---------- |
| `imprint.example.md`   | `imprint.md`         | `/imprint` |
| `privacy.example.md`   | `privacy.md`         | `/privacy` |
| `terms.example.md`     | `terms.md`           | `/terms`   |

## So kommen die Texte auf den Server

Der Ordner ist `BRICKDNN_LEGAL_DIR`:

- **Server mit `deploy/server/`:** am einfachsten im Repository unter
  `deploy/sites/<Ihre Domain>/legal/` (die Domain wie `BRICKDNN_DOMAIN` in der `.env` des
  Servers, siehe [`deploy/sites/`](../sites/README.md)); sie gehen mit dem nächsten Deploy
  online. Oder direkt auf den Server nach `/srv/brick-dnn/legal` (legt `setup.sh` an; diese
  Dateien haben Vorrang):

  ```bash
  scp imprint.md privacy.md terms.md root@SERVER:/srv/brick-dnn/legal/
  ```

- **Docker Compose lokal:** `/data/legal` im Daten-Volume:

  ```bash
  docker compose exec backend mkdir -p /data/legal
  docker compose cp privacy.md backend:/data/legal/privacy.md
  ```

- **Ohne Docker:** `backend/data/legal/` (oder ein eigener Pfad in `BRICKDNN_LEGAL_DIR`).

Änderungen gelten sofort, ohne Neustart. Fehlt eine Datei (oder ist sie leer), wird die Seite
nicht verlinkt. Liegt `terms.md` vor, müssen neue Konten die Bedingungen bei der Registrierung
akzeptieren; gespeichert werden Zeitpunkt und Fassung (Hash) von `terms.md` und `privacy.md`.
Bestehende Konten werden bei einer neuen Fassung nicht erneut gefragt – informieren Sie sie
bei wesentlichen Änderungen selbst.

## Markdown, das angezeigt wird

Überschriften (`#`, `##`, …), Absätze, Aufzählungen (`-`, `1.`, auch verschachtelt), Zitate
(`>`), Trennlinien (`---`), **fett**, *kursiv*, Links (`[Text](https://…)`, `[Text](/privacy)`,
`mailto:`, `tel:`) und einfache Web- und E-Mail-Adressen. Ein einfacher Zeilenumbruch bleibt
ein Zeilenumbruch (praktisch für Anschriften): Schreiben Sie jeden Absatz in eine Zeile, wie in
den Vorlagen. **HTML wird nicht ausgeführt**, sondern als
Text angezeigt; Tabellen und Bilder gibt es nicht. Texte aus Generatoren, die HTML liefern,
also vorher in Markdown umwandeln.

## Checkliste: Was die Datenschutzerklärung für Brick DNN abdecken muss

Aus dem Quellcode abgeleitet (Stand dieser Version). Was auf Ihren Server nicht zutrifft,
streichen.

- [ ] **Verantwortlicher** mit Anschrift und E-Mail; Hinweis, dass kein
      Datenschutzbeauftragter nötig ist.
- [ ] **Hosting bei Hetzner** (Server und alle Dateien) – AV-Vertrag in der Hetzner-Konsole
      abschließen; Rechenzentrumsstandort angeben.
- [ ] **Datenbank bei Supabase**, falls `DATABASE_URL` auf Supabase zeigt: Anbieter, Region,
      DPA abschließen, Drittlandübermittlung (USA?) und ihre Grundlage prüfen. Sonst: Datenbank
      liegt auf dem Server.
- [ ] **Server-Logs:** nginx protokolliert gekürzte IP-Adressen, Zeit, Pfad ohne Parameter,
      Status, Größe, Antwortzeit, User-Agent; Caddy schreibt kein Zugriffsprotokoll, aber
      Fehlermeldungen mit vollständigen IPs. Docker behält je Dienst höchstens 3 × 10 MB
      (Speicherdauer also je nach Verkehr Tage bis Wochen). Zweck Sicherheit/Betrieb, Art. 6
      Abs. 1 lit. f.
- [ ] **Ratenbegrenzung:** IP-Adresse bzw. Konto-ID höchstens eine Stunde im Cache.
- [ ] **Konto:** Benutzername, E-Mail (für Einladungen, nicht öffentlich), Passwort-Hash,
      Registrierung, letzter Login, letzte Nutzung, Annahme der Nutzungsbedingungen mit
      Fassung. Art. 6 Abs. 1 lit. b.
- [ ] **Inhalte:** Netze mit Versionsverlauf, Kommentare, Freigaben (eingeladene Personen,
      Freigabe-Links), hochgeladene Dateien (CSV, Excel, JSON, Parquet, ZIP …),
      Aufbereitungen, Trainings mit Kennzahlen, Protokollen und Gewichten – alles auf dem
      Server; Bitte, keine personenbezogenen Daten hochzuladen.
- [ ] **Import über Links:** der Server ruft die Datei beim Anbieter des Links ab (dieser
      sieht die Server-IP); gespeichert wird der Link ohne Query-Teil.
      `BRICKDNN_DATA_URL_IMPORT=false` schaltet das ab.
- [ ] **Echtzeit-Bearbeitung** (WebSockets, Redis nur im Arbeitsspeicher): andere Mitglieder
      sehen Benutzername und Auswahl; nichts davon wird gespeichert.
- [ ] **Keine Cookies, kein Tracking:** localStorage für Login-Token (Login höchstens 7 Tage
      ohne Nutzung), Netz-Entwurf, Einstellungen, Farbschema – § 25 Abs. 2 Nr. 2 TDDDG.
      `/admin/` (nur Betreiber) nutzt Sitzungs- und CSRF-Cookie.
- [ ] **Keine Inhalte Dritter:** Die App lädt nichts von fremden Servern; GitHub- und
      „Buy me a coffee“-Links sind nur Links.
- [ ] **API-Token und verbundene Apps (MCP, z. B. Claude):** gespeichert werden nur Name,
      Präfix, Hash, Zeitpunkte; was eine verbundene App abruft, verarbeitet deren Anbieter.
- [ ] **Protokoll der Agentenaktionen:** jeder Aufruf mit Token oder App (Aktion, Ergebnis,
      Dauer, User-Agent, betroffene Objekte, zusammengefasste Argumente ohne Inhalte und
      Geheimnisse), nur für den Nutzer sichtbar; Speicherdauer `BRICKDNN_AGENT_LOG_DAYS`
      (Standard 90 Tage), im Export enthalten, mit dem Konto gelöscht.
- [ ] **Keine E-Mails:** Die App versendet keine.
- [ ] **Speicherdauer:** bis zur Löschung durch die Nutzer; automatische Löschung nach
      `BRICKDNN_DELETE_UNUSED_UPLOADS_DAYS` / `BRICKDNN_DELETE_INACTIVE_ACCOUNTS_DAYS`, falls
      eingestellt; abgelaufene Logins täglich; Backups (Hetzner-Backups: 7 Stände) angeben.
- [ ] **Betroffenenrechte** (Art. 15–18, 20, 21, 77) mit Hinweis auf die Selbstbedienung unter
      „Account“: Datenexport (ZIP mit JSON und Dateien), E-Mail ändern, Konto löschen;
      zuständige Aufsichtsbehörde Ihres Bundeslandes.
- [ ] **Keine automatisierten Entscheidungen**, Pflicht zur Bereitstellung (nur für ein Konto).
- [ ] **Stand** (Datum).

## Was außerdem zu tun bleibt (nicht technisch)

- Texte erstellen bzw. prüfen lassen und als `imprint.md`, `privacy.md`, `terms.md` ablegen.
- AV-Verträge mit Hetzner und ggf. Supabase abschließen und ablegen.
- Ein kurzes Verzeichnis der Verarbeitungstätigkeiten (Art. 30 DSGVO) und der technischen
  und organisatorischen Maßnahmen führen (Vorlage: die Checkliste oben und docs/privacy.md).
- Ablauf für Datenpannen festlegen: Meldung an die Aufsichtsbehörde binnen 72 Stunden
  (Art. 33), ggf. Benachrichtigung der Betroffenen (Art. 34); wer prüft was, wo stehen die
  Logs.
- Anfragen von Betroffenen beantworten (innerhalb eines Monats); Exporte und Löschungen
  können Nutzer selbst auslösen.
- Entscheiden, ob und nach wie vielen Tagen ungenutzte Uploads und inaktive Konten gelöscht
  werden, und die Zeiträume in Datenschutzerklärung und Nutzungsbedingungen nennen.
- Die Online-Demo (Hugging Face) läuft ohne Konto im Browser und lädt Pyodide von
  cdn.jsdelivr.net; für sie gelten die Bedingungen von Hugging Face. Bei Bedarf dort ein
  Impressum ergänzen (README des Space).
