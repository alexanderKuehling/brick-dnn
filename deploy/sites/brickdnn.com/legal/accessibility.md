# Erklärung zur Barrierefreiheit / Accessibility statement

Last updated / Stand: 18 November 2026

## English

I want everyone to be able to use Brick DNN, including people who use a keyboard instead of a mouse, a screen reader, screen magnification or other assistive technology. This statement describes how far the web app at brickdnn.com meets this aim.

**Is the law binding here?** Brick DNN is run by one person (Alexander Kühling, see the [imprint](/imprint)). Under the German Accessibility Strengthening Act (Barrierefreiheitsstärkungsgesetz, BFSG), micro-enterprises that provide services – fewer than 10 employees and an annual turnover or balance sheet total of at most 2 million euros – are exempt from its accessibility requirements. I assume that this exemption applies. I still follow the standard the BFSG refers to, voluntarily, and publish this statement as it would require. This is my own assessment, not legal advice.

**Standard.** The aim is the European standard EN 301 549, which refers to the Web Content Accessibility Guidelines (WCAG) 2.1 at level AA; where possible I go by WCAG 2.2 AA.

### Status: partially accessible

The web app is **partially accessible**. Most pages meet WCAG 2.2 AA as far as automatic and manual tests show; the parts below do not yet, or only with limits.

### How the requirements are met

- **Keyboard:** every page can be used with the keyboard. The first Tab stop is "Skip to content". The layer palette is one Tab stop; the arrow keys move within it, Enter or Space adds a layer, and Enter in the search field adds the first match.
- **Building a network without a mouse:** layers on the canvas are reached with Tab and selected with Enter or Space; the arrow keys then move the selected layer and Delete removes it. Connections are made and removed in the properties panel under "Connections" ("Add input from …", "Add output to …"). Without a layer selected, the panel lists all layers with their inputs, as a text version of the canvas.
- **Screen readers:** pages have landmarks (header, navigation, main region), headings, a title of their own, and labelled form fields whose hints and errors are linked to them. Icon buttons have names in German and English. What an action on the canvas changed (added, connected, deleted) and notifications are announced without interrupting.
- **Charts:** training curves have a text summary, can be read value by value with the arrow keys, and the values per epoch are available as a table. The charts of the Data tab have a table view.
- **Display:** text and controls have the contrast WCAG requires in the light and in the dark theme; the keyboard focus is clearly visible. Pages can be zoomed to 200 % and used at a width of 320 pixels. With "reduce motion" switched on in the system, animations are left out.
- **Languages:** the app is available in German and English; the page language is set accordingly.
- **Identification, sign-in and payment:** sign-up and login are ordinary forms that work with password managers; two-factor authentication accepts codes from any authenticator app. Payment takes place at Paddle, whose checkout is outside my control.

### Known limitations

- **The canvas is a visual editor.** The position of a layer on the canvas carries no meaning, but the drawing itself (connection lines, the minimap) is not available to screen readers beyond the names of layers and connections. Use the list of layers and the "Connections" section of the properties panel instead. Moving several layers at once, and box selection, need a mouse or touch.
- **Dragging:** layers can be dragged from the palette and connections drawn by dragging between the dots; for both there is a way without dragging (palette: click or Enter; connections: properties panel).
- **Narrow screens:** on phones and at very high zoom the editor shows the palette and the properties as panels over the canvas; the canvas itself does not reflow (WCAG allows this for content that needs a two-dimensional layout).
- **Charts of the Data tab** (histograms, scatter plots, heat maps) are pictures with a title and a table view; some of their colours have less than 3:1 contrast against the background. Their legends and tables carry the same information.
- **Generated code and logs** are shown as preformatted text; long lines scroll sideways.
- **Payment:** the checkout of Paddle and the pages of other providers linked from the app (GitHub, Buy me a coffee) are not covered by this statement.
- **The landing pages** (brickdnn.com/ and /de/) were checked automatically only.

### How this was checked

Self-assessment, last on 15 November 2026: automated tests with axe-core on the main pages and components in both languages and both themes, keyboard-only walkthroughs, checks of the screen reader semantics (roles, names, landmarks, live regions), contrast, zoom to 200 % and reflow at 320 pixels. No external audit has been done.

### Feedback and contact

If something does not work for you, or you need content in another form, please write to **alexander.kuehling@brickdnn.com**. Describe what you tried and on which page; I answer as soon as I can, at the latest within two weeks.

### Enforcement

If the BFSG applied, the market surveillance authority of the German federal states for accessibility, the Marktüberwachungsstelle der Länder für die Barrierefreiheit von Produkten und Dienstleistungen (MLBF), would be responsible: Carl-Miller-Straße 6, 39112 Magdeburg, Germany, [www.mlbf-barrierefrei.de](https://www.mlbf-barrierefrei.de). Please contact me first, so that I can fix the problem.

## Deutsch

Brick DNN soll für alle nutzbar sein, auch für Menschen, die eine Tastatur statt einer Maus, einen Screenreader, eine Bildschirmlupe oder andere assistive Technik verwenden. Diese Erklärung beschreibt, wie weit die Web-App unter brickdnn.com dieses Ziel erreicht.

**Gilt das Gesetz hier?** Brick DNN wird von einer Einzelperson betrieben (Alexander Kühling, siehe [Impressum](/imprint)). Nach dem Barrierefreiheitsstärkungsgesetz (BFSG) sind Kleinstunternehmen, die Dienstleistungen erbringen – weniger als 10 Beschäftigte und höchstens 2 Millionen Euro Jahresumsatz oder Jahresbilanzsumme –, von seinen Barrierefreiheitsanforderungen ausgenommen. Ich gehe davon aus, dass diese Ausnahme greift. Ich richte mich trotzdem freiwillig nach dem Standard, auf den das BFSG verweist, und veröffentliche diese Erklärung so, wie es sie verlangen würde. Das ist meine eigene Einschätzung, keine Rechtsberatung.

**Standard.** Maßstab ist die europäische Norm EN 301 549, die auf die Web Content Accessibility Guidelines (WCAG) 2.1 in Stufe AA verweist; wo möglich richte ich mich nach WCAG 2.2 AA.

### Stand der Vereinbarkeit: teilweise barrierefrei

Die Web-App ist **teilweise barrierefrei**. Die meisten Seiten erfüllen WCAG 2.2 AA, soweit automatische und manuelle Prüfungen das zeigen; die unten genannten Teile noch nicht oder nur eingeschränkt.

### Wie die Anforderungen erfüllt werden

- **Tastatur:** Alle Seiten lassen sich mit der Tastatur bedienen. Der erste Tab-Halt ist „Zum Inhalt springen“. Die Schichten-Palette ist ein Tab-Halt; die Pfeiltasten bewegen sich darin, Eingabe oder Leertaste fügt eine Schicht hinzu, die Eingabetaste im Suchfeld den ersten Treffer.
- **Netze ohne Maus bauen:** Schichten auf der Arbeitsfläche erreichen Sie mit Tab und wählen sie mit Eingabe oder Leertaste aus; danach verschieben die Pfeiltasten die Schicht, Entf löscht sie. Verbindungen legen Sie im Bereich „Eigenschaften“ unter „Verbindungen“ an und entfernen sie dort („Eingang hinzufügen von …“, „Ausgang hinzufügen zu …“). Ist keine Schicht ausgewählt, listet der Bereich alle Schichten mit ihren Eingängen auf – eine Textfassung der Arbeitsfläche.
- **Screenreader:** Die Seiten haben Landmarks (Kopfbereich, Navigation, Hauptbereich), Überschriften, einen eigenen Titel und beschriftete Formularfelder, mit denen Hinweise und Fehlermeldungen verknüpft sind. Symbol-Schaltflächen haben Namen auf Deutsch und Englisch. Was eine Aktion auf der Arbeitsfläche geändert hat (hinzugefügt, verbunden, gelöscht), und Benachrichtigungen werden angesagt, ohne zu unterbrechen.
- **Diagramme:** Trainingskurven haben eine Textzusammenfassung, lassen sich mit den Pfeiltasten Wert für Wert lesen, und die Werte je Epoche gibt es als Tabelle. Die Diagramme im Bereich „Daten“ haben eine Tabellenansicht.
- **Darstellung:** Texte und Bedienelemente haben im hellen und im dunklen Design den von WCAG verlangten Kontrast; der Tastaturfokus ist deutlich sichtbar. Die Seiten lassen sich auf 200 % vergrößern und bei 320 Pixel Breite nutzen. Ist im System „Bewegung reduzieren“ eingeschaltet, entfallen Animationen.
- **Sprachen:** Die App gibt es auf Deutsch und Englisch; die Sprache der Seite ist entsprechend ausgezeichnet.
- **Identifizierung, Anmeldung und Zahlung:** Registrierung und Anmeldung sind gewöhnliche Formulare, die mit Passwort-Managern funktionieren; die Zwei-Faktor-Anmeldung nimmt Codes jeder Authenticator-App an. Bezahlt wird bei Paddle; dessen Bezahlvorgang liegt außerhalb meines Einflusses.

### Bekannte Einschränkungen

- **Die Arbeitsfläche ist ein grafischer Editor.** Die Position einer Schicht hat keine Bedeutung, aber die Zeichnung selbst (Verbindungslinien, Übersichtskarte) ist für Screenreader nur über die Namen der Schichten und Verbindungen zugänglich. Nutzen Sie stattdessen die Liste der Schichten und den Abschnitt „Verbindungen“ im Bereich „Eigenschaften“. Mehrere Schichten auf einmal verschieben und die Auswahl per Rahmen gehen nur mit Maus oder Touch.
- **Ziehen:** Schichten lassen sich aus der Palette ziehen und Verbindungen durch Ziehen zwischen den Punkten anlegen; für beides gibt es einen Weg ohne Ziehen (Palette: Klick oder Eingabetaste; Verbindungen: Bereich „Eigenschaften“).
- **Schmale Bildschirme:** Auf Smartphones und bei sehr starker Vergrößerung zeigt der Editor Palette und Eigenschaften als Bereiche über der Arbeitsfläche; die Arbeitsfläche selbst wird nicht umbrochen (WCAG erlaubt das für Inhalte, die eine zweidimensionale Anordnung brauchen).
- **Diagramme im Bereich „Daten“** (Histogramme, Streudiagramme, Heatmaps) sind Grafiken mit Titel und Tabellenansicht; einige ihrer Farben haben weniger als 3:1 Kontrast zum Hintergrund. Legenden und Tabellen enthalten dieselben Informationen.
- **Erzeugter Code und Protokolle** werden als vorformatierter Text gezeigt; lange Zeilen scrollen seitwärts.
- **Zahlung:** Der Bezahlvorgang von Paddle und die Seiten anderer Anbieter, auf die die App verlinkt (GitHub, Buy me a coffee), sind nicht Teil dieser Erklärung.
- **Die Startseiten** (brickdnn.com/ und /de/) wurden nur automatisch geprüft.

### Wie geprüft wurde

Selbstbewertung, zuletzt am 15. November 2026: automatische Tests mit axe-core auf den wichtigsten Seiten und Komponenten in beiden Sprachen und beiden Designs, Durchgänge nur mit der Tastatur, Prüfung der Screenreader-Semantik (Rollen, Namen, Landmarks, Live-Regionen), Kontrast, Vergrößerung auf 200 % und Umbruch bei 320 Pixel Breite. Eine externe Prüfung gab es nicht.

### Feedback und Kontakt

Wenn etwas für Sie nicht funktioniert oder Sie Inhalte in einer anderen Form brauchen, schreiben Sie bitte an **alexander.kuehling@brickdnn.com**. Beschreiben Sie, was Sie versucht haben und auf welcher Seite; ich antworte so bald wie möglich, spätestens innerhalb von zwei Wochen.

### Durchsetzung

Fiele Brick DNN unter das BFSG, wäre die Marktüberwachungsstelle der Länder für die Barrierefreiheit von Produkten und Dienstleistungen (MLBF) zuständig: Carl-Miller-Straße 6, 39112 Magdeburg, [www.mlbf-barrierefrei.de](https://www.mlbf-barrierefrei.de). Bitte wenden Sie sich zuerst an mich, damit ich das Problem beheben kann.
