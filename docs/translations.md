# Languages: English and German

Brick DNN can be used in English and German. This page says how the language is chosen and
stored, what is translated, how it works in the frontend and the backend, and how to add or
change a translation.

## How the language is chosen

1. **First visit:** the browser's preferred language (`navigator.languages`, the first entry).
   German for `de`, `de-DE`, `de-AT` …, English for everything else.
2. **The switch** `EN | DE` in the header (on phones in the page menu) and on the login and
   sign-up pages changes the language at once, without reloading.
3. **Saved:** the choice is kept in the browser (`localStorage`, key `brickdnn:language`) and,
   while logged in, on the account (`UserProfile.language`). After logging in on another
   device the app takes the account's language; an account without one (older accounts)
   gets the language the app has at that moment. Signing up stores the app's language on the
   new account.
4. `<html lang>` follows the language (screen readers, hyphenation, the browser's own number
   fields).

The language of the account also decides the language of the **e-mails** the server sends
(confirming the address, password reset, security notices, the warning before an inactive
account is deleted): the account's language, else the language of the request that triggered
the e-mail, else English.

## What is translated and what is not

Translated: the texts of the app (pages, menus, forms, buttons, messages, notifications),
the descriptions of layers, parameters, preparation steps, examples and training options
from the server, validation and error messages of the account and sign-up (with Django's and
Django REST framework's own German messages), the e-mails, and on bilingual legal pages the
section of the chosen language.

Not translated, on purpose:

- generated PyTorch and scikit-learn code and the names in it;
- layer type names (`Linear`, `Conv2d`, `HistGradientBoostingClassifier` …), parameter names
  (`in_features`) and other identifiers;
- the MCP server and its tool descriptions: AI assistants read them, and they stay English
  whatever language the client asks for;
- the REST API without an `Accept-Language` header (scripts, curl): English;
- user content (network names, column names, comments) and the legal texts themselves.

German uses the Sie-form and the terms of German ML courses: *Verlustfunktion*, *Lernrate*,
*Batchgröße*, *Epochen*, *Gewichte*, *Merkmale*, *Zielspalte*, *Stichprobe*; established
English terms stay where German courses use them too (*Dropout*, *Early Stopping*,
*Weight Decay*, *One-hot*, *Embedding*).

## Numbers and dates

Numbers and dates are written with `Intl` in the language of the app: `1,234.5` and
*Nov 15, 2026* in English, `1.234,5` and *15.11.2026* in German (`frontend/src/i18n/format.ts`,
`currentLocale()`). Number fields (`NumberInput`) show a decimal comma in German; they
return numbers, so values sent to the API, saved in networks or written into generated code
always use a point. Parameter values on the canvas (`p = 0.2`) are written as the code has
them.

## Frontend

[i18next](https://www.i18next.com) with [react-i18next](https://react.i18next.com):

- `frontend/src/i18n/index.ts`: set-up, detection, `setLanguage`, `useLanguage`,
  `currentLocale`;
- `frontend/src/i18n/locales/en/*.ts`: the English texts, one file per area of the app
  (`builder.ts`, `training.ts` …). English is the reference;
- `frontend/src/i18n/locales/de/*.ts`: the German texts with the same keys. Each file is
  typed `Messages<typeof en…>`, so a missing or extra key is a compiler error;
- `frontend/src/i18n/i18next.d.ts`: typed keys: `t('builder.toolbar.train')` must name an
  existing text;
- `frontend/src/i18n/mantine.tsx`: Mantine in the language (decimal comma of number fields,
  buttons of confirmation dialogs). The app uses no `@mantine/dates`;
- `frontend/src/i18n/LanguageSwitch.tsx`: the switch.

In components:

```tsx
const { t } = useTranslation();
<Button>{t('builder.toolbar.train')}</Button>
t('builder.toolbar.problems', { count })        // plurals: problems_one / problems_other
t('training.setup.deleteTitle', { name })       // interpolation: 'Delete “{{name}}”?'
<Trans i18nKey="builder.inspector.tipMerge" components={{ b: <b /> }} />  // markup
```

Outside components (formatting functions, notifications in callbacks) use `i18n.t(…)`.
Tags inside texts must not be named like HTML void elements (`<link>`, `<br>`, `<img>`):
react-i18next would treat them as empty.

Every request to the API carries `Accept-Language` with the app's language (not the
browser's), and the layer, step and training catalogues are loaded again when the language
changes.

Tests run in English (jsdom's browser language) and every test starts in English again
(`vitest.setup.mjs`). `frontend/src/i18n/i18n.test.tsx` fails when a key is missing in one of
the languages, a text is empty, or placeholders and tags differ between the languages.

## Backend

Django's gettext with `LOCALE_PATHS = backend/locale`:

- `brick_dnn_core/languages.py`: `ApiLocaleMiddleware` activates the language of
  `Accept-Language` (only `en` and `de`; without the header English; the MCP server always
  English) and answers with `Content-Language` and `Vary: Accept-Language`;
  `user_language()` picks the language of an e-mail;
- `brick_dnn_core/catalog.py`: the `brickdnn` package describes layers, steps, examples and
  options in English and knows no Django. The API translates their text fields (`label`,
  `help`, `description` …) with gettext when it returns them; identifiers stay;
- `brick_dnn_core/notices.py`: every e-mail is written inside
  `translation.override(<recipient's language>)`;
- `backend/locale/de/LC_MESSAGES/django.po`: the German texts; `django.mo` next to it is
  compiled from it and committed, so neither the Docker image nor CI needs GNU gettext.

API: `GET /api/auth/me/` returns `language` (`""` while none was chosen),
`PATCH /api/auth/account/language/` with `{"language": "de"}` saves it (website login), and
`POST /api/auth/register/` takes an optional `language`.

`tests/test_languages.py` fails when a string has no German translation or the `.po`/`.mo`
files are out of date.

## Changing or adding a text

**Frontend:** add the key to the English file of the area (`locales/en/<area>.ts`) and the
same key to the German file (`locales/de/<area>.ts`), then use it with `t('<area>.<key>')`.
`npx tsc -b --noEmit` and `npx vitest run src/i18n` say what is missing.

**Backend:** write new user-visible strings with `gettext` (or `gettext_lazy` for module
constants), with named placeholders (`gettext("Saved “%(name)s”.") % {"name": name}`), not
f-strings. Then

```bash
cd backend
python manage.py update_translations          # adds new strings to django.po, compiles django.mo
# translate the new entries (msgstr "") in locale/de/LC_MESSAGES/django.po
python manage.py update_translations          # compiles again
python manage.py update_translations --check  # what CI checks (pytest runs it too)
```

New descriptions in the layer or step catalogue are picked up the same way: the command reads
the catalogues and adds their texts.

## Adding a language

1. Frontend: add the code to `LANGUAGES` and `LANGUAGE_NAMES` in `frontend/src/i18n/index.ts`,
   copy `locales/de` to `locales/<code>`, translate it and register it in `resources`; extend
   `detectLanguage` and `currentLocale`.
2. Backend: add the code to `LANGUAGES` in `brick_dnn_site/settings.py`, to `LANGUAGES` and
   `LANGUAGE_CHOICES` in `brick_dnn_core/languages.py` and `LANGUAGES` in
   `brick_dnn_core/translations.py` (and the choices of `UserProfile.language`, with a
   migration), run `manage.py update_translations` and translate `locale/<code>/…/django.po`.
3. Legal pages: put `<page>.<code>.md` next to `<page>.md` (see
   [privacy.md](privacy.md#legal-pages-in-several-languages)).

## Known gaps

Some texts still appear in English in the German app; they are listed in the
[CHANGELOG](../CHANGELOG.md) entry of this feature and are translated step by step with the
same mechanism.
