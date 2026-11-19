# The "Open in Brick DNN" badge, link previews and citing

How course material, READMEs and papers point at Brick DNN: a badge that opens a network in
the editor, what a shared link looks like when it is pasted into a chat, and how to cite the
software.

## The badge

[![Open in Brick DNN](https://brickdnn.com/badge/open-in-brick-dnn.svg)](https://brickdnn.com/app?example=unet)

An SVG image the site serves at a fixed address, in English and in German:

| File | Text |
|---|---|
| `/badge/open-in-brick-dnn.svg` | Open in Brick DNN |
| `/badge/open-in-brick-dnn.de.svg` | In Brick DNN öffnen |

Markdown (a README, a Jupyter notebook):

```markdown
[![Open in Brick DNN](https://brickdnn.com/badge/open-in-brick-dnn.svg)](https://brickdnn.com/app?example=unet)
```

HTML (a course page):

```html
<a href="https://brickdnn.com/app?example=unet"><img src="https://brickdnn.com/badge/open-in-brick-dnn.svg" alt="Open in Brick DNN" height="28"></a>
```

Where the badge can lead:

- **A built-in example:** `https://brickdnn.com/app?example=<id>`. The ids are those of
  `brickdnn examples` and of the example pages (`cnn_mnist`, `unet`, `transformer_classifier`
  …, see <https://brickdnn.com/examples/>). The editor opens with the example loaded; nobody
  needs an account.
- **A network of your own:** turn on its link under **Share** in the editor
  (`https://brickdnn.com/share/<token>`). The dialog shows the badge for that link with
  buttons that copy the Markdown or the HTML. Whoever follows it sees the network read-only,
  can read its code and can open a copy in the editor without an account.

On your own server the files are at the same paths, and the share dialog writes the snippet
with your server's address. The files are `frontend/public/badge/*.svg`; their names must not
change, because pages elsewhere point at them.

## Link previews

A link to a shared network (`/share/<token>`) or to a shared comparison of runs
(`/compare/<token>`) unfolds in chats and social networks with a title, a description and an
image. The app is a single-page application, which the programs that make these previews do
not run, so the tags have to be in the HTML:

- `frontend/index.html` has general tags (Open Graph and Twitter card) for every page of the
  app, inside a server-side include block.
- For `/share/…` and `/compare/…`, nginx replaces that block with the answer of
  `GET /api/link-preview/<share|compare>/<token>/` (`backend/brick_dnn_core/link_previews.py`):
  the **name** of the network or comparison and what kind of thing it is. Nothing else: not
  the owner, not the description, not the layers or runs. The language is the one saved on
  the owner's account, else English.
- A link that does not exist or was turned off, an error of the backend or an answer that
  takes longer than a few seconds leave the general tags in place. The page itself is never
  held back by more than that.
- Shared links carry `X-Robots-Tag: noindex` and `<meta name="robots" content="noindex">`.
  `robots.txt` does not block them: a blocked page cannot say that it wants to stay out of
  an index, and several preview programs obey `robots.txt`.

The preview image is `frontend/public/brand/link-preview.png` (1200 × 630). It, the image of
the landing pages (`frontend/landing/images/og-image.jpg`), the PNG logo and the repository's
social preview (`docs/images/social-preview.png`) are drawn by
`frontend/scripts/brand-images.mjs` from the logo (`frontend/public/favicon.svg`).

In development (`npm run dev`) there is no nginx, so shared links show the general tags.
`docker compose up` shows the real thing: `curl -s localhost:8080/share/<token> | grep og:`.

## Citing Brick DNN

The repository has a [`CITATION.cff`](../CITATION.cff); GitHub shows it as "Cite this
repository". In short:

> Kühling, A. (2026). Brick DNN (Version 1.1.0) [Computer software]. https://github.com/alexanderKuehling/brick-dnn

Exported scripts name Brick DNN and its address in their first line, so code that ends up in
a course repository or a paper's supplement says where it came from.
