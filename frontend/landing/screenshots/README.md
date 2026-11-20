# Screenshots of the landing pages

The images of the landing pages and the guides (`frontend/landing/images/`) are real
screenshots of the app, taken with Playwright from a Brick DNN running locally with real data
and results, once in English (`<id>-<width>.webp`) and once in German
(`<id>.de-<width>.webp`, see [In German](#in-german)):

- `setup.mjs` creates the data through the REST API and the MCP server: two accounts (`ada`
  and `grace`), a messy table of field trials (generated with a fixed seed) with a data
  preparation and the training dataset made from it, trainings of a CNN on MNIST, an MLP on
  the table, a hyperparameter search of gradient boosting and a U-Net on the built-in *Shapes*
  data, a network shared between the two accounts with a comment, and an API token with a few
  MCP calls for the agent activity page. Everything is looked up by name first, so a second run
  only adds what is missing.
- `shots.mjs` says what each screenshot shows: the page, the state of the app and the part
  to keep (in CSS pixels), and the width the landing page shows it at.
- `shoot.mjs` runs the setup, takes the screenshots at a device scale factor of 2 and writes
  each as WebP in a phone width, the width on the page and twice that
  (`<id>-<width>.webp`), plus the Open Graph image (`og-image.jpg`). Chromium encodes the
  images itself, so no image library is needed.

The landing pages pick the files up at the next build (`npm run build`); `landing.test.ts`
checks that every screenshot a page names exists in at least two widths of the same
proportions and that no unused image is left in `images/`.

## Taking them again

You need Python with the backend's requirements and PyTorch (with torchvision: MNIST is
downloaded on first use, 12 MB), and Node.js. Use an empty data directory: the screenshots
show exactly what the setup creates.

```bash
# 1. The API with SQLite, and in a second terminal the training worker (same variables)
cd backend
export BRICKDNN_DATA_DIR="$(mktemp -d)" DJANGO_DEBUG=true BRICKDNN_TRAINING_WORKER=true
python manage.py migrate
python manage.py runserver 127.0.0.1:8000
python manage.py training_worker

# 2. The built app, served like nginx serves it (the preview passes /api on to port 8000)
cd frontend
npm run build
npx vite preview --port 4173

# 3. Playwright is not a dependency of the app: install it for this only
npm install --no-save playwright && npx playwright install chromium
node landing/screenshots/shoot.mjs
node landing/screenshots/shoot.mjs --lang de
```

The first run takes about 15 minutes, most of it for the trainings (CPU). Options:

- `--lang de`: the German app, for the German pages (see below);
- `--only editor,data-prep`: only these screenshots (the ids in `shots.mjs`);
- `--raw <dir>`: also keep the full-resolution PNG files there;
- `--out <dir>`: write somewhere else than `images/` (e.g. to try a crop);
- `--app <url>`: another address of the app (default `http://127.0.0.1:4173`);
- `--no-og`: leave the Open Graph image as it is.

Look at the results before committing them: crops are fixed rectangles, so a change of the
app's layout can move things out of them.

## In German

`--lang de` takes the same screenshots of the German app: it sets the app's language
(`brickdnn:language` in localStorage, and the language saved on the two accounts, which wins
once somebody is logged in) and writes `<id>.de-<width>.webp`. The German pages use those
files instead of the English ones (`shotFiles` in `images.ts`); a screenshot without German
files falls back to the English ones. Run it after the English screenshots or on its own: the
data is the same, and the Open Graph image is only made in English.

The shots find buttons and headings by their visible texts, in `TEXTS.de` in `shots.mjs` (a
text missing there is looked for in English): when the app's German wording changes, change it
there too, and adjust the alt texts in `content/de.ts` and `content/guides.de.ts` if the
pictures say something else. What the server words in English shows in English: the names of
the data and networks the setup creates, the comment, an assistant's calls on the activity
page and the results of the preparation's steps.
