# Data preparation

Real data is rarely ready for a neural network: values are missing, a few outliers dominate a
column, categories are text, dates are strings, and the interesting table is spread over two
files. The **Data** tab turns such files (CSV, Excel, JSON, Parquet …, also straight from a
link) into a clean training dataset, shows what the data looks like on the way, and keeps every
decision visible and reproducible.

Like training, data preparation runs on the machine that runs the backend and needs an account,
because imported files are stored per user. It needs pandas in the backend environment (part
of `requirements.txt`); PyTorch is not needed to prepare data, only to train on it.

## Workflow

1. Open **Data** in the header and **import data**: a file from your computer or a link (see
   [Importing](#importing) for the formats). For CSV files Brick DNN detects the separator
   (`,` `;` tab `|`), decimal commas (`1,5`), the header line and the text encoding; for every
   format it detects common placeholders for missing values (`NA`, `N/A`, `null`, `-`, `?` …)
   and gives every column a type: *numeric*, *boolean*, *date/time*, *categorical* or *text*.
   If the detection is wrong, set the options under **Import options** or fix the type later
   with a *Change type* step.
2. A **preparation** is created for the file. Choose the **target** (the column to predict)
   and the task (*automatic*, classification or regression) in the *Target & split* card.
3. Add **steps** (below). After every change the preview runs again: each step reports what
   it did ("Filled 132 missing values in 2 columns with the median."), the **Data** view shows
   a random sample of the rows, and **Analysis** shows charts and statistics. *Preview after
   this step* shows the data at any point of the pipeline.
4. The **Readiness** view lists what still prevents training (missing values, categorical or
   text features, a missing target …), each with a button that adds the fitting step, and
   the [sanity checks](#readiness-and-sanity-checks) that point out target leakage,
   duplicates, rare classes and more.
5. **Create training dataset** (or **Train with this data**, which also opens the training
   panel in the builder). The dataset appears under *Your data* in the training panel, marked
   *Prepared*.

Preparations are saved automatically. Several preparations can use the same file.

AI assistants can do all of this over MCP: import, preview, fix, export and train (see
[Preparing data](mcp.md#preparing-data)).

## Importing

| Format | Files | Notes |
| ------ | ----- | ----- |
| CSV and other text | `.csv` `.tsv` `.tab` `.psv` `.txt` | separator, decimal mark, header and encoding are detected |
| Excel | `.xlsx` `.xlsm`, `.xls` (Excel 97–2003) | choose the **sheet**; the header row is detected; formulas give the value Excel saved; error values (`#N/A`, `#DIV/0!` …) are missing values |
| OpenDocument | `.ods` (LibreOffice, OpenOffice) | like Excel |
| JSON | `.json` | a list of objects (one per row), an object of columns (`{"a": [1, 2], "b": [3, 4]}`), such a list inside an object (e.g. `{"data": [...], "meta": ...}`, found up to three levels down) or pandas' `split` format; nested objects become columns like `address.city`, lists stay JSON text |
| JSON Lines | `.jsonl` `.ndjson` | one object per line |
| Parquet | `.parquet` | column types are kept; nested records become `a.b` columns |
| Feather / Arrow | `.feather` `.arrow` (Arrow IPC file or stream) | like Parquet |
| Packed | `.gz` `.bz2` `.xz`, `.zip` | a packed file of one of the formats above; a ZIP file must hold exactly one data file (`__MACOSX`, hidden files and a README are ignored) |

The content decides, not the name: a tab-separated text named `.xls` (a common "Excel export"
of web applications) is read as text, a `.txt` file with JSON as JSON. Every format ends in
the same typed table: cells are read as text the way a CSV file would show them and typed by
the same rules (header detection, missing-value markers, leading zeros kept as text, warnings
for almost-numeric columns), while columns that already have a type in the file (numbers,
booleans and dates in Excel, JSON, Parquet) keep it.

**Workbooks with several sheets.** After the first sheet with values was read, the dialog asks
which sheet to import; choosing another one imports that sheet instead. Over the API, pass
the option `sheet` (the name); without it, the first sheet with values is imported, and the
report lists all sheets.

**From a link.** The tab *From a link* imports a file that is on the public internet: the
server downloads it (up to the import size limit) and reads it like an upload. Share links
are turned into download links:

| Link | Downloaded |
| ---- | ---------- |
| Google Sheets `https://docs.google.com/spreadsheets/d/<id>/edit#gid=<tab>` (shared with *Anyone with the link*) | the tab as CSV (`…/export?format=csv&gid=<tab>`); published sheets (`…/d/e/…/pubhtml`) too |
| Google Drive `https://drive.google.com/file/d/<id>/view` | the file |
| GitHub `https://github.com/<owner>/<repo>/blob/<branch>/<path>` | `raw.githubusercontent.com/…` |
| Hugging Face `https://huggingface.co/datasets/<owner>/<name>/blob/<branch>/<file>` | `…/resolve/…` |
| Dropbox `…?dl=0` | `…?dl=1` |

Only `http`/`https` links without a user name or password work. Because the server makes
the request, links to private networks are refused (see [Security of link
imports](#security-of-link-imports)); a self-hosted installation can allow them with
`BRICKDNN_DATA_URL_ALLOW_PRIVATE=true`, e.g. to import from a file server in the same network.
The link is stored with the table (people you share a preparation with can see it), so do not
import links that contain secrets.

**Databases** (PostgreSQL, MySQL …) cannot be connected yet: a hosted service that connects to
databases its users name needs its own security design (credentials, network access). Export
the table as CSV or Parquet and import that.

## Train, validation and test rows

Right after combining tables and the row steps that run before the split (*Remove atypical
rows*, *Find near-duplicate groups*), every row is assigned to the **training**,
**validation** or **test** part (*Target & split*: shares, seed, and for classification
optionally *stratified*, which keeps the class proportions in every part). From then on:

- Steps that **learn values** – means and medians for missing values, clipping bounds,
  outlier limits, scaling statistics, categories, power-transform parameters – learn them from
  the **training rows only** and apply them to all rows. These steps are marked *learns from
  training rows*. This prevents information from the validation and test data leaking into
  the preparation.
- Removing outliers and balancing classes affect training rows only (by default), because
  you cannot remove or duplicate rows of new data at prediction time.
- The exported training dataset keeps this split. Training uses it instead of its own random
  split (the validation and test shares in the training settings are then fixed), so the test
  rows are never seen before the final test.

### Keep rows of a group together (split by group)

When several rows belong to the same person, patient, customer or object (repeated
measurements, several scans of one person), a random split puts some of them into training
and others into validation or test. The network then partly *recognises* the person instead
of predicting, and the validation and test scores are better than they will be for new
people. *Target & split → Keep rows of a group together* takes a column whose equal values
form a group (a person id, or the `group` column of *Find near-duplicate groups*): every group
goes as a whole into one part.

- Groups with several rows are placed first, largest first, each into the part where it
  brings the row counts closest to the shares; single rows fill the remaining room at
  random (seeded). The parts can only be as exact as the group sizes allow; a warning says
  when they differ by more than 3 percentage points from the shares.
- With *Stratify* (classification), groups are split per stratum, the most common class of
  each group, so the class shares of the parts stay close to each other. Regression targets
  are not stratified (as without groups).
- Rows without a group value are each a group of their own.
- The column must exist **where the split is assigned**: in the imported data, from a join,
  or from a step that runs before the split. *Target & split* lists those columns, and says
  how many groups there are and that no group is in two parts.
- The group column itself means nothing for new rows: drop it after the split (the readiness
  check suggests *Drop columns*).

The exported dataset carries this split in `split.bin`, so training uses it unchanged.

## Steps

| Group | Steps |
| ----- | ----- |
| **Combine data** | *Join another table* (match key columns; left, inner, outer or right join) · *Append rows* (stack another table with the same columns) |
| **Columns** | Drop · Keep only · Rename · Change type · *Formula* (new column from an expression, e.g. `log1p(income) / household_size`) · Replace values |
| **Rows** | Remove duplicates · Filter rows (=, <, between, is one of, contains, is missing …) · Drop rows with missing values · Sort · *Drop rows* (picked by hand: by the values of an ID column, or by row number) · *Remove atypical rows* and *Find near-duplicate groups* (before the split, see *Rows* below) |
| **Missing values** | *Mark as missing* (sentinel values like `-999` or `unknown`) · Drop sparse columns · *Fill missing values*: mean, median, most frequent, fixed value, nearest neighbours (KNN), forward/backward fill, interpolation, optionally with an indicator column |
| **Outliers & clipping** | *Outliers*: IQR rule, z-score, robust z-score (MAD) or quantiles; clip, remove, set missing or flag · *Clip values*: fixed bounds, quantiles (e.g. 1st–99th percentile) or mean ± k standard deviations |
| **Transform numbers** | Scale (standard, min-max, robust, max-abs) · log, log1p, square root, Box-Cox, Yeo-Johnson … · Quantile transform (uniform or normal) · Bin into intervals |
| **Encode categories** | One-hot, ordinal, frequency or target encoding · Group rare categories |
| **Dates & times** | Date parts (year, quarter, month, weekday, hour …), optionally cyclical (sin/cos) |
| **Target & balance** | Balance classes (oversample or undersample the training rows) |
| **Feature selection** | Drop near-constant, highly correlated, mostly missing or uninformative columns, or number columns with *no strong partner* (they correlate with no other feature: often noise) |

Steps that combine tables always come first, then the steps that run before the split
(*Remove atypical rows*, *Find near-duplicate groups*, marked *Runs before the split*). A step
that fails shows its error and stops the pipeline there; the preview shows the data before
it.

The training engine standardises all features again (mean 0, standard deviation 1, from the
training rows), so a *Scale* step is optional for training in Brick DNN; it is useful for the
exported files.

## Analysis

The **Analysis** view is computed from the prepared data (counts and statistics from all rows,
charts that show single rows from a random sample, which is always labelled):

- **Overview**: rows, columns, missing cells, duplicate rows, the split, column types, and
  warnings with suggested fixes (ID-like columns, constant columns, strong skew, outliers,
  high cardinality, imbalanced classes, columns that are almost numeric …). Warnings of one
  kind about three or more columns are one entry (“Missing values in 828 columns (6–9 % of
  the rows)”) that lists its columns when opened, with one fix for all of them.
- **Columns**: one row per column with type, missing share, distinct values, statistics, a
  small histogram and the association with the target. Click a column for details.
- **Distribution**: histogram, box plot, quantiles and statistics; split by target; top values
  of categories.
- **Missing values**: missing share per column and a matrix of where values are missing.
- **Relationships**: how the number columns relate to each other – see below.
- **Scatter**: any two columns, coloured by the target or another column, with a trend line.
- **Target**: class balance or the target distribution, and the columns ranked by their
  association with the target (|r| for numbers, correlation ratio η, Cramér's V for
  categories). Below the ranking, *Drop below association* picks a threshold and lists the
  features under it: **Add a step** adds “Select features: little association with the
  target”, which measures again on the training rows every time; **Drop these columns**
  drops exactly the listed ones.
- **Compare**: a column before and after the preparation.
- **Rows**: the structure of the rows, see below.

### Rows: atypical rows and near twins

The other views look at columns. **Rows** compares every row with all others, so that two
problems become visible that no column statistic shows:

- **Unusually typical rows**: rows whose every value sits close to its column median. Real
  rows are never typical in *all* columns; filler or synthetic rows often are, and their
  labels are usually made up. Outlier steps do not find them – they are the opposite of
  outliers.
- **Near twins**: rows whose nearest row is far closer than a typical pair of rows, e.g.
  several scans of the same person. With a random split, a row in validation often has its
  twin in training, and validation looks better than it is.

How it is computed (on the data the preview shows, with the scaling learned from the training
rows):

- Every number column is put on one scale with the **robust z-score**
  `(x − median) / (1.4826 · MAD)`, clipped to ±10.
- The **spread** of a row is the median of its `|z|`: about 0.67 for ordinary rows, near 0
  for a row whose every value is at the median.
- The **distance** of two rows is the root mean square of their z differences over the
  columns both have (missing values are left out; pairs that share fewer than half of the
  columns have no distance). The **NN ratio** of a row is the distance to its nearest row
  divided by the **typical pair distance** (the median distance of random pairs): 1 means "as
  far as any two rows", 0.2 "five times closer".
- **Groups**: rows linked to one of their 10 nearest rows with an NN ratio below the
  threshold, as connected groups.
- **Which columns**: *Automatic* uses all number columns in tables with up to 20 of them, and
  in wider tables only the columns with a **strong partner** (largest |Spearman r| with
  another column at least 0.3, or more for few rows). In wide tables, columns without any
  correlated partner are mostly noise, and noise columns drown the differences between rows:
  with 600 noise columns out of 830, near twins are invisible on all columns but obvious on
  the partnered ones. *Partnered*, *All number columns* and *I choose* set it by hand. The
  target, ID columns and the group column of the split are never used.
- **Size**: brute force in float32, computed when the view opens and cached per data and
  columns (the preview does not wait for it). Up to 20,000 rows with up to 50 compared
  columns, about 9,000 rows with 230 columns and 4,500 with 1,000 (rows² × columns ≤ 2·10¹⁰,
  a few seconds of CPU); above that a seeded random sample, which the view says (a row's twin
  may then be outside the sample). *Find near-duplicate groups* has the same limit and asks
  for an existing group column (e.g. a person id) on larger tables.

What the view shows:

- **Two histograms with thresholds**: the spreads (a separate low cluster = unusually typical
  rows) and the NN ratios, with the ratios of random pairs as a dashed outline (a big mass far
  left of the outline = near twins). Drag a threshold (or use the arrow keys, or type it); the
  view counts the rows below it. A dashed green line marks a **clear gap** the analysis found;
  ordinary data has none, and then nothing is suggested.
- **Close pairs vs random pairs**: how much the target differs between a row and its near
  twin, compared with random pairs (e.g. 2 years of age against 10). This uses the labels:
  it tells what the twins are, not which rows to drop.
- **Projection**: the rows on the first two principal components of the compared columns,
  coloured by structure, split part, target, spread or NN ratio. Twins sit on top of each
  other; click a point to see its similar rows.
- **Row table**: every analysed row with its id, split part, target, spread, missing values,
  NN ratio, nearest row and group; sort by any column, filter (unusually typical, with a near
  twin, the group of the chosen row, selected), find a row or id, select rows, *Select all
  shown*.
- **Similar rows**: click a row to see its 10 nearest rows with distances, targets and split
  parts, and for a pair the columns in which they agree and differ most.

Acting on it:

- *Remove N rows (step)* adds **Remove atypical rows** with the threshold. It runs before the
  split on all rows and removes them from every part: they are errors, not rare cases, and
  left in validation or test their arbitrary labels only add noise to the scores. Without a
  threshold the step uses the gap it finds itself (and removes nothing if there is none). The
  pandas script removes such rows from new data as well; delete those lines to keep every row.
- *Find groups & split by them (step)* adds **Find near-duplicate groups** (a column `group`
  with the group number; single rows get their own number) and switches the split to *Keep
  rows of a group together* on it. The step finds the groups again on all rows before the
  split, so its counts can differ a little from the view.
- *Drop N selected rows (step)* adds **Drop rows** with the ids of the selected rows (or
  their row numbers if there is no ID column; row numbers count the rows right before the
  step).

An advanced idea, deliberately not offered as a step: a near twin's known target is a strong
predictor (a person's age from their earlier scan). Adding it as a feature leaks the target
unless it is computed out of fold – every training row may only see twins from other folds,
and validation and test rows only twins from training – otherwise the network learns to copy
its own label.

### Relationships between columns

For every number and yes/no feature, the **strongest partner** is the other feature it
correlates with most (Spearman ρ, the correlation of ranks; computed for all columns on the
training rows, a random sample of them on large tables). Measurements of one object – pixels,
sensor channels, scan features, questions about one topic – usually have partners. A column
related to no other column is often **noise**: added on purpose, broken, or unrelated to the
rest; a network learns such columns by heart.

- **Strongest partner of each column**: a histogram of every column's largest |ρ|. Drag the
  line (or type a threshold) and the count of columns without a partner above it updates. The
  dashed line is the **chance level**: unrelated columns reach about this |ρ| by chance, for
  the number of rows and columns (with 850 rows and 830 columns about 0.12). Read the
  histogram like this: a hump near the chance level and another one clearly to the right
  means two kinds of columns; put the threshold in the gap. One hump means the columns are
  alike. The suggested threshold is 0.15, or the level that 99.9 % of unrelated columns stay
  below if that is higher (few rows, many columns).
- **All columns, related ones together**: the whole correlation matrix (|ρ|) ordered by
  hierarchical clustering, so that groups of related columns form dark blocks on the
  diagonal; on wide tables the cells are blocks of columns. A click shows those columns in
  the heatmap.
- **Columns and their partners**: one row per column with its strongest partner, ρ, Pearson
  r with the same partner, the number of partners with |ρ| ≥ 0.3, its group of related
  columns, the association with the target and the missing share. Sort by ρ to see the
  columns without a partner first; filter by name, by the threshold, by the number of partners
  or by group. Tick columns (or *Select all matching* the filter) and add a step: *Drop
  selected*, or *Keep only selected* (drops the other number columns). A click on a name
  shows its five strongest partners and a scatter plot with the strongest one.
- **Heatmap of chosen columns**: Pearson or Spearman for the strongest pairs, the selected
  or filtered columns of the table, a group or blocks of the overview (up to 60 columns),
  optionally ordered so that related columns are next to each other. It says how many of the
  columns it shows and what the values are based on.

*Drop the … without a partner* adds the step *Select features* with the method **No strong
partner** at the threshold: it learns the columns from the training rows (the same numbers as
the view), so the choice is reproducible and part of the exported pandas script. In the step
editor, column lists can also be filled by *Select all matching* a search or by *Paste a
list* (names separated by commas, spaces or new lines).

## Readiness and sanity checks

The row checks run with every preview on a seeded sample small enough to stay fast (rows² ×
compared columns ≤ 5·10⁸: all 1,212 rows of a table with 230 compared columns, about 1,500
of a larger one; at most 4,000) and only speak up when the Rows analysis finds a clear gap:
**unusually typical rows** (fix: *Remove atypical rows*), and **near twins** when at least 5 %
of the rows have one and at least 5 % of the validation/test rows have a twin in the
training rows (fix: *Find near-duplicate groups* and split by its column). With a split by
groups, the twin check stays quiet unless the groups do not hold the twins. A group column
that is still a feature gets a *Drop* fix; a group column that does not exist where the split
is assigned is an error.

The **Readiness** view sorts what it finds into three lists. Each entry names the affected
columns and, where a step solves it, has a button that adds that step (*Fix all* adds the
fixes of all problems at once). The checks look at the prepared data, so a fix shows its
effect with the next preview. Statistics come from the **training rows** (a random sample
of 100,000 of them on large tables, fewer for very wide ones), like the steps learn from
them; duplicates compare all rows (up to a million).

**Must be fixed** – the training dataset cannot be created before:

| Check | Fix |
| ----- | --- |
| No target, missing target values, a date target, a non-numeric target for regression, fewer than 2 or more than 1,000 classes | *Drop rows with missing values* in the target · *Change type* to numbers |
| No feature columns, fewer than 4 rows, no training or validation rows, infinite values | – |
| Only numbers as column names | *Rename columns* |
| Missing values in features | *Fill missing values* (median) |
| Categorical features | *One-hot encode* |
| … holding numbers as text, also with units, currency signs or thousands separators (`$1,200`, `12 kg`, `45 %`; codes such as `Z12345`, `01067` or `12A` are left alone) | *Change type* to numbers (with *Ignore units*) |
| … holding dates as text (`2024-01-31`, `31.01.2024` …) | *Change type* to dates, then *Date parts* |
| … with a different value in (almost) every row (names, IDs) | *Drop columns* |
| … whose values differ only in upper/lower case or spaces (`Yes`, `yes`, `YES `) | *Replace values* with the most frequent spelling |
| … with more than 50 categories in the training rows (one-hot would keep the 30 most frequent) | *Encode* by target mean (regression) or frequency |
| Free text features | *Drop columns* |
| Date features | *Date parts* (year, month, weekday) |

**Worth a look** – training works, but the network or its scores probably suffer:

| Check | When | Fix |
| ----- | ---- | --- |
| Target leakage | A feature equals the target in ≥ 99 % of the rows, correlates with a numeric target with \|r\| ≥ 0.98, or predicts the class with ≤ 1 % errors (and ≤ 10 % of the errors of always guessing the largest class) – by its values (values with ≥ 5 training rows covering ≥ 80 % of them), by a single threshold (two classes) or by being almost constant within each class (correlation ratio ≥ 0.99); or its name is the target's with `_id`, `_label`, `_code`, `is_` … | *Drop columns* |
| Validation/test rows that copy training rows | ≥ 1 % of the validation and test rows (fewer: a suggestion) | *Remove duplicates* |
| Rare classes | Fewer than 5 training rows | *Filter rows*: remove their rows (if at least 2 classes remain) |
| Classes missing from the validation or test rows | A class with ≥ 5 training rows has none there | Turn on *stratified* split |
| Spellings of a class that differ only in case or spaces | | *Replace values* |
| ID columns | Whole numbers, different in (almost) every row, named like an ID (`id`, `customer_id`, `userId` …) or numbering the rows 0/1 … n | *Drop columns* |
| Too few training rows | Fewer than 50, or fewer than 5 per feature | – |
| Imbalanced classes | More than 3 : 1 | *Balance classes* |
| Duplicate rows | ≥ 5 % of the rows (fewer: a suggestion) | *Remove duplicates* |
| Constant features | One value in all rows | *Drop columns* |
| Features without a strong partner | At least 20 number features, of which ≥ 10 (and ≥ 10 %) correlate with no other feature at the suggested threshold (0.15, or the 99.9 % chance level if higher), ≥ 10 (and ≥ 10 %) have a partner at twice that, and at most 10 % lie in between (a clear gap). Tables with a few independent features stay quiet. The message says whether these columns are related to a numeric target | *Select features*: no strong partner |
| More than 1,000 features | | – |

**Suggestions** – worth considering:

| Check | When | Fix |
| ----- | ---- | --- |
| Conflicting labels | ≥ 1 % of the rows have the same features as another row but a different class | – |
| Near-constant features | One value in ≥ 99 % of at least 100 training rows | *Drop columns* |
| Redundant features | Two numeric features with \|r\| ≥ 0.98; the one less associated with the target is dropped | *Drop columns* |
| Skewed features or a skewed numeric target | \|skewness\| > 2 with at least 10 different values, if log(1 + x) halves it (values ≥ 0) or for negative values | *Transform* log1p or Yeo-Johnson |
| Very different scales | The largest standard deviation of the numeric features is ≥ 1,000 times the smallest. Training in Brick DNN standardises the features anyway; this matters for the exported files | *Scale* (standard) |

After *Balance classes* (oversampling), duplicate rows are expected and not reported, and the
*Remove duplicates* fix is not offered, as it would undo the oversampling.

## Export

| Download | Contents |
| -------- | -------- |
| **Training dataset** | Stored on the server, appears in the training panel. Numeric features plus the target as last column; class labels are kept as class names. |
| **Prepared data (.csv)** | The prepared table with a `split` column (train/val/test). |
| **Recipe (.json)** | Source files, import settings, every step with its settings and the values it learned. |
| **Python script (.py)** | A standalone pandas script with a `prepare(df)` function that applies the same steps with the learned values written in, so new data for predictions is prepared exactly like the training data. |

The script reads each source like the import did (same format, sheet, separator, decimal mark,
missing-value markers and column types; packed files too). It needs pandas and numpy, plus
`openpyxl` for Excel `.xlsx`, `xlrd` for `.xls` and `pyarrow` for Parquet / Feather sources:

```bash
python churn_prepare.py --input customers.csv --output prepared.csv   # --source-2 regions.csv … for joined tables
```

```python
from churn_prepare import FEATURES, MAIN_SOURCE, SOURCES, prepare, read_source

raw = read_source("new_customers.csv", SOURCES[MAIN_SOURCE])  # same import settings
new = prepare(raw)                      # same steps, same learned values
x = new[FEATURES].to_numpy("float32")   # inputs for the trained network
```

Tables added with *Join* or *Append* are read by `combine(df, load_source)` before
`prepare`, as in the script's `main()`. Row steps such as *Remove duplicates* or *Filter rows*
also apply to new data; class balancing does not.

## Limits and settings

| Variable | Default | Meaning |
| -------- | ------- | ------- |
| `BRICKDNN_DATA_PREP` | `true` | Set to `false` to switch the Data tab off |
| `BRICKDNN_DATA_MAX_UPLOAD_MB` | `200` | Largest import in MB: uploads, downloads from links and the unpacked content of `.gz` / `.zip` files |
| `BRICKDNN_DATA_MAX_ROWS` | `2000000` | Most rows per imported file |
| `BRICKDNN_DATA_MAX_COLUMNS` | no limit | Most columns per imported file (a single step may add at most 2,000 columns) |
| `BRICKDNN_DATA_URL_IMPORT` | `true` | Set to `false` to switch imports from links off |
| `BRICKDNN_DATA_URL_ALLOW_PRIVATE` | `false` | Set to `true` to allow links to private, loopback and other non-public addresses (only for installations in a trusted network) |
| `BRICKDNN_DATA_RATE` | `1200/hour` | Previews per user (they run on every change) |
| `BRICKDNN_UPLOAD_RATE` | `60/hour` | Imports per user (files and links count together) and dataset uploads |

Data is processed in memory with pandas. Importing needs roughly 10–15 times the file size
in RAM for a moment (a 100 MB file: about 1.5 GB) and takes a few seconds per 10 MB; previews
afterwards are fast, because the imported table is stored column by column.

Formats that pack data well get extra limits, checked before and while reading, so that a
small file cannot expand into a huge table: the unpacked content of a `.gz` / `.bz2` / `.xz`
/ `.zip` file may be at most `BRICKDNN_DATA_MAX_UPLOAD_MB`; the unpacked XML of an Excel or
OpenDocument workbook and the uncompressed data of a Parquet or Feather file at most four
times that; and a table has at most as many cells (rows × columns) as a CSV file of the size
limit could hold (half a cell per byte). Excel files are slower to read than CSV files
(roughly 10–20 seconds per million cells).

Excel `.xlsx`, `.xls`, Parquet and Feather need `openpyxl`, `xlrd` and `pyarrow` (part of
`requirements.txt` and of `pip install "brickdnn[data]"`; pyarrow adds about 170 MB to the
Docker image). Without them, the other formats keep working and these files get a clear
message.

## Security of link imports

The server downloads the file at a link, so a link must not reach services that are only
meant to be reachable from the server (server-side request forgery):

- only `http` and `https`; links with a user name or password are refused; no cookies,
  credentials or headers of the user are sent, and proxies from the environment are not used;
- the host name is resolved once, and **every** address it resolves to must be public:
  private (`10/8`, `172.16/12`, `192.168/16`, `fc00::/7`), loopback, link-local (including
  cloud metadata at `169.254.169.254`), multicast, shared (`100.64/10`), reserved and other
  special IPv4 and IPv6 addresses are refused, also when they are hidden in IPv4-mapped,
  6to4, Teredo or NAT64 IPv6 addresses;
- the connection goes to exactly the address that was checked (a second DNS lookup could
  answer differently), while TLS still verifies the certificate for the host name;
- redirects are followed by hand, at most 5, and every new location is checked again;
- the connection times out after 10 seconds, a pause of 30 seconds ends the download, and the
  whole download may take 3 minutes; the size limit applies while the file streams in (a
  larger `Content-Length` is refused at once);
- answers that are web pages (HTML) are refused with a hint, e.g. for Google Sheets that are
  not shared publicly.
