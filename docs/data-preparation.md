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

Right after combining tables, every row is assigned to the **training**, **validation** or
**test** part (*Target & split*: shares, seed, and for classification optionally
*stratified*, which keeps the class proportions in every part). From then on:

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

## Steps

| Group | Steps |
| ----- | ----- |
| **Combine data** | *Join another table* (match key columns; left, inner, outer or right join) · *Append rows* (stack another table with the same columns) |
| **Columns** | Drop · Keep only · Rename · Change type · *Formula* (new column from an expression, e.g. `log1p(income) / household_size`) · Replace values |
| **Rows** | Remove duplicates · Filter rows (=, <, between, is one of, contains, is missing …) · Drop rows with missing values · Sort |
| **Missing values** | *Mark as missing* (sentinel values like `-999` or `unknown`) · Drop sparse columns · *Fill missing values*: mean, median, most frequent, fixed value, nearest neighbours (KNN), forward/backward fill, interpolation, optionally with an indicator column |
| **Outliers & clipping** | *Outliers*: IQR rule, z-score, robust z-score (MAD) or quantiles; clip, remove, set missing or flag · *Clip values*: fixed bounds, quantiles (e.g. 1st–99th percentile) or mean ± k standard deviations |
| **Transform numbers** | Scale (standard, min-max, robust, max-abs) · log, log1p, square root, Box-Cox, Yeo-Johnson … · Quantile transform (uniform or normal) · Bin into intervals |
| **Encode categories** | One-hot, ordinal, frequency or target encoding · Group rare categories |
| **Dates & times** | Date parts (year, quarter, month, weekday, hour …), optionally cyclical (sin/cos) |
| **Target & balance** | Balance classes (oversample or undersample the training rows) |
| **Feature selection** | Drop near-constant, highly correlated, mostly missing or uninformative columns |

Steps that combine tables always come first. A step that fails shows its error and stops the
pipeline there; the preview shows the data before it.

The training engine standardises all features again (mean 0, standard deviation 1, from the
training rows), so a *Scale* step is optional for training in Brick DNN; it is useful for the
exported files.

## Analysis

The **Analysis** view is computed from the prepared data (counts and statistics from all rows,
charts that show single rows from a random sample, which is always labelled):

- **Overview**: rows, columns, missing cells, duplicate rows, the split, column types, and
  warnings with suggested fixes (ID-like columns, constant columns, strong skew, outliers,
  high cardinality, imbalanced classes, columns that are almost numeric …).
- **Columns**: one row per column with type, missing share, distinct values, statistics, a
  small histogram and the association with the target. Click a column for details.
- **Distribution**: histogram, box plot, quantiles and statistics; split by target; top values
  of categories.
- **Missing values**: missing share per column and a matrix of where values are missing.
- **Correlations**: Pearson and Spearman heatmaps and the strongest pairs.
- **Scatter**: any two columns, coloured by the target or another column, with a trend line.
- **Target**: class balance or the target distribution, and the columns ranked by their
  association with the target (|r| for numbers, correlation ratio η, Cramér's V for
  categories).
- **Compare**: a column before and after the preparation.

## Readiness and sanity checks

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
