# Data preparation

Real data is rarely ready for a neural network: values are missing, a few outliers dominate a
column, categories are text, dates are strings, and the interesting table is spread over two
files. The **Data** tab turns such CSV files into a clean training dataset, shows what the data
looks like on the way, and keeps every decision visible and reproducible.

Like training, data preparation runs on the machine that runs the backend and needs an account,
because imported files are stored per user. It needs pandas in the backend environment (part
of `requirements.txt`); PyTorch is not needed to prepare data, only to train on it.

## Workflow

1. Open **Data** in the header and **import a CSV file**. Brick DNN detects the separator
   (`,` `;` tab `|`), decimal commas (`1,5`), the header line, the text encoding and common
   placeholders for missing values (`NA`, `N/A`, `null`, `-`, `?` …). Every column gets a type:
   *numeric*, *boolean*, *date/time*, *categorical* or *text*. If the detection is wrong, set
   the options under **Advanced** or fix the type later with a *Change type* step.
2. A **preparation** is created for the file. Choose the **target** (the column to predict)
   and the task (*automatic*, classification or regression) in the *Target & split* card.
3. Add **steps** (below). After every change the preview runs again: each step reports what
   it did ("Filled 132 missing values in 2 columns with the median."), the **Data** view shows
   a random sample of the rows, and **Analysis** shows charts and statistics. *Preview after
   this step* shows the data at any point of the pipeline.
4. The **Readiness** view lists what still prevents training (missing values, categorical or
   text features, a missing target …), each with a button that adds the fitting step.
5. **Create training dataset** (or **Train with this data**, which also opens the training
   panel in the builder). The dataset appears under *Your data* in the training panel, marked
   *Prepared*.

Preparations are saved automatically. Several preparations can use the same file.

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

## Export

| Download | Contents |
| -------- | -------- |
| **Training dataset** | Stored on the server, appears in the training panel. Numeric features plus the target as last column; class labels are kept as class names. |
| **Prepared data (.csv)** | The prepared table with a `split` column (train/val/test). |
| **Recipe (.json)** | Source files, import settings, every step with its settings and the values it learned. |
| **Python script (.py)** | A standalone pandas script with a `prepare(df)` function that applies the same steps with the learned values written in, so new data for predictions is prepared exactly like the training data. |

The script needs only pandas and numpy:

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
| `BRICKDNN_DATA_MAX_UPLOAD_MB` | `200` | Largest CSV import in MB |
| `BRICKDNN_DATA_MAX_ROWS` | `2000000` | Most rows per imported file |
| `BRICKDNN_DATA_MAX_COLUMNS` | no limit | Most columns per imported file (a single step may add at most 2,000 columns) |
| `BRICKDNN_DATA_RATE` | `1200/hour` | Previews per user (they run on every change) |

Data is processed in memory with pandas. Importing needs roughly 10–15 times the file size
in RAM for a moment (a 100 MB file: about 1.5 GB) and takes a few seconds per 10 MB; previews
afterwards are fast, because the imported table is stored column by column.
