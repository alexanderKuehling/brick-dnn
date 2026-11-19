# Hugging Face Hub

Brick DNN works with the [Hugging Face Hub](https://huggingface.co) in two places, both only
when a user starts them:

- **Publish a trained run** as a model repository of the user's Hugging Face account: the
  trained model, the generated script, the network, the preprocessing and a model card.
- **Import a file of a dataset** on the Hub into the Data tab, like an import from a link.

Nothing is sent to Hugging Face otherwise: no account is linked, nothing runs in the
background and the browser loads nothing from huggingface.co.

## Access tokens

Publishing needs a Hugging Face access token that may write
([huggingface.co → Settings → Access Tokens](https://huggingface.co/settings/tokens): the role
*Write*, or a fine-grained token with write access to the repositories of the account or
organization); importing needs one only for gated or private datasets (read access is enough).

The user pastes the token for the one request that uses it. Brick DNN

- sends it in the `Authorization` header to the Hub's own host only: not to the storage
  servers the Hub hands out for uploads or redirects downloads to;
- never stores it: not in the database, not in the browser's storage, not in the run, the
  import report, the activity log or an approval request;
- never logs it: the value is wrapped as soon as the request is read (`hub_client.HubToken`
  shows `HubToken(hidden)` in every log line, traceback and error report and cannot be
  pickled), messages of the Hub are cleaned of anything that looks like a token, and the
  activity log stores `[redacted]` for the argument.

The dialogs keep the token in memory only while they are open.

## Publishing a run

A finished or stopped run with weights has **Publish to Hugging Face** under *Model and
files* (a neural network, a classical model, a hybrid network, a segmentation or a text
classifier; not an ensemble: publish its members). Who started the run and the owner and
editors of its network may publish it. The dialog asks for

- the access token;
- the repository name (default: from the network's name; `organization/name` publishes into
  an organization);
- private (the default) or public;
- the licence (`apache-2.0` by default, `mit`, `bsd-3-clause`, `cc-by-4.0`, `cc-by-sa-4.0`,
  `cc-by-nc-4.0`, `cc0-1.0`, `gpl-3.0`, `openrail` or `other`);
- whether a repository that exists already may be updated (off by default: a taken name is an
  error; an update replaces the files of the same name and leaves the visibility alone).

It lists the files with their sizes and shows the model card before anything is sent.

### The files

| File | Content |
| ---- | ------- |
| `model.safetensors` | A neural network's weights (the `state_dict` of `NeuralNetwork`). `model.pt` (`torch.save`) if they cannot be converted |
| `model.joblib` | A classical model: the fitted scikit-learn pipeline with class names and columns. A pickle, as on download |
| `model.joblib` and `model.features.pt` | A hybrid network: the classical model, and the neural layers that compute its features (`FeatureExtractor`, a `torch.save` checkpoint) |
| `<name>.py` | The generated script: the model's code and, for networks that read a table, training on a CSV file and prediction from the command line |
| `network.json` | The network's graph ([graph format](graph-format.md)); opens in Brick DNN |
| `preprocessing.json` | What new inputs need: columns, means and standard deviations, the image size and channel statistics, or the vocabulary, with the class names |
| `README.md` | The model card |

**safetensors.** The weights are converted from the run's `weights.pt` in a process of its
own (`python -m brickdnn.training.safetensors_file`), written by Brick DNN itself: the format
is a JSON header followed by the tensors' bytes, so the server needs no new dependency, and
unlike a `torch.save` file it holds no pickle. `safetensors.torch.load_file` reads it. The
neural part of a hybrid network stays a `torch.save` checkpoint, because the generated script
and the download read it like that, and its `model.joblib` is a pickle anyway.

**The model card** has YAML metadata (`library_name` `pytorch` or `sklearn`, `license`,
`tags` with `brickdnn` and the task, `pipeline_tag` such as `tabular-classification`,
`image-segmentation` or `text-classification`, `datasets` for the built-in datasets that are
on the Hub, `metrics` and a `model-index` with the results of every split) and a text: what
the model is, its inputs and classes, the layers with their parameters, the training
settings, the results per split, the files, Python code that downloads the repository with
`huggingface_hub`, loads the model with the included script and predicts one sample, and a
link back to Brick DNN.

### Limits and errors

- The files of one publish together may have `BRICKDNN_HUGGINGFACE_MAX_UPLOAD_MB` (250) MB;
  larger runs are refused before anything is sent (`413`): download them and upload them
  with the Hugging Face tools.
- `BRICKDNN_HUGGINGFACE_RATE` (`30/hour` per user) limits publishing and listing datasets.
- A publish may take 240 seconds; then it stops (`504`).

| Problem | Answer |
| ------- | ------ |
| The Hub does not accept the token (HTTP 401) | `400`, `code: token_invalid` (not `401`, which would log the user out of Brick DNN) |
| The token can only read | `403`, `token_read_only` |
| The token may not write there (another account or organization) | `403`, `forbidden` |
| The repository exists and updating was not allowed | `409`, `name_taken` |
| Too large for this server or for the Hub (HTTP 413) | `413`, `too_large` |
| The Hub is busy, cannot be reached or takes too long | `429` `busy`, `502` `unreachable`, `504` `timeout` |

### API

| Method | Path | Description |
| ------ | ---- | ----------- |
| GET | `/api/training/runs/<id>/huggingface/?repo=` | What would be uploaded: `files` with sizes, `size`, `too_large`, `model_card`, `default_repo`, `can_publish`. Sends nothing |
| POST | `/api/training/runs/<id>/huggingface/` | `{hf_token, repo?, private?, license?, update_existing?}` → `201 {repo_id, url, created, private, license, commit, commit_url, files, size}` |

`GET /api/training/` has `huggingface: {publish, max_upload_mb, licenses, default_license,
url}`. With an API token whose approval rule covers publishing the `POST` is refused (the
REST API cannot ask); use MCP or the web app.

### MCP

`publish_to_huggingface {run_id, hf_token, repo?, private?, license?, update_existing?}`
does the same for an AI assistant. It publishes for the user, so it belongs to the
**publishing** actions of the [approval rules](mcp.md#approvals): if the rule of the token or
app covers publishing, the first call uploads nothing and answers `approval_required`. The
request shows the run, the repository, public or private and the licence, never the token.
Because the token is not stored, approving cannot upload by itself: it allows the assistant
to publish exactly what was asked, once, within 24 hours. The assistant then calls the tool
again with the same arguments, `approval_id` and the token. Other settings than the approved
ones are refused, a failed attempt keeps the approval, and a second use is refused.

## Importing a dataset file

The import dialog of the Data tab has a tab **Hugging Face**. Enter a dataset as
`hf://datasets/<owner>/<name>`, `<owner>/<name>` or the link of its page on huggingface.co
(also a `tree` or `viewer` page; a dataset link pasted into *From a link* offers to open the
tab). **Show files** asks the Hub for its files and lists the CSV, TSV, JSON, JSON Lines and
Parquet files (also packed with gzip, bzip2 or xz) with their size, config and split; with
several configs or splits, two filters narrow the list. Config and split come from the
`configs` of the dataset card or, without them, from the usual names
(`data/train-00000-of-00001.parquet`, `<config>/test.csv`). A dataset without such files, for
example one with a loading script, is listed from the Parquet files Hugging Face converted it
to (`refs/convert/parquet`).

The chosen file is imported like a [link import](data-preparation.md#importing-data): the
same size limit (`BRICKDNN_DATA_MAX_UPLOAD_MB`), time limit, address checks and rate limit,
the same import options, and the source remembers the link it came from. A file of a dataset
can also be imported directly, without the list: `hf://datasets/<owner>/<name>/<path>` (a
revision as `<name>@<revision>`) or the link of the file on huggingface.co, as the `url` of
`POST /api/data/sources/` or of the MCP tool `import_data`; named like that, any file type
the import reads works (an Excel file, a ZIP file).

Public datasets need no token. For a gated dataset, accept its conditions on its page first,
then enter a token that may read it; a private dataset needs the token for the list and for
the import.

| Method | Path | Description |
| ------ | ---- | ----------- |
| POST | `/api/data/huggingface/files/` | `{dataset, hf_token?}` → `{repo_id, revision, converted, page, private, gated, files: [{path, size, format, config, split, ref}], configs, splits, truncated, …}` (POST, so that a token never is part of an address) |
| POST | `/api/data/sources/` | `{url: <ref of a file>, hf_token?, name?, ...import options}`: the import |

Over MCP, `import_data {url: "hf://datasets/<owner>/<name>"}` answers with the list of files
and their refs; called again with one of them it imports the file (`hf_token` for gated or
private datasets).

## Settings

| Variable | Default | Meaning |
| -------- | ------- | ------- |
| `BRICKDNN_HUGGINGFACE` | `true` | `false` switches publishing and the dataset import off (the button and the tab disappear; the API answers `403`). The import also needs `BRICKDNN_DATA_URL_IMPORT` |
| `BRICKDNN_HUGGINGFACE_MAX_UPLOAD_MB` | `250` | The files of one publish together |
| `BRICKDNN_HUGGINGFACE_RATE` | `30/hour` | Publishes and dataset listings per user |
| `BRICKDNN_HUGGINGFACE_URL` | `https://huggingface.co` | Another Hub (the tests use a local fake one) |
| `BRICKDNN_HUGGINGFACE_PROXY` | – | An HTTP proxy for the requests to the Hub's API and the uploads. Proxies from the environment are not used. Downloads of dataset files connect directly, like every link import |

The server must be able to reach `huggingface.co` and the storage hosts it hands out
(currently under `hf.co`) on port 443.

## How it talks to the Hub

Brick DNN uses the Hub's public HTTP API directly, with Python's standard library
(`brick_dnn_core/hub_client.py`), not the `huggingface_hub` package. The integration needs a
handful of calls, and the package would bring `requests` or `httpx`, `tqdm`, `fsspec`,
`filelock`, `pyyaml` and a native extension for Xet storage into the web server, and it reads
and caches tokens on disk and in the environment by itself, which is the opposite of "used
for one request, never stored". The calls:

| Call | For |
| ---- | --- |
| `GET /api/whoami-v2` | Whose token it is (the default owner of the repository) and whether it may write |
| `POST /api/repos/create` | The model repository; `409` if it exists |
| `POST /api/models/<repo>/preupload/main` | Which files the Hub wants as Git LFS objects |
| `POST /<repo>.git/info/lfs/objects/batch`, `PUT` to the address it returns, `verify` | Upload of those files; ones the Hub has already are skipped |
| `POST /api/models/<repo>/commit/main` | One commit with all files (small ones inline, large ones by their SHA-256) |
| `GET /api/datasets/<repo>`, `…/tree/<revision>?recursive=true` | A dataset, its card and its files |
| `GET /datasets/<repo>/resolve/<revision>/<path>` | A file (the link import, `url_import.py`) |

Redirects are followed by hand and only within the Hub's host for API calls, answers are
limited in size, and every request has a timeout.

## Privacy

Publishing sends the listed files to Hugging Face, Inc. (USA), into the repository and with
the visibility the user chose; importing asks Hugging Face for the dataset's files and
downloads one. Hugging Face sees the server's IP address, not the user's. Brick DNN stores
nothing of it besides the imported file and its link; what was published is deleted in the
Hugging Face account. The privacy policy of a public server should say so:
[privacy.md](privacy.md#hugging-face) and the template in `deploy/legal/`.
