# Database

Brick DNN keeps accounts, saved networks, datasets, data preparations and training runs in a
database. By default this is a SQLite file, which needs no setup. For a server that several
people use, or to keep the data in a managed database, set `DATABASE_URL` to use PostgreSQL
instead, for example a free [Supabase](https://supabase.com) project.

Files stay on the machine that runs the backend in either case: uploaded datasets, imported
CSV files, prepared data, training runs and trained weights live in `BRICKDNN_DATA_DIR`
(`backend/data`, `/data` in Docker). Back up this directory as well as the database.

| Variable                   | Default                   | Meaning                                            |
| -------------------------- | ------------------------- | -------------------------------------------------- |
| `DATABASE_URL`             | – (SQLite)                | `postgresql://USER:PASSWORD@HOST:PORT/NAME`        |
| `BRICKDNN_DB_PATH`         | `BRICKDNN_DATA_DIR/db.sqlite3` | The SQLite file (without `DATABASE_URL`)      |
| `BRICKDNN_DB_CONN_MAX_AGE` | `60`                      | Seconds a PostgreSQL connection is reused; `0` closes it after every request |

## PostgreSQL

Put the URL into the file `.env` in the repository root (ignored by git):

```bash
DATABASE_URL=postgresql://brick:PASSWORD@db.example.org:5432/brickdnn
```

Then start the backend: `make backend` runs the migrations (creates the tables) and starts
the API; `docker compose up -d` does the same in the container. `manage.py`, the backend
server and Docker Compose all read `.env`; a variable set in the shell (`export …`) wins over
the file. The test suite ignores `.env`.

- **Special characters in the password** must be percent-encoded, otherwise the URL falls
  apart: `#` → `%23`, `/` → `%2F`, `?` → `%3F`, `@` → `%40`, `:` → `%3A`, `%` → `%25`. This
  prints the encoded form:
  `python3 -c "import sys, urllib.parse; print(urllib.parse.quote(sys.argv[1], safe=''))" 'pa#ss/word'`
- **Options** after `?` are passed to the PostgreSQL client, e.g.
  `?sslmode=verify-full&sslrootcert=/path/ca.crt` or `?connect_timeout=10`.
- **Connection poolers** that hand every transaction to another server connection (PgBouncer
  in transaction mode): add `?pgbouncer=true`, so Django does not use server-side cursors.
- PostgreSQL 14 or newer is needed (Django 5.2). The driver (`psycopg`) is part of
  `requirements.txt`.

## Supabase

1. Create a project on [supabase.com](https://supabase.com) and keep the database password.
   A region close to the server that runs Brick DNN keeps requests fast.
2. Open the project, click **Connect** and copy the connection string of the **Session
   pooler**. It looks like this:

   ```
   postgresql://postgres.abcdefghijklmnop:[YOUR-PASSWORD]@aws-0-eu-central-1.pooler.supabase.com:5432/postgres
   ```

3. Replace `[YOUR-PASSWORD]` with the database password (percent-encoded, see above) and set
   the result as `DATABASE_URL`, as described in [PostgreSQL](#postgresql).
4. Start the backend. The first `migrate` creates the tables and reports
   `Supabase: … tables are hidden from the Data API`.

Which connection string to use:

| Connection         | Port   | Use it                                                       |
| ------------------ | ------ | ------------------------------------------------------------ |
| Session pooler     | `5432` | Recommended: works everywhere (IPv4)                         |
| Direct connection  | `5432` | Only if the server can reach IPv6 addresses (Docker containers usually cannot), or with Supabase's IPv4 add-on |
| Transaction pooler | `6543` | Works too (Brick DNN switches off server-side cursors for it), but meant for serverless functions |

Brick DNN connects to Supabase with `sslmode=require` (encrypted only), unless the URL sets
another `sslmode`.

### The Supabase Data API

Supabase publishes the tables in the `public` schema through an automatic REST API (the Data
API). Anyone who has the project's *anon* key could read or change them there, unless row
level security (RLS) is switched on. Brick DNN's tables include the user accounts with their
password hashes, so after every `migrate` Brick DNN:

- switches on **row level security** for all its tables, without any policy, and
- **revokes** all privileges of the API roles `anon` and `authenticated` on them.

The API then refuses all access to these tables. Brick DNN itself is not affected: it connects as
the owner of the tables, and owners are not restricted by RLS. Brick DNN never needs Supabase's
API keys; do not put the *service_role* key anywhere, it bypasses RLS.

The Security Advisor in the Supabase dashboard lists these tables as "RLS enabled, no policy".
That is intended. If `migrate` prints a warning that it could not switch on RLS for some
tables (this happens if they belong to another database role), switch it on for them in the
dashboard: *Table Editor*, or `ALTER TABLE … ENABLE ROW LEVEL SECURITY;` in the *SQL Editor*.
If you do not use the Data API for anything else, you can also switch it off in the project's
API settings.

### Good to know

- Free Supabase projects are **paused after a week without activity**. Until you restore the
  project in the dashboard, the backend cannot connect and every request fails with a database
  error.
- Every backend process keeps **one connection** open (Docker: `GUNICORN_WORKERS`, default 2;
  training processes do not use the database). Supabase limits the connections per project;
  if they run out, set `BRICKDNN_DB_CONN_MAX_AGE` to `0`.
- Requests take a little longer than with SQLite, because every query travels to the Supabase
  region and back.

## Moving existing data to PostgreSQL

Copy accounts, networks, datasets, preparations and training runs from the SQLite file into an
empty PostgreSQL (or Supabase) database:

```bash
cd backend

# 1. Export from SQLite (DATABASE_URL must not be set in this shell).
.venv/bin/python manage.py dumpdata --natural-foreign \
  --exclude contenttypes --exclude auth.permission --exclude admin.logentry \
  --exclude sessions --exclude token_blacklist \
  --output data/move.json

# 2. Create the tables in PostgreSQL and load the data.
export DATABASE_URL='postgresql://…'
.venv/bin/python manage.py migrate
.venv/bin/python manage.py loaddata data/move.json

# 3. The export contains password hashes: delete it.
rm data/move.json
```

Everyone has to log in once more (login tokens are not moved). The files in
`BRICKDNN_DATA_DIR` stay where they are and keep working, so run the new setup with the same
data directory.

## Running the tests on PostgreSQL

```bash
cd backend
DATABASE_URL='postgresql://postgres@127.0.0.1:5432/brickdnn' .venv/bin/pytest
```

The test runner creates and deletes its own database `test_brickdnn`; do not point it at a
production database. With roles named `anon` and `authenticated` in the PostgreSQL server
(`CREATE ROLE anon NOLOGIN; CREATE ROLE authenticated NOLOGIN;`), the tests also check the
Supabase protection. CI runs the whole suite on PostgreSQL as well.
