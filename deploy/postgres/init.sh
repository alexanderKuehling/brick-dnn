#!/bin/sh
# Runs once, when the db service of docker-compose.yml creates its data directory: the postgres
# image runs the scripts in /docker-entrypoint-initdb.d/ only into an empty volume, never again.
#
# Brick DNN logs in as "brickdnn", the owner of the database "brickdnn" but no superuser, with
# the password BRICKDNN_POSTGRES_PASSWORD of .env (POSTGRES_PASSWORD in here). The superuser
# "postgres" loses its password: it can only log in inside the container, through the socket
# (`docker compose exec db psql -U postgres`), not over the network.
#
# Changing BRICKDNN_POSTGRES_PASSWORD later does not change the role's password; see
# deploy/server/README.md, "The PostgreSQL service".
set -eu

psql --no-psqlrc --set=ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname postgres \
  --set=password="$POSTGRES_PASSWORD" <<'SQL'
CREATE ROLE brickdnn LOGIN PASSWORD :'password';
CREATE DATABASE brickdnn OWNER brickdnn;
ALTER ROLE CURRENT_USER PASSWORD NULL;
SQL
