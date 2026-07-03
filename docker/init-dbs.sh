#!/usr/bin/env bash
# Creates mlflow and airflow databases (churnops is created by POSTGRES_DB env var).
# Using a shell script so we can silently skip already-existing databases.
set -e

for db in mlflow airflow; do
    if psql -U "$POSTGRES_USER" -lqt | cut -d\| -f1 | grep -qw "$db"; then
        echo "Database '$db' already exists, skipping."
    else
        echo "Creating database '$db'..."
        psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -c "CREATE DATABASE $db;"
        psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -c "GRANT ALL PRIVILEGES ON DATABASE $db TO $POSTGRES_USER;"
    fi
done
