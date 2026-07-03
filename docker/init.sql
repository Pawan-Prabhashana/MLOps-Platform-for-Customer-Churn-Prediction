-- PostgreSQL init script
-- Creates the three application databases on first startup.
-- This file is mounted to /docker-entrypoint-initdb.d/ in the postgres container.

\connect postgres

CREATE DATABASE mlflow;
CREATE DATABASE airflow;
CREATE DATABASE churnops;

GRANT ALL PRIVILEGES ON DATABASE mlflow   TO churnops;
GRANT ALL PRIVILEGES ON DATABASE airflow  TO churnops;
GRANT ALL PRIVILEGES ON DATABASE churnops TO churnops;
