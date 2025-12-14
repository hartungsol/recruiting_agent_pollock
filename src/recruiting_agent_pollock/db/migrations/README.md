# Database Migrations

This directory contains Alembic database migrations.

## Setup

To initialize Alembic for migrations:

```bash
alembic init migrations
```

## Creating Migrations

```bash
alembic revision --autogenerate -m "description"
```

## Applying Migrations

```bash
alembic upgrade head
```
