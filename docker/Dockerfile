FROM python:3.10-slim

WORKDIR /app

COPY pyproject.toml .

RUN pip install --upgrade pip
RUN pip install poetry

RUN poetry config virtualenvs.create false && poetry install --no-interaction --no-ansi