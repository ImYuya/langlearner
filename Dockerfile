# Base image
FROM python:3.11-slim-bullseye

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
    gcc \
    g++ \
    build-essential \
    python3-dev \
    vim

# Install Poetry
RUN pip install --upgrade pip
RUN pip install poetry

# Set the working directory in the container
WORKDIR /src

# Copy the Poetry version management file to the container
# Install libraries based on this
COPY poetry.lock pyproject.toml .

# Install libraries
RUN poetry config virtualenvs.create false && poetry install

# Copy the source
COPY src/ /src/

# Execute command
# To use static files, enable the enableStaticServing option
CMD ["streamlit", "run", "--server.port", "8501", "--server.enableStaticServing", "true", "/src/app.py"]