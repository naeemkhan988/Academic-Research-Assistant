FROM python:3.10-slim

# Create non-root user (required by HuggingFace Spaces)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Install dependencies first (cached layer)
COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY --chown=user . /app

# Create required runtime directories
RUN mkdir -p /app/cache \
    /app/cashe \
    /app/static/images \
    /app/static/exports

# HuggingFace Spaces uses port 9838
EXPOSE 9838

CMD ["gunicorn", "--bind", "0.0.0.0:9838", "--timeout", "120", "--workers", "1", "--threads", "4", "app:app"]
