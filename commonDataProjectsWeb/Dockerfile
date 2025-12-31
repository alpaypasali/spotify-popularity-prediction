# Stage 1: Build Tailwind CSS
FROM node:20-slim AS tailwind-builder

WORKDIR /build

# Copy package files and install dependencies
COPY package.json package-lock.json* ./
RUN npm install

# Copy Tailwind source, config, and templates (needed for content scanning)
COPY static/css/input.css ./static/css/
COPY tailwind.config.js ./
COPY templates/ ./templates/
COPY projects/ ./projects/
COPY api/ ./api/
RUN echo "=== Verifying files before build ===" && \
    echo "Templates:" && find ./templates -name "*.html" 2>&1 | head -5 && \
    echo "Projects:" && find ./projects -name "*.html" 2>&1 | head -5 && \
    echo "=== Running Tailwind build (verbose) ===" && \
    npx tailwindcss -i ./static/css/input.css -o ./static/css/output.css --minify 2>&1 && \
    echo "=== Checking build result ===" && \
    ls -la /build/static/css/ && \
    if [ -f /build/static/css/output.css ]; then \
        SIZE=$(wc -c < /build/static/css/output.css) && \
        LINES=$(wc -l < /build/static/css/output.css) && \
        echo "File exists, size: $SIZE bytes, lines: $LINES" && \
        if [ "$SIZE" -lt 1000 ]; then \
            echo "WARNING: File is very small, showing full content:" && \
            cat /build/static/css/output.css; \
        else \
            head -30 /build/static/css/output.css; \
        fi; \
    else \
        echo "ERROR: output.css not found!" && exit 1; \
    fi && \
    test -s /build/static/css/output.css || (echo "ERROR: output.css is empty!" && exit 1)

# Stage 2: Python runtime (no Node.js)
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (no Node.js)
# libgomp1 is required for LightGBM (libgomp.so.1)
RUN apt-get update && apt-get install -y \
    postgresql-client \
    netcat-openbsd \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy project first (to access projects/ directory)
COPY . .

# Auto-merge project requirements.txt files into main requirements
RUN echo "" >> requirements.txt && \
    echo "# Auto-detected project dependencies" >> requirements.txt && \
    find projects/ -name "requirements.txt" -type f -exec sh -c 'echo "" >> requirements.txt && echo "# From: {}" >> requirements.txt && cat {} >> requirements.txt' \; || true

# Install all dependencies (main + projects)
RUN pip install --no-cache-dir -r requirements.txt

# Copy built CSS from builder stage
# Copy directly to staticfiles/css/ (WhiteNoise serves from here, avoids volume mount override)
RUN mkdir -p staticfiles/css
COPY --from=tailwind-builder /build/static/css/output.css ./staticfiles/css/output.css
# Also copy to static/css/ (for development, but volume mount may override)
COPY --from=tailwind-builder /build/static/css/output.css ./static/css/output.css
RUN test -s /app/staticfiles/css/output.css || (echo "ERROR: output.css is empty after copy!" && exit 1)

# Expose port
EXPOSE 8000

# Use entrypoint - inline script (avoids Windows line ending issues)
ENTRYPOINT ["sh", "-c", "echo 'Waiting for database...' && while ! nc -z $DB_HOST $DB_PORT; do sleep 0.1; done && echo 'Database ready!' && python manage.py migrate --noinput && mkdir -p staticfiles/css && echo 'Ensuring output.css is available...' && if [ -f /app/staticfiles/css/output.css ] && [ -s /app/staticfiles/css/output.css ]; then echo 'Using built output.css from staticfiles'; elif [ -f /app/static/css/output.css ] && [ -s /app/static/css/output.css ]; then echo 'Copying output.css from static to staticfiles' && cp /app/static/css/output.css /app/staticfiles/css/output.css; else echo 'WARNING: output.css not found or empty!'; fi && python manage.py collectstatic --noinput && exec gunicorn --bind 0.0.0.0:8000 --workers 2 --timeout 120 config.wsgi:application"]

