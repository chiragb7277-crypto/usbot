FROM python:3.11-slim-bookworm

# Prevent Python from buffering stdout/stderr and generating .pyc files
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=7860

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

# Set up non-root user for Hugging Face Spaces (UID 1000)
RUN useradd -m -u 1000 user
WORKDIR /app

# Copy requirements.txt and install python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files into the container
COPY --chown=user:user . .

# Create necessary runtime directories and set permissions
RUN mkdir -p /app/fonts /app/db /app/workspace /app/resources/emojis /app/scratch \
    && chown -R user:user /app \
    && chmod -R 777 /app

USER user

# Expose ports for Hugging Face (7860) and other cloud hosts
EXPOSE 7860 10000 8000 8080

# Run the textbox_bot with unbuffered output so logs stream immediately
CMD ["python", "-u", "textbox_bot.py"]

