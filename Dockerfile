# Use Python official image as base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1


# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends tzdata \
    curl \
    vim \
    && ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime \
    && echo "Asia/Shanghai" > /etc/timezone \
    postgresql-client \
    build-essential \
    libpq-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV TZ=Asia/Shanghai

# Copy project files
COPY . /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN python train_model.py

# Expose Streamlit default port
EXPOSE 8501

# Set health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Set startup command
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"] 
