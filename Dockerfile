# Use a lightweight Python 3.10 base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Make the project root available for Python imports
ENV PYTHONPATH=/app

# Prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1

# Show Python output immediately in Docker logs
ENV PYTHONUNBUFFERED=1

# Copy requirements first so Docker can cache dependency installation
COPY requirements.txt .

# Install project dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code, processed data, models, and other project files
COPY . .

# Streamlit listens on port 8501
EXPOSE 8501

# Start the Streamlit dashboard
CMD ["python", "-m", "streamlit", "run", "app/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]