# syntax=docker/dockerfile:1.2
FROM python:3.10.14

WORKDIR /mle_model

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy APP
COPY . .

# Azure Port
EXPOSE 3100

# Run APP
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:3100", "challenge:app"]