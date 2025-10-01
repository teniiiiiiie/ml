FROM python:3.9-slim

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

RUN pip install boto3

COPY init_storage.py /app/init_storage.py

WORKDIR /app

CMD ["python", "init_storage.py"]
