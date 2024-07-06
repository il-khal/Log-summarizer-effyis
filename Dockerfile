FROM python:3.10-slim

WORKDIR /

ENV CC=/usr/bin/gcc

# Install build essentials to ensure the compiler is available
RUN apt-get update && \
    apt-get install -y build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir runpod

COPY requirements.txt .

RUN apt-get update && apt-get install -y git
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python3", "-u", "rp_handler.py"]
