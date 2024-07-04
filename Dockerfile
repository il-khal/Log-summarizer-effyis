FROM python:3.10-slim

WORKDIR /

RUN pip install --no-cache-dir runpod

COPY requirements.txt .

RUN apt-get update && apt-get install -y git
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python3", "-u", "rp_handler.py"]
