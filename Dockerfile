FROM python:3.12-slim

WORKDIR /app

COPY . /app

RUN apt-get -y update && apt-get -y install \
    gcc \
    python3-dev

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 8000

CMD [ "python", "main.py" ]