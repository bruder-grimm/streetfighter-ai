FROM python:3-alpine

RUN apk add --update alpine-sdk && apk update && apk add --virtual build-dependencies build-base gcc wget git

COPY requirements.txt /tmp/

RUN pip install -r /tmp/requirements.txt

COPY . /tmp/

WORKDIR /app
