# syntax=docker/dockerfile:1

FROM ubuntu:22.04

WORKDIR /app

# Update the package manager and install Python3 and pip3
RUN apt-get update && apt-get install -y python3 python3-pip

ENV TZ=GMT
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get install libpq-dev python-dev-is-python3 -y
RUN apt-get install postgresql postgresql-contrib -y

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

CMD [ "gunicorn", "app:app"]