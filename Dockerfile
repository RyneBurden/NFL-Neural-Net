FROM python:3.10-buster as model
WORKDIR /staley
ENV TZ "America/New_York"
COPY ./requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt
ENTRYPOINT [ "echo", "built!" ]

FROM postgres:14-bullseye as staley_database
WORKDIR /staley
ENV TZ "America/New_York"
ENV POSTGRES_USER "staley"
ENV POSTGRES_PASSWORD "staley_but_password"
ENV POSTGRES_DB "staley"
COPY ./init.sql .
COPY ./data/teams_logos_colors.csv .
RUN mv ./init.sql /docker-entrypoint-initdb.d/init.sql

FROM python:3.10-bullseye as api_server
WORKDIR /staley
ENV TZ "America/New_York"
ENV POSTGRES_USER "staley"
ENV POSTGRES_PASSWORD "staley_but_password"
ENV POSTGRES_DB "staley"
COPY ./api/api_requirements.txt .