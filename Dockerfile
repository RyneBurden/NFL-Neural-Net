FROM python-3.10-buster as model
WORKDIR /staley
ENV TZ "America/New_York"
COPY .\training_data.csv .
COPY .\requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt
ENTRYPOINT [ "echo", "built!" ]

FROM postgres:14.3-bullseye as database
WORKDIR /staley
ENV TZ "America/New_York"
ENV POSTGRES_USER "staley"
ENV POSTGRES_PASSWORD "staley_but_password"
ENV POSTGRES_DB "staley"
COPY .\init.sql .

