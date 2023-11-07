FROM python:3.10.13-slim

RUN pip install pipenv


WORKDIR /app

COPY ["pipfile", "pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["predict.py", "model_C=10.bin", "./"]

EXPOSE 9696

ENTRYPOINT ["waitress-serve", "--listen=0.0.0.0:9696", "predict:app"]