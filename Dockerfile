FROM python:3.10-slim-bullseye

RUN pip install pipenv

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["flask_serv.py", "./"]
COPY ["model.bin", "./"]
COPY ["dv.bin", "./"]

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "flask_serv:app"]