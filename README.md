# ML CI Demo

FastAPI + Scikit-learn + Joblib + Docker + GitHub Actions.

## Local Test

```bash
pip install -r requirements-dev.txt
pytest -v
```

## Local Docker

```bash
docker build -t ml-ci-api:local .
docker run -d --name ml-ci-api -p 8000:8000 ml-ci-api:local
```

Open:

http://127.0.0.1:8000/docs
