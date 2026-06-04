# Dockerized Streamlit and FastAPI Starter

This project runs a FastAPI backend and a Streamlit frontend in separate Docker containers.

## Services

- FastAPI: <http://localhost:8000>
- Streamlit: <http://localhost:8501>
- FastAPI docs: <http://localhost:8000/docs>

## FastAPI Endpoints

- `GET /health` - service health and available endpoint status
- `GET /summarize` - summarize endpoint status
- `POST /summarize` - returns a simple generated summary for submitted text
- `GET /compare` - compare endpoint status
- `POST /compare` - compares two submitted text values

## Run

```powershell
docker compose up --build
```

Open Streamlit at <http://localhost:8501>. Each Streamlit page checks and displays the matching FastAPI endpoint status.

## Stop

```powershell
docker compose down
```
