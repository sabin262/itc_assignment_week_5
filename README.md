# Dockerized Streamlit and FastAPI Starter

This project runs a FastAPI backend and a Streamlit frontend in separate Docker containers.

## Services

- FastAPI: <http://localhost:8000>
- Streamlit: <http://localhost:8501>
- FastAPI docs: <http://localhost:8000/docs>

## FastAPI Endpoints

- `GET /health` - service health and available endpoint status
- `GET /summarise` - summarise endpoint status
- `POST /summarise` - returns a simple generated summary for submitted text or an uploaded file
- `GET /compare` - compare endpoint status
- `POST /compare` - compares two submitted text values or two uploaded files

The upload endpoints accept `.txt`, `.pdf`, and `.docx` files. Streamlit sends uploaded files to FastAPI and displays the extracted contents in scrollable text fields.

## Run

```powershell
docker compose up --build
```

Open Streamlit at <http://localhost:8501>. Each Streamlit page checks and displays the matching FastAPI endpoint status.

## Stop

```powershell
docker compose down
```
