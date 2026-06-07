# Smart Lease Summariser

Grounded residential lease extraction and comparison system using Azure OpenAI.

The project has two containerised services:

- FastAPI backend for text and document-based lease summarisation/comparison
- Streamlit frontend for pasting lease text or uploading `.txt`, `.pdf`, or `.docx` lease files

PDF support is for text-based PDFs. Scanned/image-only PDFs need OCR before this app can summarise them.

## Project Structure

```text
app/                         FastAPI backend, Azure OpenAI client, schemas, prompts, document parser
docs/                        Editable project diagrams
frontend/                    Streamlit frontend package and Dockerfile
sample_leases/               Test lease inputs and saved API responses
scripts/                     Utility and sample-generation scripts
tests/                       Mocked API and parser tests
docker-compose.yml           Runs backend and frontend together
```

## Environment

Create a `.env` file and fill in your Azure OpenAI values:

Required variables:

```text
AZURE_OPENAI_API_KEY=
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_API_VERSION=
AZURE_OPENAI_DEPLOYMENT=
```

Do not commit `.env`; it is ignored by `.gitignore` and `.dockerignore`.

## Run With Docker Compose

Build and start both services:

```powershell
docker compose up -d --build
```

Open:

- Streamlit frontend: `http://localhost:8501`
- FastAPI docs: `http://localhost:8000/docs`
- API health check: `http://localhost:8000/health`

Check service status:

```powershell
docker compose ps
```

Stop services:

```powershell
docker compose down
```

## Dockerfiles

The services use separate Dockerfiles:

- `app/Dockerfile` builds the FastAPI API image
- `frontend/Dockerfile` builds the Streamlit frontend image

Build either image directly:

```powershell
docker build -f app/Dockerfile -t smart-lease-summariser-api .
docker build -f frontend/Dockerfile -t smart-lease-summariser-frontend .
```

For normal use, prefer Docker Compose because it wires the frontend to the API service automatically.

## Frontend

The Streamlit app supports both workflows:

- `Summarise`: provide one lease using pasted text or a `.txt`, `.pdf`, or `.docx` upload
- `Compare`: provide Lease A and Lease B, each using pasted text or a `.txt`, `.pdf`, or `.docx` upload

You do not need to both paste text and upload a file. Select one input source per lease. Mixed compare inputs work too, for example Lease A pasted as text and Lease B uploaded as a PDF.

Guardrail results are shown before the extracted summary, lease details, or comparison table. If Azure OpenAI flags unsupported extracted values, the frontend displays those warnings first so you can review grounding before relying on the rest of the response.

## API Endpoints

### `POST /summarise`

Multipart document upload request:

```text
file=<lease .txt, .pdf, or .docx>
```

### `POST /summarise-text`

JSON text request:

```json
{
  "lease_text": "Full lease text of at least 100 words..."
}
```

### `POST /compare`

Multipart document upload request:

```text
lease_a=<first lease .txt, .pdf, or .docx>
lease_b=<second lease .txt, .pdf, or .docx>
```

### `POST /compare-text`

JSON text request:

```json
{
  "lease_a": "First full lease text of at least 100 words...",
  "lease_b": "Second full lease text of at least 100 words..."
}
```

All successful responses include grounded extraction results, guardrail verification checks, and warnings for unsupported extracted values. The backend runs extraction first, then guardrail verification, and only returns the response after verification is complete. `/compare` summarises and verifies both leases before asking Azure OpenAI for structured differences.

Both summarisation and comparison reject lease text under 100 words with HTTP `422`. Unsupported file types and files with no extractable text also return HTTP `422`.

## Local Development

Install all development dependencies from the root requirements file:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run the API locally:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

Run the frontend locally in a second terminal:

```powershell
$env:API_BASE_URL = "http://localhost:8000"
.\.venv\Scripts\streamlit.exe run frontend/streamlit_app.py
```

## Testing

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_app.py -q
```
```

Sample lease files are available in `sample_leases/`, including short valid fixtures, long comprehensive fixtures, and dense legal-language fixtures.

Most sample leases are available as `.txt`, `.docx`, and `.pdf` so both text and document-upload paths can be tested.

