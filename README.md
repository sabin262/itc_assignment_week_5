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

Optional S3 variables for the S3 lease page and S3 API endpoints:

```text
S3_BUCKET_NAME=
S3_PREFIX=sample_leases
AWS_REGION=
```

AWS credentials use the normal boto3 environment/default credential chain.

Optional RAG variables for S3 search and chat:

```text
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=
CHROMA_PERSIST_DIR=./chroma_db
CHROMA_COLLECTION_NAME=lease_chunks
```

`AZURE_OPENAI_EMBEDDING_DEPLOYMENT` is required when using the RAG index, search, or chat endpoints. ChromaDB stores local vector index data in `CHROMA_PERSIST_DIR`; the RAG index also stores structured lease summaries in `lease_summaries.json` in the same directory. Docker Compose mounts `/app/chroma_db` as a named volume so indexed chunks and summaries survive API container restarts.

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

The Streamlit app has a sidebar workspace selector:

- `Local Leases`: tabs for `Summarise` and `Compare` using pasted text or local `.txt`, `.pdf`, or `.docx` uploads
- `S3 Leases`: tabs for `Summarise`, `Compare`, `Index`, `Search`, and `Chat` using leases from the configured S3 bucket/prefix

On the local page, you do not need to both paste text and upload a file. Select one input source per lease. Mixed compare inputs work too, for example Lease A pasted as text and Lease B uploaded as a PDF.

On the S3 page, run `Index S3 Leases` after uploading or changing S3 lease files. `Search` queries all indexed leases. `Chat` can filter to selected S3 leases and keeps the current session's Q&A history in Streamlit session state. Chat uses both structured lease summaries and retrieved chunks, so field-based comparisons such as cheapest rent can use the indexed summaries. The S3 summarise and compare tabs can use either live S3 files or the already indexed lease text stored in ChromaDB.

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

### `GET /s3/leases`

Lists supported `.txt`, `.pdf`, and `.docx` lease files from the configured S3 prefix.

### `POST /summarise-s3`

JSON S3 request:

```json
{
  "key": "sample_leases/valid_lease_a.txt"
}
```

### `POST /compare-s3`

JSON S3 request:

```json
{
  "lease_a_key": "sample_leases/valid_lease_a.txt",
  "lease_b_key": "sample_leases/valid_lease_b.txt"
}
```

### `POST /summarise-indexed`

JSON indexed lease request:

```json
{
  "key": "sample_leases/valid_lease_a.txt"
}
```

Loads the lease text from indexed ChromaDB chunks instead of downloading the file from S3, then reuses the normal structured summary pipeline.

### `POST /compare-indexed`

JSON indexed lease request:

```json
{
  "lease_a_key": "sample_leases/valid_lease_a.txt",
  "lease_b_key": "sample_leases/valid_lease_b.txt"
}
```

Loads both lease texts from indexed ChromaDB chunks instead of downloading the files from S3, then reuses the normal structured comparison pipeline.

### `GET /rag/status`

Returns Chroma collection status, indexed lease count, chunk count, indexed summary count, and the last indexed timestamp.

### `POST /rag/index`

Starts a background indexing job for all supported S3 lease files under the configured bucket/prefix. The request returns quickly with a job status instead of waiting for the full index to complete. Re-indexing clears stale chunks and stale structured summaries for that S3 prefix before inserting fresh data. Each successfully parsed lease is also summarised using the normal structured summary pipeline, including verification and warnings.

### `GET /rag/index/status`

Returns the current indexing job state: `idle`, `running`, `completed`, or `failed`. Running jobs include progress fields such as `progress_current`, `progress_total`, `progress_percent`, `message`, and `current_key`. Completed jobs include the index result counts, skipped files, failed files, summarised lease count, and summary failures.

### `POST /rag/search`

JSON RAG search request:

```json
{
  "question": "Which leases mention pets?",
  "top_k": 5
}
```

### `POST /rag/chat`

JSON RAG chat request:

```json
{
  "question": "When is rent due?",
  "lease_keys": ["sample_leases/valid_lease_a.txt"],
  "history": [
    {"role": "user", "content": "What is the rent?"},
    {"role": "assistant", "content": "The monthly rent is 1,500 pounds."}
  ],
  "top_k": 5
}
```

All successful responses include grounded extraction results, guardrail verification checks, and warnings for unsupported extracted values. The backend runs extraction first, then guardrail verification, and only returns the response after verification is complete. `/compare` summarises and verifies both leases before asking Azure OpenAI for structured differences.

RAG search and chat do not replace the structured summary pipeline. Search uses S3 lease chunks for lookup. Chat uses indexed structured summaries plus retrieved chunks for Q&A, while `/summarise`, `/summarise-text`, `/summarise-s3`, `/compare`, `/compare-text`, and `/compare-s3` keep the existing extraction and comparison flow.

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
.\.venv\Scripts\python.exe -m pytest tests -q
```

Sample lease files are available in `sample_leases/`, including short valid fixtures, long comprehensive fixtures, and dense legal-language fixtures.

Most sample leases are available as `.txt`, `.docx`, and `.pdf` so both text and document-upload paths can be tested.

## LangFuse
Install the Langfuse AI skill from github.com/langfuse/skills and use it to add tracing to this application with Langfuse following best practices.
