# Use Case 1 — Smart Lease Summariser
# Domain: Real Estate / Legal Documents

# Background
A property management company receives hundreds of residential lease agreements every month. Each lease is between 15 and 40 pages long and written in dense legal language. Property managers, landlords, and tenants all struggle to quickly understand what is actually in a document — key dates, obligations, restrictions, and clauses that matter in day-to-day decisions.
The company wants a tool that any staff member can use without legal training. They upload a lease, the tool reads it, and they get back a structured plain-English summary they can act on immediately.

# Problem Statement
Build a GenAI-powered lease summarisation service that accepts a lease document as input and returns a structured extraction in plain English. The service must be containerised and accessible via an API so it can be integrated into the company's internal web portal.

# What the tool must handle
The tool receives a lease document — either as raw pasted text or a text string representing the contents. It must extract and return the following fields:
- Tenant name and landlord name
- Property address
- Lease start date and end date
- Monthly rent amount and payment due date
- Security deposit amount
- Notice period required to vacate
- Key tenant obligations (maintenance, pets, guests, noise)
- Key landlord obligations (repairs, access notice)
- Any unusual or non-standard clauses that stand out
- A one-paragraph plain-English summary of the overall agreement

# Deliverables
- A Python script or notebook that makes an LLM API call to extract the above fields from a lease text, returning structured JSON output
- A prompt that is clearly grounded — it should not infer or fabricate clauses that are not in the document
- A guardrail call that checks whether the extracted clauses are actually present in the original text
- A FastAPI endpoint that accepts a POST request with the lease text and returns the JSON extraction
- A Dockerfile that containerises the API so it runs identically on any machine
- A .env file pattern for managing the API key and endpoint — the key must not be hardcoded in any script

# Constraints
The API must return a 422 response with a clear error message if the input text is too short to be a real lease (under 100 words)
The extraction prompt must be written so that missing fields return null rather than guessing
Temperature must be justified — explain your choice in a comment

# Stretch goal
Add a second endpoint POST /compare that accepts two lease texts and returns a comparison highlighting what differs between them — useful when a tenant is choosing between two properties.