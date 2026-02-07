# AI Document Extraction Service

A **schema-driven** universal document extractor using **multimodal LLMs (no OCR)**. This service accepts documents (file upload or URL) and a JSON schema, then returns structured JSON data extracted directly from the document following the schema exactly.

## Features

- ✅ **Schema-driven extraction** - Define exact fields and types via JSON schema
- ✅ Direct document → LLM (no OCR required)
- ✅ File upload or URL-based document processing
- ✅ Support for images (PNG, JPG, JPEG, GIF, BMP, TIFF, WEBP) and PDFs
- ✅ Multi-page PDF support (first page used for multimodal extraction)
- ✅ **Strict validation** - Ensures output matches schema exactly
- ✅ **Automatic retry** - Retries once if validation fails
- ✅ OpenAI Responses API (gpt-4.1-mini) with Gemini fallback
- ✅ Multimodal extraction (image/PDF sent directly to LLM)
- ✅ Comprehensive error handling and logging
- ✅ Temperature = 0 for consistent results

## Prerequisites

- Python 3.8+
- pip
- For PDF processing: `poppler` (required by pdf2image)
  - macOS: `brew install poppler`
  - Ubuntu/Debian: `sudo apt-get install poppler-utils`
  - Windows: Download from [poppler releases](https://github.com/oschwartz10612/poppler-windows/releases)

## Installation

1. **Navigate to the project directory:**
   ```bash
   cd "/Users/akashngowda/Desktop/kelsa/AI pipeline"
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   
   Create a `.env` file in the project root:
   ```bash
   # .env file
   OPENAI_API_KEY=your_openai_api_key_here   # Primary multimodal extraction
   GEMINI_API_KEY=your_gemini_api_key_here   # Optional fallback
   ```
   
   Note: OPENAI_API_KEY is recommended (gpt-4.1-mini). Gemini is optional fallback.

## Running the Server

Start the FastAPI server:

```bash
python main.py
```

Or using uvicorn directly:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The server will start at `http://localhost:8000`

## API Documentation

Once the server is running, visit:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

## Endpoints

### 1. Health Check

**GET** `/`

Returns server status.

**Response:**
```json
{
  "status": "running"
}
```

### 2. Extract from File or URL

**POST** `/extract`

Accepts a document (file upload or URL) and a JSON schema, returns extracted structured data.

**Request (multipart/form-data):**
- `file` (optional): Uploaded file
- `file_url` (optional): URL of the document
- `schema` (required): JSON string describing fields to extract

**Note:** Either `file` OR `file_url` must be provided.

**Example Schema:**
```json
{
  "customer_name": "string",
  "gst_number": "string",
  "invoice_amount": "number",
  "invoice_date": "string",
  "is_paid": "boolean"
}
```

**Supported Types:**
- `"string"` - Text values
- `"number"` - Numeric values (int or float)
- `"integer"` - Integer values
- `"boolean"` - True/false values
- `"array"` - List values
- `"object"` - Nested objects

**Example Request (using curl):**
```bash
# With file upload
curl -X POST "http://localhost:8000/extract" \
  -F "file=@invoice.pdf" \
  -F 'schema={"customer_name":"string","gst_number":"string","invoice_amount":"number"}'

# With file URL
curl -X POST "http://localhost:8000/extract" \
  -F "file_url=https://example.com/document.pdf" \
  -F 'schema={"customer_name":"string","gst_number":"string","invoice_amount":"number"}'
```

**Example Response:**
```json
{
  "success": true,
  "data": {
    "customer_name": "ABC Corporation",
    "gst_number": "29ABCDE1234F1Z5",
    "invoice_amount": 50000.00
  },
  "filename": "invoice.pdf",
  "error": null
}
```

## Testing with Swagger UI

1. Start the server
2. Navigate to http://localhost:8000/docs
3. Click on `/extract` endpoint
4. Click "Try it out"
5. Fill in the form:
   - Upload a file OR provide a `file_url`
   - Enter your `schema` as a JSON string (e.g., `{"customer_name":"string","gst_number":"string"}`)
6. Click "Execute"
7. View the response

## Architecture

### Processing Pipeline

```
Input API (File / URL)
    ↓
Load Document (image / PDF)
    ↓
Multimodal LLM (document + schema)
    ↓
Validation Layer
    ↓
Retry once if invalid
    ↓
Structured JSON
```

### Key Components

1. **Input Validation:** Checks for file or URL, validates schema JSON
2. **File Acquisition:** Downloads from URL or uses uploaded file
3. **Document Loading:** Converts PDF/image for multimodal LLM input
4. **Schema Builder:** Converts user schema to strict extraction prompt
5. **LLM Extraction:**
   - Sends document + schema to OpenAI gpt-4.1-mini (multimodal)
   - Uses strict system prompt enforcing schema compliance
   - Temperature = 0 for consistent results
6. **Validation Layer:**
   - Ensures all schema keys exist
   - Removes extra fields
   - Fills missing keys with null
   - Validates basic types
   - Retries once if invalid
7. **Response Formatting:** Returns validated JSON matching schema exactly

### Strict System Prompt

The LLM uses this exact prompt:

```
You are a document extraction engine.

Rules:
- Extract ONLY from provided text.
- Do NOT infer or guess.
- Output VALID JSON only.
- Follow schema EXACTLY.
- Missing values must be null.
- No extra keys allowed.
```

## Validation & Retry Logic

The service includes a robust validation layer:

1. **Schema Compliance Check:**
   - Verifies all schema keys are present
   - Removes any extra fields not in schema
   - Fills missing keys with `null`

2. **Type Validation:**
   - Validates that extracted values match expected types
   - Converts or sets to `null` if type mismatch

3. **Automatic Retry:**
   - If validation fails, retries LLM extraction once
   - Uses enhanced prompt: "Fix output to exactly match schema. Return JSON only."
   - If retry also fails, uses fixed data (missing fields set to null)

## Error Handling

The service includes comprehensive error handling:
- Invalid file formats
- Missing required parameters
- Invalid JSON schema
- API failures
- JSON parsing errors
- Network errors (for URL downloads)
- Validation failures (with automatic retry)

All errors return readable JSON responses with appropriate HTTP status codes.

## Configuration

- **Max file size:** 10MB (configurable in `main.py`)
- **Supported formats:** PNG, JPG, JPEG, GIF, BMP, TIFF, WEBP, PDF
- **LLM model:** OpenAI gpt-4.1-mini
- **LLM temperature:** 0 (for consistent extraction)
- **Retry attempts:** 1 (configurable in code)

## Example Schemas

### Invoice Extraction
```json
{
  "customer_name": "string",
  "invoice_number": "string",
  "invoice_date": "string",
  "invoice_amount": "number",
  "gst_number": "string",
  "due_date": "string"
}
```

### Receipt Extraction
```json
{
  "merchant_name": "string",
  "transaction_date": "string",
  "total_amount": "number",
  "payment_method": "string",
  "items": "array"
}
```

### Form Extraction
```json
{
  "applicant_name": "string",
  "email": "string",
  "phone": "string",
  "date_of_birth": "string",
  "address": "string"
}
```

## Future Enhancements

- [ ] Multi-page multimodal PDF support
- [ ] CRM field mapper integration
- [ ] Original document + extracted preview UI
- [ ] Batch processing endpoint
- [ ] Confidence scoring
- [ ] Authentication / API keys
- [ ] Rate limiting

## Troubleshooting

### PDF processing fails
- Ensure `poppler` is installed (see Prerequisites)

### LLM API errors
- Verify API keys are set in `.env` file
- Check API key validity and quota
- Review logs for detailed error messages

### Validation failures
- Check that schema JSON is valid
- Ensure field names match expected format
- Review logs for validation details

### Import errors
- Ensure virtual environment is activated
- Run `pip install -r requirements.txt` again

## License

This is a POC (Proof of Concept) - not production-ready code.
