"""
AI-Based Document Extraction Service
Schema-driven universal document extractor using Multimodal LLM (no OCR)
"""

import os
import io
import base64
import json
import logging
import tempfile
import re
from typing import Optional, Dict, Any, Tuple
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header
from fastapi.responses import HTMLResponse
from pathlib import Path

import httpx
from pydantic import BaseModel
from dotenv import load_dotenv
from PIL import Image
import pdf2image
import fitz  # PyMuPDF
import google.generativeai as genai
from openai import OpenAI

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Document Extraction Service",
    description="Schema-driven universal document extractor",
    version="2.0.0",
    openapi_tags=[
        {
            "name": "extraction",
            "description": "Document extraction endpoints"
        }
    ]
)

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
SUPPORTED_IMAGE_FORMATS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp"}
SUPPORTED_PDF_FORMAT = ".pdf"

# --- API Security ---
API_TOKEN = os.getenv("API_TOKEN")
if not API_TOKEN:
    logger.error("API_TOKEN environment variable is not set. Please define it in your .env file.")
    raise RuntimeError("API_TOKEN environment variable must be set")

def verify_token(authorization: str = Header(None)):
    # Enforce strict token validation

    if authorization is None:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid Authorization format")

    parts = authorization.split(" ")
    if len(parts) != 2 or not parts[1]:
        raise HTTPException(status_code=401, detail="Invalid Authorization format")

    token = parts[1]

    if token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid API token")

# Initialize LLM clients
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
    logger.info("Gemini API configured")
else:
    gemini_model = None
    logger.warning("GEMINI_API_KEY not found, will use OpenAI fallback")

if OPENAI_API_KEY:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    logger.info("OpenAI API configured")
else:
    openai_client = None
    logger.warning("OPENAI_API_KEY not found")



class ExtractionResponse(BaseModel):
    """Response model for extraction"""
    success: bool
    data: Dict[str, Any]
    filename: Optional[str] = None
    error: Optional[str] = None


# New Pydantic model for JSON-based extract endpoint
class ExtractRequest(BaseModel):
    file_url: str
    schema: Dict[str, str]


def download_file_from_url(url: str) -> bytes:
    """Download file from URL with proper headers and validation"""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/pdf"
        }

        with httpx.Client(timeout=30.0, follow_redirects=True) as client:
            response = client.get(url, headers=headers)
            response.raise_for_status()

            content = response.content

            # Validate it's actually a PDF
            if not content.startswith(b"%PDF"):
                raise HTTPException(
                    status_code=400,
                    detail="URL did not return a valid PDF file (likely HTML page instead of direct PDF)"
                )

            if len(content) > MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=400,
                    detail=f"File size exceeds maximum allowed size of {MAX_FILE_SIZE / 1024 / 1024}MB"
                )

            return content

    except httpx.HTTPError as e:
        logger.error(f"Error downloading file from URL: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to download file from URL: {str(e)}")



# --- Multimodal LLM document loader ---
def load_document_as_images(file_bytes: bytes, filename: str) -> list[Image.Image]:
    """
    Load document into image(s) for multimodal LLM.
    Images are sent directly to LLM (no OCR).
    """
    file_ext = Path(filename).suffix.lower()

    if file_ext == ".pdf":
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(file_bytes)
                tmp_path = tmp_file.name

            images = pdf2image.convert_from_path(tmp_path, dpi=200)
            os.unlink(tmp_path)
            return images
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load PDF: {str(e)}")

    elif file_ext in SUPPORTED_IMAGE_FORMATS:
        try:
            return [Image.open(io.BytesIO(file_bytes))]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load image: {str(e)}")

    else:
        raise HTTPException(status_code=400, detail=f"Unsupported file format: {file_ext}")


# --- PDF text extraction helper ---
def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """Extract text from PDF using PyMuPDF"""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text_parts = []
        for page in doc:
            text_parts.append(page.get_text() or "")
        return "\n".join(text_parts).strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to extract text from PDF: {str(e)}")


def build_schema_prompt(schema: Dict[str, str]) -> str:
    """
    Schema Builder - Convert user schema to LLM-readable format
    """
    schema_description = "Extract the following fields with their types:\n"
    for field_name, field_type in schema.items():
        schema_description += f"- {field_name}: {field_type}\n"
    return schema_description



# --- Multimodal LLM extraction ---
from typing import Optional

# --- Helper: Clean LLM JSON output ---
def clean_llm_json(text: str) -> str:
    import re
    if not text:
        return text
    t = text.strip()
    # remove markdown fences
    t = re.sub(r"```json", "", t, flags=re.IGNORECASE)
    t = re.sub(r"```", "", t)
    # extract first JSON object
    m = re.search(r"\{.*\}", t, re.DOTALL)
    if m:
        t = m.group(0)
    # remove trailing commas
    t = re.sub(r",\s*}", "}", t)
    t = re.sub(r",\s*]", "]", t)
    return t.strip()

def call_llm_for_extraction(
    images: Optional[list[Image.Image]] = None,
    text: Optional[str] = None,
    schema: Dict[str, str] = None,
    is_retry: bool = False
) -> Dict[str, Any]:

    system_instruction = (
        "You are a document extraction engine.\n\n"
        "Rules:\n"
        "- Extract ONLY from provided document.\n"
        "- Extract from visible text OR clearly identifiable context (logo, header, branding).\n"
        "- For certificates/documents, organization is usually present in header, logo, or issuing authority section. Extract that as organization.\n"
        "- Prefer official organization names (e.g., Infosys, Google, TCS) if clearly visible.\n"
        "- Do NOT hallucinate beyond document context.\n"
        "- Output VALID JSON only.\n"
        "- Follow schema EXACTLY.\n"
        "- Missing values must be null.\n"
        "- No extra keys allowed."
    )

    if is_retry:
        system_instruction += "\n\nFix output to exactly match schema. Return JSON only."

    schema_prompt = json.dumps(schema, indent=2)

    if text:
        user_message = f"""
Schema:
{schema_prompt}

Document Text:
{text}

Instructions:
- Extract all fields strictly.
- If organization is not explicitly labeled, infer it from header/logo text.

Return ONLY valid JSON.
"""
    else:
        user_message = f"""
Schema:
{schema_prompt}

Extract fields from the attached document.
Return ONLY valid JSON.
"""

    # Gemini (primary)
    if gemini_model:
        try:
            if text:
                content = [f"{system_instruction}\n\n{user_message}"]
            else:
                content = [f"{system_instruction}\n\n{user_message}"] + images

            response = gemini_model.generate_content(
                content,
                generation_config={
                    "temperature": 0,
                    "max_output_tokens": 2048,
                }
            )

            result_text = response.text.strip()
            result_text = clean_llm_json(result_text)
            print("GEMINI CLEANED:", result_text)
            return json.loads(result_text)

        except Exception as e:
            logger.warning(f"Gemini failed: {e}, trying OpenAI fallback")

    # OpenAI fallback (multimodal via Responses API)
    if openai_client:
        try:
            if text:
                response = openai_client.responses.create(
                    model="gpt-4.1-mini",
                    input=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": user_message}
                    ]
                )
                result_text = response.output_text.strip()
                result_text = clean_llm_json(result_text)
                print("OPENAI TEXT CLEANED:", result_text)
                return json.loads(result_text)

            # otherwise image flow continues below
            image_inputs = []

            for img in images:
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                img_b64 = base64.b64encode(buf.getvalue()).decode()

                image_inputs.append({
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{img_b64}"
                })

            response = openai_client.responses.create(
                model="gpt-4.1-mini",
                input=[
                    {
                        "role": "system",
                        "content": system_instruction
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": user_message},
                            *image_inputs
                        ]
                    }
                ]
            )

            result_text = response.output_text.strip()
            result_text = clean_llm_json(result_text)
            print("OPENAI IMAGE CLEANED:", result_text)
            return json.loads(result_text)

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"LLM extraction failed: {str(e)}")

    raise HTTPException(status_code=500, detail="No LLM API key configured")


def validate_type(value: Any, expected_type: str) -> bool:
    """
    Basic type validation
    """
    if value is None:
        return True
    
    expected_type = expected_type.lower()
    
    if expected_type == "string":
        return isinstance(value, str)
    elif expected_type == "number":
        return isinstance(value, (int, float))
    elif expected_type == "integer":
        return isinstance(value, int)
    elif expected_type == "boolean":
        return isinstance(value, bool)
    elif expected_type == "array":
        return isinstance(value, list)
    elif expected_type == "object":
        return isinstance(value, dict)
    else:
        # Default: accept any type
        return True


def validate_and_fix_output(extracted_data: Dict[str, Any], schema: Dict[str, str]) -> Dict[str, Any]:
    """
    Validation Layer
    
    1. Ensure all schema keys exist
    2. Remove extra fields
    3. Fill missing keys with null
    4. Validate types (basic)
    
    Returns validated and fixed data
    """
    validated_data = {}
    
    # Process each field in schema
    for field_name, field_type in schema.items():
        if field_name in extracted_data:
            value = extracted_data[field_name]
            # Validate type
            if validate_type(value, field_type):
                validated_data[field_name] = value
            else:
                # Type mismatch - set to null
                logger.warning(f"Type mismatch for {field_name}: expected {field_type}, got {type(value).__name__}")
                validated_data[field_name] = None
        else:
            # Missing field - set to null
            validated_data[field_name] = None
    
    # Remove any extra fields not in schema
    # (already done by only processing schema keys)
    
    return validated_data


def validate_output(extracted_data: Dict[str, Any], schema: Dict[str, str]) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate output against schema
    
    Returns:
        (is_valid, fixed_data)
    """
    # Check if all required fields are present (even if null)
    missing_fields = set(schema.keys()) - set(extracted_data.keys())
    if missing_fields:
        logger.warning(f"Missing fields: {missing_fields}")
        return False, validate_and_fix_output(extracted_data, schema)
    
    # Check for extra fields
    extra_fields = set(extracted_data.keys()) - set(schema.keys())
    if extra_fields:
        logger.warning(f"Extra fields found: {extra_fields}")
        return False, validate_and_fix_output(extracted_data, schema)
    
    # Validate types
    type_errors = []
    for field_name, field_type in schema.items():
        value = extracted_data.get(field_name)
        if value is not None and not validate_type(value, field_type):
            type_errors.append(f"{field_name}: expected {field_type}, got {type(value).__name__}")
    
    if type_errors:
        logger.warning(f"Type errors: {type_errors}")
        return False, validate_and_fix_output(extracted_data, schema)
    
    # All validations passed
    return True, extracted_data



@app.get("/", response_class=HTMLResponse)
async def web_ui():
    return """
    <html>
    <head>
        <title>AI Document Extraction</title>
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, sans-serif;
                background: #f5f7fb;
                margin: 0;
                padding: 40px;
            }
            .card {
                background: white;
                padding: 30px;
                border-radius: 12px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.08);
                max-width: 900px;
                margin: auto;
            }
            h2 {
                margin-top: 0;
            }
            input[type=file], textarea {
                width: 100%;
                padding: 12px;
                border-radius: 8px;
                border: 1px solid #ddd;
                margin-top: 10px;
                margin-bottom: 20px;
                font-size: 14px;
            }
            button {
                background: #2563eb;
                color: white;
                padding: 12px 20px;
                border: none;
                border-radius: 8px;
                cursor: pointer;
                font-size: 14px;
            }
            button:hover {
                background: #1d4ed8;
            }
        </style>
    </head>
    <body>
        <div class="card">
            <h2> AI Document Extraction Demo</h2>
            <form action="/preview-extract" method="post" enctype="multipart/form-data">
                <label><b>Upload Document (PDF/Image)</b></label>
                <input type="file" name="file" required>

                <label><b>Schema (JSON format)</b></label>
                <textarea name="schema" placeholder='{"customer_name": "string", "invoice_amount": "number"}' required></textarea>

                <button type="submit">Extract & Preview</button>
            </form>
        </div>
    </body>
    </html>
    """

# --- Preview extraction web UI endpoint ---


@app.post("/preview-extract", response_class=HTMLResponse)
async def preview_extract(
    file: UploadFile = File(...),
    schema: str = Form(...),
    authorization: Optional[str] = Header(None)
):
    try:
        verify_token(authorization)
        schema_dict = json.loads(schema)
        file_bytes = await file.read()
        filename = file.filename or "uploaded_file"

        file_ext = Path(filename).suffix.lower()

        if file_ext == ".pdf":
            text = extract_text_from_pdf_bytes(file_bytes)
            if not text or len(text.strip()) < 500:
                images = load_document_as_images(file_bytes, filename)
                text = None
            else:
                images = None
        else:
            images = load_document_as_images(file_bytes, filename)
            text = None

        extracted_data = call_llm_for_extraction(images=images, text=text, schema=schema_dict)
        _, validated_data = validate_output(extracted_data, schema_dict)

        # Preview generation
        if images:
            buf = io.BytesIO()
            images[0].save(buf, format="PNG")
            img_b64 = base64.b64encode(buf.getvalue()).decode()
            preview_html = f'<img src="data:image/png;base64,{img_b64}" />'
        else:
            pdf_b64 = base64.b64encode(file_bytes).decode()
            preview_html = f'''
            <iframe 
                src="data:application/pdf;base64,{pdf_b64}" 
                width="100%" 
                height="600px" 
                style="border:1px solid #ddd; border-radius:8px;">
            </iframe>
            '''

        return f"""
        <html>
        <head>
            <style>
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, sans-serif;
                    background: #f5f7fb;
                    margin: 0;
                    padding: 40px;
                }}
                .container {{
                    display: flex;
                    gap: 30px;
                }}
                .card {{
                    background: white;
                    padding: 20px;
                    border-radius: 12px;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.08);
                }}
                .left {{ width: 40%; }}
                .right {{ width: 60%; }}
                pre {{
                    background: #0f172a;
                    color: #f8fafc;
                    padding: 15px;
                    border-radius: 8px;
                    overflow-x: auto;
                }}
                img {{
                    width: 100%;
                    border-radius: 8px;
                    border: 1px solid #ddd;
                }}
                .header {{
                    margin-bottom: 30px;
                }}
                .btn {{
                    margin-top: 20px;
                    display: inline-block;
                    padding: 10px 16px;
                    background: #2563eb;
                    color: white;
                    text-decoration: none;
                    border-radius: 8px;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h2> Extraction Result</h2>
            </div>

            <div class="container">
                <div class="card left">
                    <h3>Extracted JSON</h3>
                    <pre>{json.dumps(validated_data, indent=2)}</pre>
                </div>

                <div class="card right">
                    <h3>Document Preview</h3>
                    {preview_html}
                </div>
            </div>

            <a href="/" class="btn">Upload Another Document</a>
        </body>
        </html>
        """

    except Exception as e:
        return f"<h3>Error: {str(e)}</h3><br><a href='/'>Go Back</a>"


# New JSON-based /extract endpoint
@app.post("/extract", response_model=ExtractionResponse)
async def extract_from_json(
    request: ExtractRequest,
    authorization: Optional[str] = Header(None)
):
    try:
        verify_token(authorization)

        if not request.file_url:
            raise HTTPException(status_code=400, detail="file_url is required")

        schema_dict = request.schema

        logger.info(f"Downloading file from URL: {request.file_url}")
        file_bytes = download_file_from_url(request.file_url)
        filename = request.file_url.split("/")[-1] or "downloaded_file"

        logger.info("Loading document for multimodal LLM...")
        file_ext = Path(filename).suffix.lower()

        if file_ext == ".pdf":
            text = extract_text_from_pdf_bytes(file_bytes)
            if not text or len(text.strip()) < 500:
                images = load_document_as_images(file_bytes, filename)
                text = None
            else:
                images = None
        else:
            images = load_document_as_images(file_bytes, filename)
            text = None

        logger.info("Calling LLM for structured extraction...")
        extracted_data = call_llm_for_extraction(images=images, text=text, schema=schema_dict, is_retry=False)

        logger.info("Validating extracted data...")
        is_valid, validated_data = validate_output(extracted_data, schema_dict)

        if not is_valid:
            logger.warning("Initial extraction failed validation, retrying...")
            extracted_data_retry = call_llm_for_extraction(images=images, text=text, schema=schema_dict, is_retry=True)
            is_valid_retry, validated_data_retry = validate_output(extracted_data_retry, schema_dict)

            if is_valid_retry:
                validated_data = validated_data_retry
                logger.info("Retry succeeded")
            else:
                logger.warning("Retry also failed validation, using fixed data")
                validated_data = validate_and_fix_output(extracted_data_retry, schema_dict)

        return ExtractionResponse(
            success=True,
            data=validated_data,
            filename=filename
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in extraction: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
