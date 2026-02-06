"""
AI-Based Document Extraction Service
Schema-driven universal document extractor using OCR + LLM
"""

import os
import io
import json
import logging
import tempfile
import re
from typing import Optional, Dict, Any, Tuple
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pathlib import Path

import httpx
from pydantic import BaseModel
from dotenv import load_dotenv
from PIL import Image
import pdf2image
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


def download_file_from_url(url: str) -> bytes:
    """Download file from URL"""
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.get(url)
            response.raise_for_status()
            if len(response.content) > MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=400,
                    detail=f"File size exceeds maximum allowed size of {MAX_FILE_SIZE / 1024 / 1024}MB"
                )
            return response.content
    except httpx.HTTPError as e:
        logger.error(f"Error downloading file from URL: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to download file from URL: {str(e)}")


def convert_pdf_to_images(pdf_bytes: bytes) -> list[Image.Image]:
    """Convert PDF bytes to list of PIL Images"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_bytes)
            tmp_path = tmp_file.name
        
        images = pdf2image.convert_from_path(tmp_path, dpi=200)
        os.unlink(tmp_path)  # Clean up temp file
        logger.info(f"Converted PDF to {len(images)} images")
        return images
    except Exception as e:
        logger.error(f"Error converting PDF to images: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")


def mock_ocr_from_image(image: Image.Image) -> str:
    """
    Mock OCR - returns placeholder text
    TODO: Replace with Google Vision OCR or Tesseract
    """
    # For POC, return a mock text
    # In production, use: pytesseract.image_to_string(image) or Google Vision API
    logger.info("Using mock OCR (placeholder)")
    return f"[MOCK OCR TEXT from image: {image.size[0]}x{image.size[1]} pixels]"


def extract_text_from_file(file_bytes: bytes, filename: str) -> str:
    """Extract text from file (image or PDF)"""
    file_ext = Path(filename).suffix.lower()
    
    if file_ext == SUPPORTED_PDF_FORMAT:
        # Convert PDF to images and extract text from each page
        images = convert_pdf_to_images(file_bytes)
        texts = []
        for i, image in enumerate(images):
            text = mock_ocr_from_image(image)
            texts.append(f"--- Page {i+1} ---\n{text}")
        return "\n\n".join(texts)
    
    elif file_ext in SUPPORTED_IMAGE_FORMATS:
        # Process image directly
        try:
            image = Image.open(io.BytesIO(file_bytes))
            return mock_ocr_from_image(image)
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to process image: {str(e)}")
    
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format: {file_ext}. Supported: {', '.join(SUPPORTED_IMAGE_FORMATS | {SUPPORTED_PDF_FORMAT})}"
        )


def cleanup_text(text: str) -> str:
    """
    Text Cleanup / Normalization
    - Remove excessive whitespace
    - Normalize line breaks
    - Remove special characters that might interfere
    """
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    # Normalize line breaks
    text = re.sub(r'\n\s*\n', '\n\n', text)
    # Remove leading/trailing whitespace from each line
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)
    return text.strip()


def build_schema_prompt(schema: Dict[str, str]) -> str:
    """
    Schema Builder - Convert user schema to LLM-readable format
    """
    schema_description = "Extract the following fields with their types:\n"
    for field_name, field_type in schema.items():
        schema_description += f"- {field_name}: {field_type}\n"
    return schema_description


def call_llm_for_extraction(raw_text: str, schema: Dict[str, str], is_retry: bool = False) -> Dict[str, Any]:
    """
    Call LLM (Gemini Flash or OpenAI) to extract structured data following schema
    
    Args:
        raw_text: OCR extracted text
        schema: Dictionary mapping field names to types (e.g., {"customer_name": "string"})
        is_retry: Whether this is a retry attempt after validation failure
    """
    # STRICT SYSTEM PROMPT
    system_instruction = (
        "You are a document extraction engine.\n\n"
        "Rules:\n"
        "- Extract ONLY from provided text.\n"
        "- Do NOT infer or guess.\n"
        "- Output VALID JSON only.\n"
        "- Follow schema EXACTLY.\n"
        "- Missing values must be null.\n"
        "- No extra keys allowed."
    )
    
    # Build schema description
    schema_prompt = build_schema_prompt(schema)
    
    # Add retry instruction if needed
    if is_retry:
        system_instruction += "\n\nFix output to exactly match schema. Return JSON only."
    
    user_message = f"""Schema to Extract:
{schema_prompt}

Document Text:
{raw_text}

Extract the fields according to the schema. Return ONLY valid JSON matching the schema exactly."""

    # Try Gemini first, fallback to OpenAI
    if gemini_model:
        try:
            logger.info(f"Calling Gemini Flash API {'(retry)' if is_retry else ''}")
            response = gemini_model.generate_content(
                f"{system_instruction}\n\n{user_message}",
                generation_config={
                    "temperature": 0,
                    "max_output_tokens": 2048,
                }
            )
            result_text = response.text.strip()
            # Remove markdown code blocks if present
            if result_text.startswith("```json"):
                result_text = result_text[7:]
            if result_text.startswith("```"):
                result_text = result_text[3:]
            if result_text.endswith("```"):
                result_text = result_text[:-3]
            result_text = result_text.strip()
            
            return json.loads(result_text)
        except Exception as e:
            logger.warning(f"Gemini API call failed: {e}, trying OpenAI fallback")
    
    # Fallback to OpenAI
    if openai_client:
        try:
            logger.info(f"Calling OpenAI API {'(retry)' if is_retry else ''}")
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_message}
                ],
                temperature=0,
                max_tokens=2048
            )
            result_text = response.choices[0].message.content.strip()
            # Remove markdown code blocks if present
            if result_text.startswith("```json"):
                result_text = result_text[7:]
            if result_text.startswith("```"):
                result_text = result_text[3:]
            if result_text.endswith("```"):
                result_text = result_text[:-3]
            result_text = result_text.strip()
            
            return json.loads(result_text)
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"LLM extraction failed: {str(e)}"
            )
    
    raise HTTPException(
        status_code=500,
        detail="No LLM API key configured. Please set GEMINI_API_KEY or OPENAI_API_KEY environment variable."
    )


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


@app.get("/")
async def health_check():
    """Health check endpoint"""
    return {"status": "running"}


@app.post(
    "/extract",
    response_model=ExtractionResponse,
    summary="Extract structured data from document",
    description="Upload a document file or provide a URL, along with a JSON schema to extract structured data"
)
async def extract_from_file_or_url(
    file: UploadFile = File(..., description="Upload a document file (PDF or image)"),
    file_url: Optional[str] = Form(None, description="URL of the document to process"),
    schema: str = Form(..., description="JSON schema string defining fields to extract (e.g., {\"customer_name\": \"string\", \"invoice_amount\": \"number\"})")
):
    """
    Extract structured data from document using JSON schema
    
    - Either file (upload) OR file_url must be provided
    - schema: JSON string describing fields to extract (e.g., {"customer_name": "string", "gst_number": "string", "invoice_amount": "number"})
    """
    try:
        # Validate input
        if not file and (not file_url or not file_url.strip()):
            raise HTTPException(
                status_code=400,
                detail="Either 'file' or 'file_url' must be provided"
            )
        
        if not schema or not schema.strip():
            raise HTTPException(
                status_code=400,
                detail="'schema' is required and cannot be empty"
            )
        
        # Parse schema JSON
        try:
            schema_dict = json.loads(schema)
            if not isinstance(schema_dict, dict):
                raise HTTPException(
                    status_code=400,
                    detail="Schema must be a JSON object"
                )
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid JSON schema: {str(e)}"
            )
        
        # Get file bytes and filename
        if file_url and file_url.strip():
            logger.info(f"Downloading file from URL: {file_url}")
            file_bytes = download_file_from_url(file_url)
            filename = file_url.split("/")[-1] or "downloaded_file"
        elif file:
            logger.info(f"Processing uploaded file: {file.filename}")
            file_bytes = await file.read()
            filename = file.filename or "uploaded_file"
        else:
            raise HTTPException(
                status_code=400,
                detail="Either 'file' or 'file_url' must be provided"
            )
        
        # Step 1: Extract text using OCR (mock for now)
        logger.info("Extracting text from document...")
        raw_text = extract_text_from_file(file_bytes, filename)
        logger.info(f"Extracted {len(raw_text)} characters of text")
        
        # Step 2: Text Cleanup / Normalization
        logger.info("Cleaning and normalizing text...")
        cleaned_text = cleanup_text(raw_text)
        
        # Step 3: Call LLM for extraction
        logger.info("Calling LLM for structured extraction...")
        extracted_data = call_llm_for_extraction(cleaned_text, schema_dict, is_retry=False)
        
        # Step 4: Validation
        logger.info("Validating extracted data...")
        is_valid, validated_data = validate_output(extracted_data, schema_dict)
        
        # Step 5: Retry once if invalid
        if not is_valid:
            logger.warning("Initial extraction failed validation, retrying...")
            extracted_data_retry = call_llm_for_extraction(cleaned_text, schema_dict, is_retry=True)
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
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse LLM response as JSON: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error in extraction: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
