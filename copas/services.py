"""
Copas Service Layer

Business logic for copas operations, separated from views
for better testability and future API support.
"""
import os

from core.gemini_client import GeminiPDFExtractor, ExtractionResult as CoreExtractionResult
from .models import ExtractionResult


def extract_text_from_pdf(uploaded_file) -> CoreExtractionResult:
    """
    Extract text from an uploaded PDF file using Gemini API.

    This function bridges Django's UploadedFile with the framework-agnostic
    GeminiPDFExtractor in the core module.

    Args:
        uploaded_file: Django UploadedFile object

    Returns:
        CoreExtractionResult with extracted text or error
    """
    # Get API key from environment
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        return CoreExtractionResult(
            success=False,
            error="Gemini API key not configured. Please set GEMINI_API_KEY in .env file."
        )

    try:
        # Read file content into memory
        pdf_bytes = uploaded_file.read()

        # Create extractor and process
        extractor = GeminiPDFExtractor(api_key)
        result = extractor.extract_text(pdf_bytes, uploaded_file.name)

        return result

    except Exception as e:
        return CoreExtractionResult(
            success=False,
            error=f"An error occurred during extraction: {str(e)}"
        )


def save_extraction_result(
    user,
    filename: str,
    file_size: int,
    extracted_text: str,
    prompt_tokens: int = None,
    completion_tokens: int = None,
    total_tokens: int = None,
    file_type: str = 'PDF'
) -> ExtractionResult:
    """
    Save extraction result to database.

    Args:
        user: Django User object
        filename: Original filename
        file_size: File size in bytes
        extracted_text: Extracted text content
        prompt_tokens: Tokens used for input (optional)
        completion_tokens: Tokens used for output (optional)
        total_tokens: Total tokens used (optional)
        file_type: File type (default: 'PDF')

    Returns:
        ExtractionResult model instance
    """
    return ExtractionResult.objects.create(
        user=user,
        filename=filename,
        file_type=file_type,
        file_size=file_size,
        extracted_text=extracted_text,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens
    )


def get_user_extractions(user, limit: int = 20):
    """Get user's extraction history."""
    return ExtractionResult.objects.filter(user=user)[:limit]
