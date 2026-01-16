"""
Copas Service Layer

Business logic for copas operations, separated from views
for better testability and future API support.
"""
import os

from core.gemini_client import GeminiPDFExtractor, ExtractionResult


def extract_text_from_pdf(uploaded_file) -> ExtractionResult:
    """
    Extract text from an uploaded PDF file using Gemini API.

    This function bridges Django's UploadedFile with the framework-agnostic
    GeminiPDFExtractor in the core module.

    Args:
        uploaded_file: Django UploadedFile object

    Returns:
        ExtractionResult with extracted text or error
    """
    # Get API key from environment
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        return ExtractionResult(
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
        return ExtractionResult(
            success=False,
            error=f"An error occurred during extraction: {str(e)}"
        )


def create_resource(user):
    """Create a new resource for the user."""
    pass


def get_current_resource(user):
    """Get the current resource for display."""
    pass


def get_user_history(user, limit=20):
    """Get the user's resource history."""
    pass
