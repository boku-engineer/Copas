"""
Gemini API Client for PDF Processing

Pure Python implementation - NO Django imports.
Handles communication with Google's Gemini API for PDF text extraction.
"""
import base64
import json
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Optional


@dataclass
class ExtractionResult:
    """Result of PDF text extraction."""
    success: bool
    text: Optional[str] = None
    error: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


def validate_pdf_bytes(pdf_bytes: bytes) -> tuple[bool, str]:
    """
    Validate that bytes represent a valid PDF file.

    Args:
        pdf_bytes: Raw file content

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not pdf_bytes:
        return False, "File is empty"

    if not pdf_bytes.startswith(b'%PDF-'):
        return False, "File is not a valid PDF"

    return True, ""


class GeminiPDFExtractor:
    """
    Handles PDF text extraction via Gemini API.

    This class is framework-agnostic and can be used in any Python context.
    """

    GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

    def __init__(self, api_key: str):
        """
        Initialize the extractor with a Gemini API key.

        Args:
            api_key: Google Gemini API key
        """
        if not api_key:
            raise ValueError("Gemini API key is required")
        self.api_key = api_key

    def extract_text(self, pdf_bytes: bytes, filename: str = "document.pdf") -> ExtractionResult:
        """
        Extract text content from PDF bytes using Gemini API.

        Args:
            pdf_bytes: Raw PDF file content as bytes
            filename: Original filename for context (optional)

        Returns:
            ExtractionResult with extracted text or error details
        """
        # Validate PDF
        is_valid, error_msg = validate_pdf_bytes(pdf_bytes)
        if not is_valid:
            return ExtractionResult(success=False, error=error_msg)

        try:
            # Encode PDF to base64
            base64_pdf = self._encode_pdf_to_base64(pdf_bytes)

            # Build request payload
            payload = self._build_request_payload(base64_pdf)

            # Call Gemini API
            response = self._call_gemini_api(payload)

            # Parse response
            return self._parse_response(response)

        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8', errors='ignore')
            return ExtractionResult(
                success=False,
                error=f"API request failed ({e.code}): {error_body}"
            )
        except urllib.error.URLError as e:
            return ExtractionResult(
                success=False,
                error=f"Network error: {str(e.reason)}"
            )
        except Exception as e:
            return ExtractionResult(
                success=False,
                error=f"Unexpected error: {str(e)}"
            )

    def _encode_pdf_to_base64(self, pdf_bytes: bytes) -> str:
        """Encode PDF bytes to base64 string for API transmission."""
        return base64.b64encode(pdf_bytes).decode('utf-8')

    def _build_request_payload(self, base64_pdf: str) -> dict:
        """Build the Gemini API request payload."""
        return {
            "contents": [{
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": "application/pdf",
                            "data": base64_pdf
                        }
                    },
                    {
                        "text": "Extract all text content from this PDF document. Return only the text, preserving paragraph structure. Do not add any commentary or explanation."
                    }
                ]
            }]
        }

    def _call_gemini_api(self, payload: dict) -> dict:
        """Make the HTTP request to Gemini API."""
        url = f"{self.GEMINI_API_URL}?key={self.api_key}"

        data = json.dumps(payload).encode('utf-8')

        request = urllib.request.Request(
            url,
            data=data,
            headers={
                'Content-Type': 'application/json',
            },
            method='POST'
        )

        with urllib.request.urlopen(request, timeout=60) as response:
            response_data = response.read().decode('utf-8')
            return json.loads(response_data)

    def _parse_response(self, response: dict) -> ExtractionResult:
        """Parse Gemini API response into ExtractionResult."""
        try:
            candidates = response.get('candidates', [])
            if not candidates:
                return ExtractionResult(
                    success=False,
                    error="No response from Gemini API"
                )

            candidate = candidates[0]

            # Check finish_reason to detect truncated responses
            finish_reason = candidate.get('finishReason', '')
            if finish_reason and finish_reason != 'STOP':
                reason_messages = {
                    'MAX_TOKENS': 'Response was truncated due to maximum token limit',
                    'SAFETY': 'Response was blocked due to safety filters',
                    'RECITATION': 'Response was blocked due to recitation concerns',
                    'OTHER': 'Response generation stopped unexpectedly',
                }
                error_msg = reason_messages.get(
                    finish_reason,
                    f'Response incomplete (finish_reason: {finish_reason})'
                )
                return ExtractionResult(success=False, error=error_msg)

            content = candidate.get('content', {})
            parts = content.get('parts', [])

            if not parts:
                return ExtractionResult(
                    success=False,
                    error="Empty response from Gemini API"
                )

            text = parts[0].get('text', '')

            if not text:
                return ExtractionResult(
                    success=False,
                    error="No text extracted from PDF"
                )

            # Extract token usage from usageMetadata
            usage_metadata = response.get('usageMetadata', {})
            prompt_tokens = usage_metadata.get('promptTokenCount')
            completion_tokens = usage_metadata.get('candidatesTokenCount')
            total_tokens = usage_metadata.get('totalTokenCount')

            return ExtractionResult(
                success=True,
                text=text,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens
            )

        except (KeyError, IndexError) as e:
            return ExtractionResult(
                success=False,
                error=f"Failed to parse API response: {str(e)}"
            )
