"""
Gemini API Client for PDF Processing

Pure Python implementation - NO Django imports.
Handles communication with Google's Gemini API for PDF text extraction.
Uses the google-genai SDK for all API calls.
Supports context caching for large PDFs (>5 pages).
"""

import io
import os
from dataclasses import dataclass
from typing import Optional

from google import genai
from google.genai import types
from pypdf import PdfReader

# Default model name, can be overridden via GEMINI_MODEL_NAME environment variable
DEFAULT_MODEL_NAME = "gemini-2.5-flash"


@dataclass
class ExtractionResult:
    """Result of PDF text extraction."""

    success: bool
    text: Optional[str] = None
    error: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    page_count: Optional[int] = None
    used_caching: bool = False


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

    if not pdf_bytes.startswith(b"%PDF-"):
        return False, "File is not a valid PDF"

    return True, ""


def get_page_count(pdf_bytes: bytes) -> int:
    """
    Get the number of pages in a PDF.

    Args:
        pdf_bytes: Raw PDF file content

    Returns:
        Number of pages in the PDF
    """
    reader = PdfReader(io.BytesIO(pdf_bytes))
    return len(reader.pages)


class CacheExpiredError(Exception):
    """Raised when the Gemini cache has expired or is invalid."""

    pass


class GeminiPDFExtractor:
    """
    Handles PDF text extraction via Gemini SDK.

    This class is framework-agnostic and can be used in any Python context.
    Uses the google-genai SDK for API calls.
    """

    def __init__(self, api_key: str, model_name: str = None):
        """
        Initialize the extractor with a Gemini API key.

        Args:
            api_key: Google Gemini API key
            model_name: Gemini model to use (defaults to GEMINI_MODEL_NAME env var or gemini-2.5-flash)
        """
        if not api_key:
            raise ValueError("Gemini API key is required")
        self.api_key = api_key
        self.model_name = model_name or os.environ.get("GEMINI_MODEL_NAME", DEFAULT_MODEL_NAME)
        self.client = genai.Client(api_key=api_key)

    def extract_text(self, pdf_bytes: bytes, filename: str = "document.pdf") -> ExtractionResult:
        """
        Extract text content from PDF bytes using Gemini SDK.

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
            # Build prompt
            prompt = self._build_prompt()

            # Call Gemini SDK with inline PDF data
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
                            types.Part.from_text(text=prompt),
                        ],
                    )
                ],
            )

            # Parse response
            return self._parse_response(response)

        except Exception as e:
            return ExtractionResult(
                success=False, error=f"Extraction failed: {type(e).__name__}: {str(e)}"
            )

    def _build_prompt(self) -> str:
        """Build the extraction prompt."""
        return """
        Look at ALL PAGES of this document.

        Extract the table data found specifically on every PAGES into a Markdown table.
        - Columns: No., Item No., Description, Brand, Origin, HS Code, Qty, Unit Price, Total.
        - Do NOT extract data from other pages.
        - Do NOT include the document headers (Shipper/Consignee) in the table, just the line items.
        - If this is the first page, start with the table header.
        - If this is a subsequent page, do NOT repeat the table header row.
        - If one or few of the columns are not found, return an empty value for that column.
        """

    def _parse_response(self, response) -> ExtractionResult:
        """Parse Gemini SDK response into ExtractionResult."""
        # Check for valid candidates
        if not response.candidates:
            return ExtractionResult(success=False, error="No response from Gemini API")

        candidate = response.candidates[0]

        # Check finish_reason to detect truncated responses
        finish_reason = candidate.finish_reason
        if finish_reason and finish_reason.name not in ("STOP", "FINISH_REASON_UNSPECIFIED"):
            reason_messages = {
                "MAX_TOKENS": "Response was truncated due to maximum token limit",
                "SAFETY": "Response was blocked due to safety filters",
                "RECITATION": "Response was blocked due to recitation concerns",
                "OTHER": "Response generation stopped unexpectedly",
            }
            error_msg = reason_messages.get(
                finish_reason.name, f"Response incomplete (finish_reason: {finish_reason.name})"
            )
            return ExtractionResult(success=False, error=error_msg)

        # Get text from response
        text = response.text if response.text else ""

        if not text:
            return ExtractionResult(success=False, error="No text extracted from PDF")

        # Extract token usage
        usage = response.usage_metadata
        prompt_tokens = usage.prompt_token_count if usage else None
        completion_tokens = usage.candidates_token_count if usage else None
        total_tokens = usage.total_token_count if usage else None

        return ExtractionResult(
            success=True,
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )


class GeminiCachedExtractor:
    """
    Handles large PDF extraction using Gemini's context caching.

    For PDFs with more than LARGE_PDF_THRESHOLD pages, this class:
    1. Uploads the file to Gemini's File API
    2. Creates a cached content reference
    3. Extracts data page-by-page
    4. Cleans up cache and file when done
    """

    LARGE_PDF_THRESHOLD = 5
    PAGES_PER_BATCH = 2  # Number of pages per API call for batched extraction
    CACHE_TTL_SECONDS = 600  # 10 minutes

    def __init__(self, api_key: str, model_name: str = None):
        """
        Initialize with Gemini API key and SDK client.

        Args:
            api_key: Google Gemini API key
            model_name: Gemini model to use (defaults to GEMINI_MODEL_NAME env var or gemini-2.5-flash)
        """
        if not api_key:
            raise ValueError("Gemini API key is required")
        self.api_key = api_key
        self.model_name = model_name or os.environ.get("GEMINI_MODEL_NAME", DEFAULT_MODEL_NAME)
        self.client = genai.Client(api_key=api_key)

    def extract_text(self, pdf_bytes: bytes, filename: str = "document.pdf") -> ExtractionResult:
        """
        Extract text from PDF, using caching for large PDFs.

        Args:
            pdf_bytes: Raw PDF file content
            filename: Original filename

        Returns:
            ExtractionResult with extracted text or error
        """
        # Validate PDF
        is_valid, error_msg = validate_pdf_bytes(pdf_bytes)
        if not is_valid:
            return ExtractionResult(success=False, error=error_msg)

        try:
            page_count = get_page_count(pdf_bytes)
        except Exception as e:
            return ExtractionResult(success=False, error=f"Failed to read PDF: {str(e)}")

        # Use simple extractor for small PDFs
        if page_count <= self.LARGE_PDF_THRESHOLD:
            simple_extractor = GeminiPDFExtractor(self.api_key, self.model_name)
            result = simple_extractor.extract_text(pdf_bytes, filename)
            # Add page count to result
            result.page_count = page_count
            result.used_caching = False
            return result

        # Use cached extraction for large PDFs
        return self._extract_large_pdf(pdf_bytes, filename, page_count)

    def _calculate_batches(self, page_count: int) -> list[tuple[int, int]]:
        """
        Calculate page batches for extraction.

        Args:
            page_count: Total number of pages in the PDF

        Returns:
            List of (start_page, end_page) tuples (1-indexed, inclusive)
        """
        batches = []
        for start in range(1, page_count + 1, self.PAGES_PER_BATCH):
            end = min(start + self.PAGES_PER_BATCH - 1, page_count)
            batches.append((start, end))
        return batches

    def _extract_large_pdf(
        self, pdf_bytes: bytes, filename: str, page_count: int
    ) -> ExtractionResult:
        """Extract from large PDF using context caching with batched pages."""
        uploaded_file = None
        cached_content = None

        try:
            # Upload file to File API
            uploaded_file = self._upload_file(pdf_bytes, filename)

            # Try to create cache - may fail if minimum token requirement not met
            try:
                cached_content = self._create_cache(uploaded_file)
            except Exception as e:
                error_msg = str(e).lower()
                # If minimum token requirement not met, fall back to batched extraction without caching
                if "minimum" in error_msg or "token" in error_msg or "too few" in error_msg:
                    return self._extract_batched_without_cache(uploaded_file, page_count)
                raise  # Re-raise if it's a different error

            # Calculate batches
            batches = self._calculate_batches(page_count)

            # Extract batch by batch
            results = {}
            total_prompt_tokens = 0
            total_completion_tokens = 0

            for batch_idx, (start_page, end_page) in enumerate(batches):
                try:
                    batch_result = self._generate_batch_with_cache(
                        cached_content, start_page, end_page, is_first_batch=(batch_idx == 0)
                    )
                except CacheExpiredError:
                    # Cache expired - recreate and retry
                    cached_content = self._create_cache(uploaded_file)
                    batch_result = self._generate_batch_with_cache(
                        cached_content, start_page, end_page, is_first_batch=(batch_idx == 0)
                    )

                results[(start_page, end_page)] = batch_result["text"]
                total_prompt_tokens += batch_result.get("prompt_tokens", 0)
                total_completion_tokens += batch_result.get("completion_tokens", 0)

            # Combine results
            combined_text = "\n\n".join(
                f"## Pages {start}-{end}\n{text}" for (start, end), text in results.items()
            )

            return ExtractionResult(
                success=True,
                text=combined_text,
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
                total_tokens=total_prompt_tokens + total_completion_tokens,
                page_count=page_count,
                used_caching=True,
            )

        except CacheExpiredError as e:
            return ExtractionResult(success=False, error=f"Cache error: {str(e)}")
        except Exception as e:
            return ExtractionResult(
                success=False, error=f"Extraction failed: {type(e).__name__}: {str(e)}"
            )

        finally:
            # Always cleanup resources
            if cached_content:
                self._delete_cache(cached_content)
            if uploaded_file:
                self._delete_file(uploaded_file)

    def _extract_batched_without_cache(
        self, uploaded_file: types.File, page_count: int
    ) -> ExtractionResult:
        """
        Extract from PDF using batched requests without caching.

        Used when the PDF doesn't meet minimum token requirements for caching.
        """
        try:
            # Calculate batches
            batches = self._calculate_batches(page_count)

            # Extract batch by batch without caching
            results = {}
            total_prompt_tokens = 0
            total_completion_tokens = 0

            for batch_idx, (start_page, end_page) in enumerate(batches):
                batch_result = self._generate_batch_without_cache(
                    uploaded_file, start_page, end_page, is_first_batch=(batch_idx == 0)
                )

                results[(start_page, end_page)] = batch_result["text"]
                total_prompt_tokens += batch_result.get("prompt_tokens", 0)
                total_completion_tokens += batch_result.get("completion_tokens", 0)

            # Combine results
            combined_text = "\n\n".join(
                f"## Pages {start}-{end}\n{text}" for (start, end), text in results.items()
            )

            return ExtractionResult(
                success=True,
                text=combined_text,
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
                total_tokens=total_prompt_tokens + total_completion_tokens,
                page_count=page_count,
                used_caching=False,
            )

        except Exception as e:
            return ExtractionResult(
                success=False, error=f"Batched extraction failed: {type(e).__name__}: {str(e)}"
            )

        finally:
            # Cleanup uploaded file
            if uploaded_file:
                self._delete_file(uploaded_file)

    def _upload_file(self, pdf_bytes: bytes, filename: str) -> types.File:
        """
        Upload file to Gemini File API using SDK.

        Returns:
            Uploaded File object from SDK
        """
        uploaded_file = self.client.files.upload(
            file=io.BytesIO(pdf_bytes),
            config=types.UploadFileConfig(display_name=filename, mime_type="application/pdf"),
        )
        return uploaded_file

    def _create_cache(self, uploaded_file: types.File) -> types.CachedContent:
        """
        Create cached content from uploaded file using SDK.

        Args:
            uploaded_file: The uploaded File object from SDK

        Returns:
            CachedContent object from SDK
        """
        cached_content = self.client.caches.create(
            model=self.model_name,
            config=types.CreateCachedContentConfig(
                contents=[
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_uri(
                                file_uri=uploaded_file.uri, mime_type="application/pdf"
                            )
                        ],
                    )
                ],
                system_instruction="You are a document parser. Extract table data as Markdown.",
                ttl=f"{self.CACHE_TTL_SECONDS}s",
            ),
        )
        return cached_content

    def _generate_batch_with_cache(
        self,
        cached_content: types.CachedContent,
        start_page: int,
        end_page: int,
        is_first_batch: bool = False,
    ) -> dict:
        """
        Generate content for a batch of pages using cached content.

        Args:
            cached_content: The CachedContent object from SDK
            start_page: First page in batch (1-indexed)
            end_page: Last page in batch (1-indexed, inclusive)
            is_first_batch: If True, include table header in output

        Returns:
            Dict with "text", "prompt_tokens", "completion_tokens"

        Raises:
            CacheExpiredError: If cache is expired or invalid
        """
        if start_page == end_page:
            page_spec = f"PAGE {start_page}"
        else:
            page_spec = f"PAGES {start_page} to {end_page}"

        header_instruction = (
            "Start with the table header row."
            if is_first_batch
            else "Do NOT include the table header row."
        )

        prompt = f"""
        Look at {page_spec} of this document.

        Extract the table data found specifically on {page_spec} into a Markdown table.
        - Columns: No., Item No., Description, Brand, Origin, HS Code, Qty, Unit Price, Total.
        - Do NOT extract data from other pages.
        - Do NOT include document headers (Shipper/Consignee), just line items.
        - {header_instruction}
        - If columns are not found, return empty values for those columns.
        """

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(cached_content=cached_content.name),
            )
        except Exception as e:
            error_msg = str(e).lower()
            if "cache" in error_msg or "expired" in error_msg or "not found" in error_msg:
                raise CacheExpiredError(f"Cache expired or invalid: {e}")
            raise

        # Check finish reason
        if response.candidates:
            candidate = response.candidates[0]
            finish_reason = candidate.finish_reason
            if finish_reason and finish_reason.name not in ("STOP", "FINISH_REASON_UNSPECIFIED"):
                raise ValueError(f"Response incomplete: {finish_reason.name}")

        text = response.text if response.text else ""

        # Extract token usage
        usage = response.usage_metadata
        prompt_tokens = usage.prompt_token_count if usage else 0
        completion_tokens = usage.candidates_token_count if usage else 0

        return {
            "text": text,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }

    def _generate_batch_without_cache(
        self,
        uploaded_file: types.File,
        start_page: int,
        end_page: int,
        is_first_batch: bool = False,
    ) -> dict:
        """
        Generate content for a batch of pages WITHOUT using cached content.

        Args:
            uploaded_file: The uploaded File object from SDK
            start_page: First page in batch (1-indexed)
            end_page: Last page in batch (1-indexed, inclusive)
            is_first_batch: If True, include table header in output

        Returns:
            Dict with "text", "prompt_tokens", "completion_tokens"
        """
        if start_page == end_page:
            page_spec = f"PAGE {start_page}"
        else:
            page_spec = f"PAGES {start_page} to {end_page}"

        header_instruction = (
            "Start with the table header row."
            if is_first_batch
            else "Do NOT include the table header row."
        )

        prompt = f"""
        Look at {page_spec} of this document.

        Extract the table data found specifically on {page_spec} into a Markdown table.
        - Columns: No., Item No., Description, Brand, Origin, HS Code, Qty, Unit Price, Total.
        - Do NOT extract data from other pages.
        - Do NOT include document headers (Shipper/Consignee), just line items.
        - {header_instruction}
        - If columns are not found, return empty values for those columns.
        """

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_uri(
                            file_uri=uploaded_file.uri, mime_type="application/pdf"
                        ),
                        types.Part.from_text(text=prompt),
                    ],
                )
            ],
        )

        # Check finish reason
        if response.candidates:
            candidate = response.candidates[0]
            finish_reason = candidate.finish_reason
            if finish_reason and finish_reason.name not in ("STOP", "FINISH_REASON_UNSPECIFIED"):
                raise ValueError(f"Response incomplete: {finish_reason.name}")

        text = response.text if response.text else ""

        # Extract token usage
        usage = response.usage_metadata
        prompt_tokens = usage.prompt_token_count if usage else 0
        completion_tokens = usage.candidates_token_count if usage else 0

        return {
            "text": text,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }

    def _delete_cache(self, cached_content: types.CachedContent) -> None:
        """Delete cached content using SDK."""
        try:
            self.client.caches.delete(name=cached_content.name)
        except Exception:
            pass  # Ignore errors on cleanup

    def _delete_file(self, uploaded_file: types.File) -> None:
        """Delete uploaded file using SDK."""
        try:
            self.client.files.delete(name=uploaded_file.name)
        except Exception:
            pass  # Ignore errors on cleanup
