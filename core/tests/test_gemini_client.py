"""
Tests for the Gemini PDF extraction client.
"""
import io
import unittest
from unittest.mock import patch, MagicMock

from core.gemini_client import (
    GeminiPDFExtractor,
    GeminiCachedExtractor,
    ExtractionResult,
    CacheExpiredError,
    validate_pdf_bytes,
    get_page_count,
)


class TestValidatePDFBytes(unittest.TestCase):
    """Tests for PDF validation function."""

    def test_valid_pdf_header(self):
        """Valid PDF bytes should pass validation."""
        pdf_bytes = b'%PDF-1.4 fake pdf content'
        is_valid, error = validate_pdf_bytes(pdf_bytes)
        self.assertTrue(is_valid)
        self.assertEqual(error, '')

    def test_invalid_pdf_header(self):
        """Non-PDF bytes should fail validation."""
        not_pdf = b'This is not a PDF file'
        is_valid, error = validate_pdf_bytes(not_pdf)
        self.assertFalse(is_valid)
        self.assertIn('not a valid PDF', error)

    def test_empty_bytes(self):
        """Empty bytes should fail validation."""
        is_valid, error = validate_pdf_bytes(b'')
        self.assertFalse(is_valid)
        self.assertIn('empty', error.lower())


def _create_mock_response(text='Extracted text', finish_reason_name='STOP',
                          prompt_tokens=None, completion_tokens=None, total_tokens=None,
                          has_candidates=True):
    """Helper to create mock SDK response objects."""
    mock_response = MagicMock()

    if has_candidates:
        mock_candidate = MagicMock()
        mock_finish_reason = MagicMock()
        mock_finish_reason.name = finish_reason_name
        mock_candidate.finish_reason = mock_finish_reason
        mock_response.candidates = [mock_candidate]
        mock_response.text = text
    else:
        mock_response.candidates = []
        mock_response.text = None

    # Usage metadata
    if prompt_tokens is not None:
        mock_usage = MagicMock()
        mock_usage.prompt_token_count = prompt_tokens
        mock_usage.candidates_token_count = completion_tokens
        mock_usage.total_token_count = total_tokens
        mock_response.usage_metadata = mock_usage
    else:
        mock_response.usage_metadata = None

    return mock_response


class TestGeminiPDFExtractor(unittest.TestCase):
    """Tests for GeminiPDFExtractor class (SDK-based)."""

    def test_init_requires_api_key(self):
        """Extractor should require an API key."""
        with self.assertRaises(ValueError):
            GeminiPDFExtractor('')

        with self.assertRaises(ValueError):
            GeminiPDFExtractor(None)

    def test_init_with_valid_key(self):
        """Extractor should initialize with valid API key."""
        extractor = GeminiPDFExtractor('test-api-key')
        self.assertEqual(extractor.api_key, 'test-api-key')

    def test_extract_text_invalid_pdf(self):
        """Invalid PDF should return error result."""
        extractor = GeminiPDFExtractor('test-key')
        result = extractor.extract_text(b'not a pdf')

        self.assertFalse(result.success)
        self.assertIsNone(result.text)
        self.assertIn('not a valid PDF', result.error)

    def test_extract_text_empty_pdf(self):
        """Empty bytes should return error result."""
        extractor = GeminiPDFExtractor('test-key')
        result = extractor.extract_text(b'')

        self.assertFalse(result.success)
        self.assertIsNone(result.text)

    @patch('core.gemini_client.genai.Client')
    def test_extract_text_success(self, mock_client_class):
        """Successful extraction should return text."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.models.generate_content.return_value = _create_mock_response(
            text='Extracted PDF content here'
        )

        extractor = GeminiPDFExtractor('test-key')
        pdf_bytes = b'%PDF-1.4 fake pdf'

        result = extractor.extract_text(pdf_bytes)

        self.assertTrue(result.success)
        self.assertEqual(result.text, 'Extracted PDF content here')
        self.assertIsNone(result.error)

    @patch('core.gemini_client.genai.Client')
    def test_extract_text_empty_response(self, mock_client_class):
        """Empty API response should return error."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.models.generate_content.return_value = _create_mock_response(
            has_candidates=False
        )

        extractor = GeminiPDFExtractor('test-key')
        pdf_bytes = b'%PDF-1.4 fake pdf'

        result = extractor.extract_text(pdf_bytes)

        self.assertFalse(result.success)
        self.assertIn('No response', result.error)

    @patch('core.gemini_client.genai.Client')
    def test_extract_text_api_error(self, mock_client_class):
        """API error should return failure result."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.models.generate_content.side_effect = Exception('API connection failed')

        extractor = GeminiPDFExtractor('test-key')
        pdf_bytes = b'%PDF-1.4 fake pdf'

        result = extractor.extract_text(pdf_bytes)

        self.assertFalse(result.success)
        self.assertIsNone(result.text)
        self.assertIn('API connection failed', result.error)

    @patch('core.gemini_client.genai.Client')
    def test_extract_text_truncated_max_tokens(self, mock_client_class):
        """Truncated response due to MAX_TOKENS should return error."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.models.generate_content.return_value = _create_mock_response(
            text='Partial content...',
            finish_reason_name='MAX_TOKENS'
        )

        extractor = GeminiPDFExtractor('test-key')
        pdf_bytes = b'%PDF-1.4 fake pdf'

        result = extractor.extract_text(pdf_bytes)

        self.assertFalse(result.success)
        self.assertIn('truncated', result.error.lower())

    @patch('core.gemini_client.genai.Client')
    def test_extract_text_blocked_safety(self, mock_client_class):
        """Response blocked by safety filters should return error."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.models.generate_content.return_value = _create_mock_response(
            text='',
            finish_reason_name='SAFETY'
        )

        extractor = GeminiPDFExtractor('test-key')
        pdf_bytes = b'%PDF-1.4 fake pdf'

        result = extractor.extract_text(pdf_bytes)

        self.assertFalse(result.success)
        self.assertIn('safety', result.error.lower())

    @patch('core.gemini_client.genai.Client')
    def test_extract_text_finish_reason_stop_succeeds(self, mock_client_class):
        """Response with finishReason STOP should succeed."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.models.generate_content.return_value = _create_mock_response(
            text='Complete extracted text',
            finish_reason_name='STOP'
        )

        extractor = GeminiPDFExtractor('test-key')
        pdf_bytes = b'%PDF-1.4 fake pdf'

        result = extractor.extract_text(pdf_bytes)

        self.assertTrue(result.success)
        self.assertEqual(result.text, 'Complete extracted text')

    @patch('core.gemini_client.genai.Client')
    def test_extract_text_returns_token_usage(self, mock_client_class):
        """Successful extraction should include token usage data."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.models.generate_content.return_value = _create_mock_response(
            text='Extracted text content',
            finish_reason_name='STOP',
            prompt_tokens=1500,
            completion_tokens=200,
            total_tokens=1700
        )

        extractor = GeminiPDFExtractor('test-key')
        pdf_bytes = b'%PDF-1.4 fake pdf'

        result = extractor.extract_text(pdf_bytes)

        self.assertTrue(result.success)
        self.assertEqual(result.prompt_tokens, 1500)
        self.assertEqual(result.completion_tokens, 200)
        self.assertEqual(result.total_tokens, 1700)

    @patch('core.gemini_client.genai.Client')
    def test_extract_text_handles_missing_token_usage(self, mock_client_class):
        """Extraction should succeed even without token usage data."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.models.generate_content.return_value = _create_mock_response(
            text='Extracted text',
            finish_reason_name='STOP',
            prompt_tokens=None  # No usage metadata
        )

        extractor = GeminiPDFExtractor('test-key')
        pdf_bytes = b'%PDF-1.4 fake pdf'

        result = extractor.extract_text(pdf_bytes)

        self.assertTrue(result.success)
        self.assertEqual(result.text, 'Extracted text')
        self.assertIsNone(result.prompt_tokens)
        self.assertIsNone(result.completion_tokens)
        self.assertIsNone(result.total_tokens)


class TestExtractionResult(unittest.TestCase):
    """Tests for ExtractionResult dataclass."""

    def test_success_result(self):
        """Success result should have text and no error."""
        result = ExtractionResult(success=True, text='Some text')
        self.assertTrue(result.success)
        self.assertEqual(result.text, 'Some text')
        self.assertIsNone(result.error)

    def test_failure_result(self):
        """Failure result should have error and no text."""
        result = ExtractionResult(success=False, error='Something went wrong')
        self.assertFalse(result.success)
        self.assertIsNone(result.text)
        self.assertEqual(result.error, 'Something went wrong')

    def test_result_with_token_fields(self):
        """ExtractionResult should support token usage fields."""
        result = ExtractionResult(
            success=True,
            text='Extracted content',
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150
        )
        self.assertTrue(result.success)
        self.assertEqual(result.text, 'Extracted content')
        self.assertEqual(result.prompt_tokens, 100)
        self.assertEqual(result.completion_tokens, 50)
        self.assertEqual(result.total_tokens, 150)

    def test_result_with_caching_fields(self):
        """ExtractionResult should support page_count and used_caching fields."""
        result = ExtractionResult(
            success=True,
            text='Extracted content',
            page_count=10,
            used_caching=True
        )
        self.assertTrue(result.success)
        self.assertEqual(result.page_count, 10)
        self.assertTrue(result.used_caching)

    def test_result_defaults_used_caching_to_false(self):
        """ExtractionResult should default used_caching to False."""
        result = ExtractionResult(success=True, text='Content')
        self.assertFalse(result.used_caching)
        self.assertIsNone(result.page_count)


class TestGetPageCount(unittest.TestCase):
    """Tests for get_page_count function."""

    @patch('core.gemini_client.PdfReader')
    def test_get_page_count_returns_count(self, mock_reader_class):
        """get_page_count should return number of pages."""
        mock_reader = MagicMock()
        mock_reader.pages = [MagicMock(), MagicMock(), MagicMock()]  # 3 pages
        mock_reader_class.return_value = mock_reader

        pdf_bytes = b'%PDF-1.4 fake pdf'
        count = get_page_count(pdf_bytes)

        self.assertEqual(count, 3)

    @patch('core.gemini_client.PdfReader')
    def test_get_page_count_single_page(self, mock_reader_class):
        """get_page_count should work for single page PDF."""
        mock_reader = MagicMock()
        mock_reader.pages = [MagicMock()]  # 1 page
        mock_reader_class.return_value = mock_reader

        pdf_bytes = b'%PDF-1.4 fake pdf'
        count = get_page_count(pdf_bytes)

        self.assertEqual(count, 1)


class TestCacheExpiredError(unittest.TestCase):
    """Tests for CacheExpiredError exception."""

    def test_cache_expired_error_message(self):
        """CacheExpiredError should store error message."""
        error = CacheExpiredError("Cache not found")
        self.assertEqual(str(error), "Cache not found")


class TestCalculateBatches(unittest.TestCase):
    """Tests for _calculate_batches method with PAGES_PER_BATCH=2."""

    def test_calculate_batches_11_pages(self):
        """11 pages should produce 6 batches with PAGES_PER_BATCH=2."""
        extractor = GeminiCachedExtractor('test-key')
        batches = extractor._calculate_batches(11)

        self.assertEqual(len(batches), 6)
        self.assertEqual(batches[0], (1, 2))
        self.assertEqual(batches[1], (3, 4))
        self.assertEqual(batches[2], (5, 6))
        self.assertEqual(batches[3], (7, 8))
        self.assertEqual(batches[4], (9, 10))
        self.assertEqual(batches[5], (11, 11))

    def test_calculate_batches_6_pages(self):
        """6 pages should produce 3 batches: (1-2), (3-4), (5-6)."""
        extractor = GeminiCachedExtractor('test-key')
        batches = extractor._calculate_batches(6)

        self.assertEqual(len(batches), 3)
        self.assertEqual(batches[0], (1, 2))
        self.assertEqual(batches[1], (3, 4))
        self.assertEqual(batches[2], (5, 6))

    def test_calculate_batches_7_pages(self):
        """7 pages should produce 4 batches: (1-2), (3-4), (5-6), (7-7)."""
        extractor = GeminiCachedExtractor('test-key')
        batches = extractor._calculate_batches(7)

        self.assertEqual(len(batches), 4)
        self.assertEqual(batches[0], (1, 2))
        self.assertEqual(batches[1], (3, 4))
        self.assertEqual(batches[2], (5, 6))
        self.assertEqual(batches[3], (7, 7))

    def test_calculate_batches_4_pages(self):
        """4 pages should produce 2 batches: (1-2), (3-4)."""
        extractor = GeminiCachedExtractor('test-key')
        batches = extractor._calculate_batches(4)

        self.assertEqual(len(batches), 2)
        self.assertEqual(batches[0], (1, 2))
        self.assertEqual(batches[1], (3, 4))

    def test_calculate_batches_3_pages(self):
        """3 pages should produce 2 batches: (1-2), (3-3)."""
        extractor = GeminiCachedExtractor('test-key')
        batches = extractor._calculate_batches(3)

        self.assertEqual(len(batches), 2)
        self.assertEqual(batches[0], (1, 2))
        self.assertEqual(batches[1], (3, 3))


class TestGeminiCachedExtractor(unittest.TestCase):
    """Tests for GeminiCachedExtractor class."""

    def test_init_requires_api_key(self):
        """Extractor should require an API key."""
        with self.assertRaises(ValueError):
            GeminiCachedExtractor('')

        with self.assertRaises(ValueError):
            GeminiCachedExtractor(None)

    def test_init_with_valid_key(self):
        """Extractor should initialize with valid API key."""
        extractor = GeminiCachedExtractor('test-api-key')
        self.assertEqual(extractor.api_key, 'test-api-key')

    def test_extract_text_invalid_pdf(self):
        """Invalid PDF should return error result."""
        extractor = GeminiCachedExtractor('test-key')
        result = extractor.extract_text(b'not a pdf')

        self.assertFalse(result.success)
        self.assertIsNone(result.text)
        self.assertIn('not a valid PDF', result.error)

    @patch('core.gemini_client.get_page_count')
    @patch('core.gemini_client.genai.Client')
    def test_small_pdf_uses_simple_extractor(self, mock_client_class, mock_page_count):
        """PDFs with <= 5 pages should use simple extractor."""
        mock_page_count.return_value = 3
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.models.generate_content.return_value = _create_mock_response(
            text='Small PDF content',
            finish_reason_name='STOP'
        )

        extractor = GeminiCachedExtractor('test-key')
        pdf_bytes = b'%PDF-1.4 fake pdf'

        result = extractor.extract_text(pdf_bytes)

        self.assertTrue(result.success)
        self.assertEqual(result.text, 'Small PDF content')
        mock_page_count.assert_called_once()

    @patch('core.gemini_client.get_page_count')
    @patch.object(GeminiCachedExtractor, '_extract_large_pdf')
    def test_large_pdf_uses_cached_extractor(self, mock_extract_large, mock_page_count):
        """PDFs with > 5 pages should use cached extraction."""
        mock_page_count.return_value = 10
        mock_extract_large.return_value = ExtractionResult(
            success=True,
            text='Large PDF content',
            prompt_tokens=1000,
            completion_tokens=500,
            total_tokens=1500
        )

        extractor = GeminiCachedExtractor('test-key')
        pdf_bytes = b'%PDF-1.4 fake pdf'

        result = extractor.extract_text(pdf_bytes)

        self.assertTrue(result.success)
        self.assertEqual(result.text, 'Large PDF content')
        mock_extract_large.assert_called_once()

    @patch('core.gemini_client.get_page_count')
    def test_pdf_read_error_returns_failure(self, mock_page_count):
        """Failed PDF read should return error result."""
        mock_page_count.side_effect = Exception("Corrupt PDF")

        extractor = GeminiCachedExtractor('test-key')
        pdf_bytes = b'%PDF-1.4 corrupt pdf'

        result = extractor.extract_text(pdf_bytes)

        self.assertFalse(result.success)
        self.assertIn('Failed to read PDF', result.error)

    @patch('core.gemini_client.get_page_count')
    @patch.object(GeminiCachedExtractor, '_upload_file')
    @patch.object(GeminiCachedExtractor, '_create_cache')
    @patch.object(GeminiCachedExtractor, '_generate_batch_with_cache')
    @patch.object(GeminiCachedExtractor, '_delete_cache')
    @patch.object(GeminiCachedExtractor, '_delete_file')
    def test_large_pdf_extraction_flow(
        self, mock_delete_file, mock_delete_cache,
        mock_generate_batch, mock_create_cache, mock_upload, mock_page_count
    ):
        """Large PDF extraction should follow correct flow with batched pages (PAGES_PER_BATCH=2)."""
        mock_page_count.return_value = 6
        mock_upload.return_value = MagicMock(uri="https://example.com/files/abc123", name="files/abc123")
        mock_create_cache.return_value = MagicMock(name="cachedContents/xyz789")
        mock_generate_batch.return_value = {
            "text": "| Batch content |",
            "prompt_tokens": 100,
            "completion_tokens": 50
        }

        extractor = GeminiCachedExtractor('test-key')
        pdf_bytes = b'%PDF-1.4 fake pdf'

        result = extractor.extract_text(pdf_bytes)

        self.assertTrue(result.success)
        # With PAGES_PER_BATCH=2 and 6 pages: (1-2), (3-4), (5-6) = 3 batches
        self.assertIn("## Pages 1-2", result.text)
        self.assertIn("## Pages 5-6", result.text)
        self.assertEqual(mock_generate_batch.call_count, 3)  # 3 batches: (1-2), (3-4), (5-6)
        mock_delete_cache.assert_called_once()
        mock_delete_file.assert_called_once()

    @patch('core.gemini_client.get_page_count')
    @patch.object(GeminiCachedExtractor, '_upload_file')
    @patch.object(GeminiCachedExtractor, '_create_cache')
    @patch.object(GeminiCachedExtractor, '_generate_batch_with_cache')
    @patch.object(GeminiCachedExtractor, '_delete_cache')
    @patch.object(GeminiCachedExtractor, '_delete_file')
    def test_cache_expired_recreates_cache(
        self, mock_delete_file, mock_delete_cache,
        mock_generate_batch, mock_create_cache, mock_upload, mock_page_count
    ):
        """Cache expiration should trigger cache recreation."""
        mock_page_count.return_value = 6  # With PAGES_PER_BATCH=2: 3 batches (1-2), (3-4), (5-6)
        mock_upload.return_value = MagicMock(uri="https://example.com/files/abc123", name="files/abc123")
        mock_create_cache.side_effect = [
            MagicMock(name="cache1"),
            MagicMock(name="cache2")
        ]  # First create, then recreate

        # First batch succeeds, second fails with cache expired, then retry succeeds
        mock_generate_batch.side_effect = [
            {"text": "Batch 1-2", "prompt_tokens": 100, "completion_tokens": 50},
            CacheExpiredError("Cache expired"),  # Batch 3-4 fails
            {"text": "Batch 3-4", "prompt_tokens": 100, "completion_tokens": 50},  # Retry
            {"text": "Batch 5-6", "prompt_tokens": 50, "completion_tokens": 25},
        ]

        extractor = GeminiCachedExtractor('test-key')
        pdf_bytes = b'%PDF-1.4 fake pdf'

        result = extractor.extract_text(pdf_bytes)

        self.assertTrue(result.success)
        self.assertEqual(mock_create_cache.call_count, 2)  # Initial + recreation

    @patch('core.gemini_client.get_page_count')
    @patch.object(GeminiCachedExtractor, '_upload_file')
    @patch.object(GeminiCachedExtractor, '_create_cache')
    @patch.object(GeminiCachedExtractor, '_delete_cache')
    @patch.object(GeminiCachedExtractor, '_delete_file')
    def test_cleanup_on_error(
        self, mock_delete_file, mock_delete_cache,
        mock_create_cache, mock_upload, mock_page_count
    ):
        """Resources should be cleaned up even on error."""
        mock_page_count.return_value = 6
        mock_upload.return_value = MagicMock(uri="https://example.com/files/abc123", name="files/abc123")
        mock_create_cache.side_effect = Exception("Cache creation failed")

        extractor = GeminiCachedExtractor('test-key')
        pdf_bytes = b'%PDF-1.4 fake pdf'

        result = extractor.extract_text(pdf_bytes)

        self.assertFalse(result.success)
        mock_delete_file.assert_called_once()  # File should still be deleted


if __name__ == '__main__':
    unittest.main()
