"""
Tests for the Gemini PDF extraction client.
"""
import unittest
from unittest.mock import patch, MagicMock

from core.gemini_client import (
    GeminiPDFExtractor,
    ExtractionResult,
    validate_pdf_bytes,
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


class TestGeminiPDFExtractor(unittest.TestCase):
    """Tests for GeminiPDFExtractor class."""

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

    @patch.object(GeminiPDFExtractor, '_call_gemini_api')
    def test_extract_text_success(self, mock_api_call):
        """Successful extraction should return text."""
        mock_api_call.return_value = {
            'candidates': [{
                'content': {
                    'parts': [{'text': 'Extracted PDF content here'}]
                }
            }]
        }

        extractor = GeminiPDFExtractor('test-key')
        pdf_bytes = b'%PDF-1.4 fake pdf'

        result = extractor.extract_text(pdf_bytes)

        self.assertTrue(result.success)
        self.assertEqual(result.text, 'Extracted PDF content here')
        self.assertIsNone(result.error)

    @patch.object(GeminiPDFExtractor, '_call_gemini_api')
    def test_extract_text_empty_response(self, mock_api_call):
        """Empty API response should return error."""
        mock_api_call.return_value = {'candidates': []}

        extractor = GeminiPDFExtractor('test-key')
        pdf_bytes = b'%PDF-1.4 fake pdf'

        result = extractor.extract_text(pdf_bytes)

        self.assertFalse(result.success)
        self.assertIn('No response', result.error)

    @patch.object(GeminiPDFExtractor, '_call_gemini_api')
    def test_extract_text_api_error(self, mock_api_call):
        """API error should return failure result."""
        mock_api_call.side_effect = Exception('API connection failed')

        extractor = GeminiPDFExtractor('test-key')
        pdf_bytes = b'%PDF-1.4 fake pdf'

        result = extractor.extract_text(pdf_bytes)

        self.assertFalse(result.success)
        self.assertIsNone(result.text)
        self.assertIn('API connection failed', result.error)

    @patch.object(GeminiPDFExtractor, '_call_gemini_api')
    def test_extract_text_truncated_max_tokens(self, mock_api_call):
        """Truncated response due to MAX_TOKENS should return error."""
        mock_api_call.return_value = {
            'candidates': [{
                'content': {
                    'parts': [{'text': 'Partial content...'}]
                },
                'finishReason': 'MAX_TOKENS'
            }]
        }

        extractor = GeminiPDFExtractor('test-key')
        pdf_bytes = b'%PDF-1.4 fake pdf'

        result = extractor.extract_text(pdf_bytes)

        self.assertFalse(result.success)
        self.assertIn('truncated', result.error.lower())

    @patch.object(GeminiPDFExtractor, '_call_gemini_api')
    def test_extract_text_blocked_safety(self, mock_api_call):
        """Response blocked by safety filters should return error."""
        mock_api_call.return_value = {
            'candidates': [{
                'content': {
                    'parts': [{'text': ''}]
                },
                'finishReason': 'SAFETY'
            }]
        }

        extractor = GeminiPDFExtractor('test-key')
        pdf_bytes = b'%PDF-1.4 fake pdf'

        result = extractor.extract_text(pdf_bytes)

        self.assertFalse(result.success)
        self.assertIn('safety', result.error.lower())

    @patch.object(GeminiPDFExtractor, '_call_gemini_api')
    def test_extract_text_finish_reason_stop_succeeds(self, mock_api_call):
        """Response with finishReason STOP should succeed."""
        mock_api_call.return_value = {
            'candidates': [{
                'content': {
                    'parts': [{'text': 'Complete extracted text'}]
                },
                'finishReason': 'STOP'
            }]
        }

        extractor = GeminiPDFExtractor('test-key')
        pdf_bytes = b'%PDF-1.4 fake pdf'

        result = extractor.extract_text(pdf_bytes)

        self.assertTrue(result.success)
        self.assertEqual(result.text, 'Complete extracted text')


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


if __name__ == '__main__':
    unittest.main()
