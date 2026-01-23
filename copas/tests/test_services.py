"""
Tests for Copas services.
"""
from django.test import TestCase

from accounts.models import CustomUser
from copas.models import ExtractionResult
from copas.services import save_extraction_result


class SaveExtractionResultTests(TestCase):
    """Tests for save_extraction_result service function."""

    def setUp(self):
        self.user = CustomUser.objects.create_user(
            username='testuser',
            email='test@example.com',
            password='testpass123'
        )

    def test_save_extraction_result_creates_record(self):
        """save_extraction_result creates a database record."""
        result = save_extraction_result(
            user=self.user,
            filename='test.pdf',
            file_size=1024,
            extracted_text='Extracted content here.'
        )

        self.assertIsInstance(result, ExtractionResult)
        self.assertEqual(ExtractionResult.objects.count(), 1)
        self.assertEqual(result.user, self.user)
        self.assertEqual(result.filename, 'test.pdf')
        self.assertEqual(result.file_size, 1024)
        self.assertEqual(result.extracted_text, 'Extracted content here.')

    def test_save_extraction_result_with_token_data(self):
        """save_extraction_result saves token usage data."""
        result = save_extraction_result(
            user=self.user,
            filename='document.pdf',
            file_size=2048,
            extracted_text='Some text.',
            prompt_tokens=500,
            completion_tokens=100,
            total_tokens=600
        )

        self.assertEqual(result.prompt_tokens, 500)
        self.assertEqual(result.completion_tokens, 100)
        self.assertEqual(result.total_tokens, 600)

    def test_save_extraction_result_default_file_type(self):
        """save_extraction_result defaults file_type to PDF."""
        result = save_extraction_result(
            user=self.user,
            filename='test.pdf',
            file_size=512,
            extracted_text='Text.'
        )

        self.assertEqual(result.file_type, 'PDF')

    def test_save_extraction_result_custom_file_type(self):
        """save_extraction_result accepts custom file_type."""
        result = save_extraction_result(
            user=self.user,
            filename='document.docx',
            file_size=1024,
            extracted_text='Word content.',
            file_type='DOCX'
        )

        self.assertEqual(result.file_type, 'DOCX')

    def test_save_extraction_result_without_tokens(self):
        """save_extraction_result works without token data."""
        result = save_extraction_result(
            user=self.user,
            filename='no_tokens.pdf',
            file_size=256,
            extracted_text='Content without token tracking.'
        )

        self.assertIsNone(result.prompt_tokens)
        self.assertIsNone(result.completion_tokens)
        self.assertIsNone(result.total_tokens)

    def test_save_extraction_result_returns_model_instance(self):
        """save_extraction_result returns the created model instance."""
        result = save_extraction_result(
            user=self.user,
            filename='return_test.pdf',
            file_size=100,
            extracted_text='Test.'
        )

        self.assertIsNotNone(result.id)
        self.assertIsNotNone(result.created_at)

    def test_save_multiple_extractions_for_same_user(self):
        """Multiple extractions can be saved for the same user."""
        save_extraction_result(
            user=self.user,
            filename='first.pdf',
            file_size=100,
            extracted_text='First extraction.'
        )
        save_extraction_result(
            user=self.user,
            filename='second.pdf',
            file_size=200,
            extracted_text='Second extraction.'
        )

        self.assertEqual(ExtractionResult.objects.filter(user=self.user).count(), 2)

    def test_save_extraction_result_used_caching_defaults_to_false(self):
        """save_extraction_result defaults used_caching to False."""
        result = save_extraction_result(
            user=self.user,
            filename="no_cache.pdf",
            file_size=1024,
            extracted_text="Small PDF extraction.",
        )

        self.assertFalse(result.used_caching)

    def test_save_extraction_result_with_used_caching_true(self):
        """save_extraction_result saves used_caching when True."""
        result = save_extraction_result(
            user=self.user,
            filename="large_cached.pdf",
            file_size=10240,
            extracted_text="Large PDF extraction with caching.",
            used_caching=True,
        )

        self.assertTrue(result.used_caching)

    def test_save_extraction_result_model_name_defaults_to_none(self):
        """save_extraction_result defaults model_name to None."""
        result = save_extraction_result(
            user=self.user,
            filename="no_model.pdf",
            file_size=1024,
            extracted_text="Extraction without model name.",
        )

        self.assertIsNone(result.model_name)

    def test_save_extraction_result_with_model_name(self):
        """save_extraction_result saves model_name when provided."""
        result = save_extraction_result(
            user=self.user,
            filename="gemini.pdf",
            file_size=2048,
            extracted_text="Extracted with Gemini model.",
            model_name="gemini-2.5-flash",
        )

        self.assertEqual(result.model_name, "gemini-2.5-flash")
