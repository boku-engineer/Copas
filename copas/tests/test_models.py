"""
Tests for Copas models.
"""

from django.test import TestCase

from accounts.models import CustomUser
from copas.models import ExtractionResult


class ExtractionResultModelTests(TestCase):
    """Tests for the ExtractionResult model."""

    def setUp(self):
        self.user = CustomUser.objects.create_user(
            username="testuser", email="test@example.com", password="testpass123"
        )

    def test_create_extraction_result(self):
        """ExtractionResult can be created with required fields."""
        extraction = ExtractionResult.objects.create(
            user=self.user,
            filename="test.pdf",
            file_size=1024,
            extracted_text="This is the extracted text.",
        )

        self.assertEqual(extraction.user, self.user)
        self.assertEqual(extraction.filename, "test.pdf")
        self.assertEqual(extraction.file_type, "PDF")  # default
        self.assertEqual(extraction.file_size, 1024)
        self.assertEqual(extraction.extracted_text, "This is the extracted text.")
        self.assertIsNone(extraction.prompt_tokens)
        self.assertIsNone(extraction.completion_tokens)
        self.assertIsNone(extraction.total_tokens)
        self.assertIsNotNone(extraction.created_at)

    def test_create_extraction_result_with_tokens(self):
        """ExtractionResult can be created with token usage data."""
        extraction = ExtractionResult.objects.create(
            user=self.user,
            filename="test.pdf",
            file_size=2048,
            extracted_text="Extracted content here.",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        )

        self.assertEqual(extraction.prompt_tokens, 100)
        self.assertEqual(extraction.completion_tokens, 50)
        self.assertEqual(extraction.total_tokens, 150)

    def test_str_representation(self):
        """String representation shows filename and username."""
        extraction = ExtractionResult.objects.create(
            user=self.user, filename="document.pdf", file_size=512, extracted_text="Some text."
        )

        self.assertEqual(str(extraction), "document.pdf - testuser")

    def test_file_size_display_bytes(self):
        """file_size_display returns bytes for small files."""
        extraction = ExtractionResult.objects.create(
            user=self.user, filename="tiny.pdf", file_size=500, extracted_text="Tiny file."
        )

        self.assertEqual(extraction.file_size_display, "500 B")

    def test_file_size_display_kilobytes(self):
        """file_size_display returns KB for medium files."""
        extraction = ExtractionResult.objects.create(
            user=self.user,
            filename="medium.pdf",
            file_size=5120,  # 5 KB
            extracted_text="Medium file.",
        )

        self.assertEqual(extraction.file_size_display, "5.0 KB")

    def test_file_size_display_megabytes(self):
        """file_size_display returns MB for large files."""
        extraction = ExtractionResult.objects.create(
            user=self.user,
            filename="large.pdf",
            file_size=2097152,  # 2 MB
            extracted_text="Large file.",
        )

        self.assertEqual(extraction.file_size_display, "2.0 MB")

    def test_ordering_by_created_at_descending(self):
        """ExtractionResults are ordered by created_at descending."""
        extraction1 = ExtractionResult.objects.create(
            user=self.user, filename="first.pdf", file_size=100, extracted_text="First."
        )
        extraction2 = ExtractionResult.objects.create(
            user=self.user, filename="second.pdf", file_size=200, extracted_text="Second."
        )

        extractions = list(ExtractionResult.objects.all())
        self.assertEqual(extractions[0], extraction2)
        self.assertEqual(extractions[1], extraction1)

    def test_cascade_delete_on_user_deletion(self):
        """ExtractionResults are deleted when user is deleted."""
        ExtractionResult.objects.create(
            user=self.user, filename="test.pdf", file_size=100, extracted_text="Test."
        )

        self.assertEqual(ExtractionResult.objects.count(), 1)
        self.user.delete()
        self.assertEqual(ExtractionResult.objects.count(), 0)

    def test_db_table_name(self):
        """Model uses custom table name 'extraction_result'."""
        self.assertEqual(ExtractionResult._meta.db_table, "extraction_result")
