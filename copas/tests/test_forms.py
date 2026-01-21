"""
Tests for Copas forms.
"""

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase

from copas.forms import PDFUploadForm


class PDFUploadFormTests(TestCase):
    """Tests for the PDF upload form."""

    def test_valid_pdf_accepted(self):
        """Valid PDF file should pass validation."""
        pdf_content = b"%PDF-1.4 valid pdf content"
        pdf_file = SimpleUploadedFile("test.pdf", pdf_content, content_type="application/pdf")

        form = PDFUploadForm(data={}, files={"pdf_file": pdf_file})
        self.assertTrue(form.is_valid())

    def test_non_pdf_content_type_rejected(self):
        """Files with non-PDF content type should be rejected."""
        text_file = SimpleUploadedFile(
            "document.pdf", b"%PDF-1.4 content", content_type="text/plain"
        )

        form = PDFUploadForm(data={}, files={"pdf_file": text_file})
        self.assertFalse(form.is_valid())
        self.assertIn("pdf_file", form.errors)

    def test_wrong_extension_rejected(self):
        """Files without .pdf extension should be rejected."""
        wrong_ext = SimpleUploadedFile(
            "document.txt", b"%PDF-1.4 content", content_type="application/pdf"
        )

        form = PDFUploadForm(data={}, files={"pdf_file": wrong_ext})
        self.assertFalse(form.is_valid())
        self.assertIn("pdf_file", form.errors)

    def test_invalid_magic_bytes_rejected(self):
        """Files without PDF magic bytes should be rejected."""
        fake_pdf = SimpleUploadedFile(
            "fake.pdf", b"This is not a real PDF file", content_type="application/pdf"
        )

        form = PDFUploadForm(data={}, files={"pdf_file": fake_pdf})
        self.assertFalse(form.is_valid())
        self.assertIn("pdf_file", form.errors)

    def test_file_too_large_rejected(self):
        """Files over 10MB should be rejected."""
        # Create content larger than 10MB
        large_content = b"%PDF-1.4" + (b"x" * (11 * 1024 * 1024))
        large_file = SimpleUploadedFile("large.pdf", large_content, content_type="application/pdf")

        form = PDFUploadForm(data={}, files={"pdf_file": large_file})
        self.assertFalse(form.is_valid())
        self.assertIn("exceeds", str(form.errors["pdf_file"]))

    def test_no_file_submitted(self):
        """Form should be invalid if no file is submitted."""
        form = PDFUploadForm(data={}, files={})
        self.assertFalse(form.is_valid())
        self.assertIn("pdf_file", form.errors)
