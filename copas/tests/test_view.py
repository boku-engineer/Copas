"""
Tests for Copas views.
"""
from django.test import TestCase, Client
from django.urls import reverse
from django.core.files.uploadedfile import SimpleUploadedFile
from unittest.mock import patch

from accounts.models import CustomUser
from core.gemini_client import ExtractionResult


class IndexViewTests(TestCase):
    """Tests for the index view (PDF extraction home page)."""

    def setUp(self):
        self.client = Client()
        self.user = CustomUser.objects.create_user(
            username='testuser',
            email='test@example.com',
            password='testpass123'
        )
        self.url = reverse('copas:index')

    def test_index_requires_login(self):
        """Unauthenticated users should be redirected to login."""
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 302)
        self.assertIn('login', response.url)

    def test_index_loads_for_authenticated_user(self):
        """Authenticated users should see the PDF extraction form."""
        self.client.login(username='testuser', password='testpass123')
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'PDF Text Extraction')
        self.assertContains(response, 'Upload')


class PDFExtractFunctionalityTests(TestCase):
    """Tests for PDF extraction functionality on the index page."""

    def setUp(self):
        self.client = Client()
        self.user = CustomUser.objects.create_user(
            username='testuser',
            email='test@example.com',
            password='testpass123'
        )
        self.url = reverse('copas:index')

    def test_invalid_file_type_rejected(self):
        """Non-PDF files should be rejected with error."""
        self.client.login(username='testuser', password='testpass123')

        text_file = SimpleUploadedFile(
            "document.txt",
            b"This is a text file",
            content_type="text/plain"
        )

        response = self.client.post(self.url, {'pdf_file': text_file})

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Invalid')

    @patch('copas.views.extract_text_from_pdf')
    def test_successful_extraction(self, mock_extract):
        """Valid PDF should return extracted text."""
        mock_extract.return_value = ExtractionResult(
            success=True,
            text='This is the extracted text from the PDF.'
        )

        self.client.login(username='testuser', password='testpass123')

        pdf_content = b'%PDF-1.4 fake pdf content'
        pdf_file = SimpleUploadedFile(
            "test.pdf",
            pdf_content,
            content_type="application/pdf"
        )

        response = self.client.post(self.url, {'pdf_file': pdf_file})

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Extracted Text')
        self.assertContains(response, 'This is the extracted text')

    @patch('copas.views.extract_text_from_pdf')
    def test_extraction_failure_shows_error(self, mock_extract):
        """Failed extraction should show error message."""
        mock_extract.return_value = ExtractionResult(
            success=False,
            error='API connection failed'
        )

        self.client.login(username='testuser', password='testpass123')

        pdf_content = b'%PDF-1.4 fake pdf content'
        pdf_file = SimpleUploadedFile(
            "test.pdf",
            pdf_content,
            content_type="application/pdf"
        )

        response = self.client.post(self.url, {'pdf_file': pdf_file})

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'API connection failed')
