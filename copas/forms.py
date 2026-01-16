"""
Forms for Copas application.
"""
from django import forms


class PDFUploadForm(forms.Form):
    """Form for uploading PDF files for text extraction."""

    pdf_file = forms.FileField(
        label="Select PDF File",
        help_text="Maximum file size: 10MB. Only PDF files are accepted.",
        widget=forms.FileInput(attrs={
            'accept': 'application/pdf,.pdf',
            'class': 'form-control',
        })
    )

    # Maximum file size: 10MB
    MAX_FILE_SIZE = 10 * 1024 * 1024

    def clean_pdf_file(self):
        """Validate uploaded file is a PDF and within size limits."""
        pdf_file = self.cleaned_data.get('pdf_file')

        if pdf_file:
            # Check file size
            if pdf_file.size > self.MAX_FILE_SIZE:
                raise forms.ValidationError(
                    f"File size exceeds 10MB limit. Your file is {pdf_file.size / (1024*1024):.1f}MB."
                )

            # Check content type
            if pdf_file.content_type != 'application/pdf':
                raise forms.ValidationError(
                    "Invalid file type. Please upload a PDF file."
                )

            # Check file extension
            if not pdf_file.name.lower().endswith('.pdf'):
                raise forms.ValidationError(
                    "Invalid file extension. Please upload a .pdf file."
                )

            # Check magic bytes
            pdf_file.seek(0)
            header = pdf_file.read(5)
            pdf_file.seek(0)
            if header != b'%PDF-':
                raise forms.ValidationError(
                    "Invalid PDF file. The file does not appear to be a valid PDF."
                )

        return pdf_file
