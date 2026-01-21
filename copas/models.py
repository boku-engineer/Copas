from django.conf import settings
from django.db import models


class ExtractionResult(models.Model):
    """Model to store PDF extraction results."""

    FILE_TYPE_CHOICES = [
        ("PDF", "PDF"),
        # Future: ('DOCX', 'Word Document'), ('IMAGE', 'Image'), etc.
    ]

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="extraction_results"
    )
    filename = models.CharField(max_length=255)
    file_type = models.CharField(max_length=50, choices=FILE_TYPE_CHOICES, default="PDF")
    file_size = models.IntegerField()  # bytes
    extracted_text = models.TextField()
    prompt_tokens = models.IntegerField(null=True, blank=True)
    completion_tokens = models.IntegerField(null=True, blank=True)
    total_tokens = models.IntegerField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "extraction_result"
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.filename} - {self.user.username}"

    @property
    def file_size_display(self):
        """Return human-readable file size."""
        if self.file_size < 1024:
            return f"{self.file_size} B"
        elif self.file_size < 1024 * 1024:
            return f"{self.file_size / 1024:.1f} KB"
        return f"{self.file_size / (1024 * 1024):.1f} MB"
