from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.contrib import messages

from .forms import PDFUploadForm
from .services import extract_text_from_pdf


@login_required
def index(request):
    return render(request, 'copas/index.html')


@login_required
def pdf_extract_view(request):
    """
    Handle PDF upload and text extraction.

    GET: Display upload form
    POST: Process uploaded PDF and display extracted text
    """
    result = None
    filename = None

    if request.method == 'POST':
        form = PDFUploadForm(request.POST, request.FILES)

        if form.is_valid():
            uploaded_file = form.cleaned_data['pdf_file']
            filename = uploaded_file.name

            # Extract text using service layer
            result = extract_text_from_pdf(uploaded_file)

            if result.success:
                messages.success(request, 'Text extracted successfully!')
            else:
                messages.error(request, 'Failed to extract text from PDF.')
    else:
        form = PDFUploadForm()

    return render(request, 'copas/pdf_extract.html', {
        'form': form,
        'result': result,
        'filename': filename,
    })
