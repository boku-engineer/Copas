from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.contrib import messages

from .forms import PDFUploadForm
from .services import extract_text_from_pdf, save_extraction_result


@login_required
def index(request):
    """
    Home page with PDF upload and text extraction.

    GET: Display upload form
    POST: Process uploaded PDF, save to database, and display extracted text
    """
    result = None
    filename = None
    file_size = None

    if request.method == 'POST':
        form = PDFUploadForm(request.POST, request.FILES)

        if form.is_valid():
            uploaded_file = form.cleaned_data['pdf_file']
            filename = uploaded_file.name
            file_size = uploaded_file.size

            # Extract text using service layer
            result = extract_text_from_pdf(uploaded_file)

            if result.success:
                # Save extraction result to database
                save_extraction_result(
                    user=request.user,
                    filename=filename,
                    file_size=file_size,
                    extracted_text=result.text,
                    prompt_tokens=result.prompt_tokens,
                    completion_tokens=result.completion_tokens,
                    total_tokens=result.total_tokens,
                    used_caching=result.used_caching,
                    model_name=result.model_name,
                )

                # Inform user about caching if used
                if result.used_caching:
                    messages.info(
                        request,
                        f'Context caching enabled for {result.page_count}-page PDF. '
                        'Extracted page by page for optimal performance.'
                    )
                messages.success(request, 'Text extracted and saved successfully!')
            else:
                messages.error(request, 'Failed to extract text from PDF.')
    else:
        form = PDFUploadForm()

    return render(request, 'copas/pdf_extract.html', {
        'form': form,
        'result': result,
        'filename': filename,
    })
