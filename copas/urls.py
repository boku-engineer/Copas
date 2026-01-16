from django.urls import path
from . import views

app_name = 'copas'

urlpatterns = [
    path('', views.index, name='index'),
    path('pdf-extract/', views.pdf_extract_view, name='pdf_extract'),
]
