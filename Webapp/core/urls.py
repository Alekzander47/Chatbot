from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('login', views.login, name='login'),
    path('signup', views.signup, name='signup'),
    path('chat/', views.index, name='chat'),
    path('upload-files/', views.upload_files, name='upload_files'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
