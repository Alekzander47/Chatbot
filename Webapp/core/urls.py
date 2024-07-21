from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from django.contrib.auth import views as auth_views
from . import views

urlpatterns = [
    path('', views.index, name='index'),   
    path('chat/', views.index, name='chat'),
    path('upload-files/', views.upload_files, name='upload_files'),
    path ('signup/' ,views.signup, name='signup'),
    path('logout/', views.logout_view, name='logout'),
    path('accounts/', include('allauth.urls')), # Include allauth URLs
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
