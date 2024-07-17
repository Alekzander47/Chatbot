from django.contrib import admin
from django.urls import path
from . import views


urlpatterns = [
    path('',views.index,name='index'),
    path('login',views.login, name='login'),
    path('signup', views.signup, name='signup'),
    path('ask-question/', views.ask_question, name='ask_question'),
    
]