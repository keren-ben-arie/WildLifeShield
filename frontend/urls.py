from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path("", include('ContentBlocker.urls')),
    path('admin/', admin.site.urls)
]