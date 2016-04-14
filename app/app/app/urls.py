from django.conf.urls import include, url
from django.contrib import admin

urlpatterns = [
    url(r'^polls/', include('poemapp.urls')),
    url(r'^admin/', admin.site.urls),
]