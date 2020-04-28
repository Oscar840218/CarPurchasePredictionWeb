from django.urls import path
from .views import predict, user_data

urlpatterns = [
    path('predict', predict),
    path('user_data', user_data)
]