from django.shortcuts import render

from predictor.apps import PredictorConfig
from .models import Person
from .serializers import PersonSerializer
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
import numpy as np


@api_view(['GET', 'POST', 'PUT'])
def user_data(request):

    if request.method == 'GET':
        persons = Person.objects.all()
        serializer = PersonSerializer(persons, many=True)
        return Response(serializer.data)

    elif request.method == 'POST':
        serializer = PersonSerializer(data=request.data)

        if serializer.is_valid():
            serializer.save()
            return Response(status=status.HTTP_201_CREATED)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(['POST'])
def predict(request):

    if request.method == 'POST':
        serializer = PersonSerializer(data=request.data)

        data = request.data

        pre_data = [int(data['age']), int(data['salary'])]

        unit = np.array(pre_data)
        unit = unit.reshape(1, -1)

        X = PredictorConfig.vectorizer.transform(unit)

        y_pred = PredictorConfig.classifier.predict(X)

        if serializer.is_valid():
            return Response(y_pred == 1, status=status.HTTP_200_OK)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
