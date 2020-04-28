from django.apps import AppConfig
import os
import pickle
from django.conf import settings


class PredictorConfig(AppConfig):
    path = os.path.join(settings.MODELS, 'finalized_model.p')

    with open(path, 'rb') as pickled:
        data = pickle.load(pickled)

    classifier = data['classifier']
    vectorizer = data['vectorizer']
