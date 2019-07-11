"""
Module for model related functions

Module contains functions for running models and combining predictions, 
adjusting in-scope predictions, and adjusting predictions for publication type. 
"""

import numpy as np
from pathlib import Path
import time
from pkg_resources import resource_string, resource_filename

import keras.backend as K
from keras.models import model_from_json

from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report

from .embedding_custom import EmbeddingWithDropout
from .thresholds import *
from . import item_select
from .publication_types import pub_strings, pub_types


def run_CNN(CNN_path, X):
    """
    load the CNN and return predictions
    """

    print("Making CNN predictions")
    model_path = resource_filename(__name__, "models/model_CNN.json")

    with open(model_path, 'rt') as model_json_file:
        model_json = model_json_file.read()

    model = model_from_json(model_json, custom_objects={EmbeddingWithDropout.__name__: EmbeddingWithDropout})
    model.load_weights(CNN_path)
    model.compile(loss='binary_crossentropy', optimizer='adam')
    result = model.predict(X).flatten()

    return result


def run_voting(ensemble_path, X):
    """
    Run the voting model
    """

    print("Making ensemble predictions")
    model = joblib.load(ensemble_path)
    y_probs = model.predict_proba(X)[:, 0]

    return y_probs


def adjust_thresholds(predictions_dict, group_ids, group_thresh):
    """
    Adjust threshold depending on the category of the journal 
    By default, no group thresholding will be performed. 
    2 is the label indicating uncertainty of in-scope status.
    0 is the label indicating certainty of out-of-scope prediction.
    """

    print("Combining predictions")
    
    if not group_thresh:
        return [2 if y >= COMBINED_THRESH else 0 for y in predictions_dict['predictions']]
    
    else:
        predictions = []
        for y, journal_id in zip(predictions_dict['predictions'], predictions_dict['journal_ids']):
            if journal_id in group_ids['science']:
                predictions.append(2 if y>= SCIENCE_THRESH else 0)
            elif journal_id in group_ids['jurisprudence']:
                predictions.append(2 if y>= JURISPRUDENCE_THRESH else 0)
            else:
                predictions.append(2 if y>= COMBINED_THRESH else 0)
        return predictions


def adjust_in_scope_predictions(predictions, predictions_dict):
    """
    For the predictions that fall above the threshold,
    mark them as in-scope without review.
    1 is the label indicating confidence in in-scope prediction.
    """

    for i, prob in enumerate(predictions_dict['predictions']):
        # If prob is greater than threshold, mark for automatic selection
        if prob > PRECISION_THRESH:
            predictions[i] = 1

    return predictions


def combine_predictions(voting_predictions, cnn_predictions):
    """
    Combine the predictions of the two models
    """
    return voting_predictions*cnn_predictions


def filter_pub_type(citations, predictions):
    """
    Filter citations based on the pubtype

    This is indicated in two places: 
    Either in the title, where there will be 
    string, usually at the beginning of the title,
    or at PubType status that in the xml itself. 
    """
    
    for i, citation in enumerate(citations):
        # Check the title for the strings that indicate the citation should sent to indexers for review
        if any(pub_string in citation['title'].lower() for pub_string in pub_strings):
            predictions[i] = 3
        # Check the list of pub types as well
        elif any(pub_type in citation['pub_type'] for pub_type in pub_types):
            predictions[i] = 3

    return predictions
