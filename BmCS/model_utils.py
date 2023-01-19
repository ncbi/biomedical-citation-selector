"""
Module for model related functions

Module contains functions for running models and combining predictions,
adjusting in-scope predictions, and adjusting predictions for publication type.
"""

import os
import json

import numpy as np
from pathlib import Path
import time
from pkg_resources import resource_string, resource_filename

import tensorflow.keras.backend as K
from tensorflow.keras.models import model_from_json

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report

from .embedding_custom import EmbeddingWithDropout
from .thresholds import *
from . import item_select
from .publication_types import pub_strings, pub_types

from .bmcs_exceptions import BmCS_Exception


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
        return [2 if y > COMBINED_THRESH else 0 for y in predictions_dict['predictions']]

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
# -----------------------------------------------------------------------------------------------------------------------


def get_fname(fname, base, subst: str):

    if fname is None:
        fname = ""
        
    if base is None:
        base = ""
            
    if subst is None:
        subst = ""

    for file_name in [fname, os.path.join(base, fname), 
                             os.path.join(base, subst), 
                             resource_filename(__name__, subst)]:
        if os.path.isfile(file_name):
            return file_name

    # If we are here, file does not exist
    msg = "File can't be found. fname: \"{}\", base: \"{}\", subst: \"{}\".".format(fname, base, subst)
    raise BmCS_Exception(msg)
# -----------------------------------------------------------------------------------------------------------------------


def load_cfg(env: str, subst, base: str=None, is_file: bool=False, load: bool=False):
    value = None

    try:
        value = os.environ[env]
        value = None if len(value) == 0 else value
    except KeyError:
        # No such environment variable, let it be for now
        pass

    if is_file:
        # Get a file name (from environment or from hardcoded places)
        fname = get_fname(value, base, subst)

        if not load:
            return fname

        ids = None
        with open(fname) as f:
            ids = json.load(f)

        return ids
    else:
        if subst is not None:
            value = subst
        else:
            return None
        
        # This is just a environment variable
        if not isinstance(value, bool) and not isinstance(value, int):
            m = re.search(r'^\s*(\d+)\s*$', value)
            if m is not None:
                value = int(m.group(1))
            else:
                m = re.search(r'\s*(true|false)\s*$', value, re.I)
                if m is not None:
                    value = True if m.group(1).lower() == 'true' else False

    return value
# -----------------------------------------------------------------------------------------------------------------------
