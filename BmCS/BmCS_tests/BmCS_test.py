"""
Module for verification of performance of system on test or validation data
"""

import json
import argparse
from math import isclose
from sklearn.metrics import classification_report, precision_score, recall_score
import datetime
import pickle
import sys

from ..model_utils import *
from ..preprocess_CNN_data import get_batch_data
from .preprocess_voting_data_test import preprocess_data
from ..thresholds import *


def parse_test_citations(XML_path, journal_drop, misindexed_ids):
    """
    Parse the test citations from json
    """
    
    with open(XML_path) as f:
        citations = json.load(f) 

    if journal_drop:
        citations = [citation for citation in citations if citation['journal_nlmid'] not in misindexed_ids]

    return citations


def evaluate_individual_models(cnn_predictions, voting_predictions, labels, group_thresh, journal_ids, group_ids):
    """
    Evaluate the performance of each model individuall 
    
    For the the CNN and voting ensemble, using the validation or test data sets, calculate metrics. 
    Returns precision and recall for each model.
    Labels 1 and 0 are used here, which do not correspond to prediction output. 
    """

    if not group_thresh:
        adj_voting_preds = [1 if y >= VOTING_THRESH else 0 for y in voting_predictions]
        adj_cnn_preds = [1 if y >= CNN_THRESH else 0 for y in cnn_predictions]
    # Performance doesn't necessarily improve as expected, as validation thresholds don't apply to test set.
    else:
        adj_voting_preds = []
        adj_cnn_preds = []
        for cnn_prob, voting_prob, journal_id in zip(cnn_predictions, voting_predictions, journal_ids):
            if journal_id in group_ids['science']:
                adj_voting_preds.append(1 if voting_prob >= VOTING_SCIENCE_THRESH else 0)
                adj_cnn_preds.append(1 if cnn_prob >= CNN_SCIENCE_THRESH else 0)
            elif journal_id in group_ids['jurisprudence']:
                adj_voting_preds.append(1 if voting_prob >= VOTING_JURISPRUDENCE_THRESH else 0)
                adj_cnn_preds.append(1 if cnn_prob >= CNN_JURISPRUDENCE_THRESH else 0)
            else:
                adj_voting_preds.append(1 if voting_prob >= VOTING_THRESH else 0)
                adj_cnn_preds.append(1 if cnn_prob >= CNN_THRESH else 0)

    voting_precision = precision_score(labels, adj_voting_preds)
    voting_recall = recall_score(labels, adj_voting_preds)
    cnn_precision = precision_score(labels, adj_cnn_preds)
    cnn_recall = recall_score(labels, adj_cnn_preds)

    return cnn_recall, cnn_precision, voting_recall, voting_precision


def BmCS_test_main(
        dataset, journal_ids_path, word_indicies_path, 
        group_thresh, journal_drop, destination, group_ids, misindexed_ids, args):
    """
    Main function for testing models on datasets
    Run voting and CNN, combine results,
    adjust decision threshold, and make new predictions
    """

    if dataset == "validation":
        XML_path = resource_filename(__name__, "datasets/pipeline_validation_set.json")
    else:
        XML_path = resource_filename(__name__, "datasets/pipeline_test_set.json")
   
    citations = parse_test_citations(XML_path, journal_drop, misindexed_ids) 
    voting_citations, journal_ids, labels = preprocess_data(citations)
    voting_predictions = run_voting(args.ensemble_path, voting_citations)
    CNN_citations = get_batch_data(citations, journal_ids_path, word_indicies_path)
    cnn_predictions = run_CNN(args.CNN_path, CNN_citations)
