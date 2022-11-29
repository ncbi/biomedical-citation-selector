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
from ..preprocess_voting_data import preprocess_data
from ..thresholds import *
import gzip


def parse_test_citations(XML_path, journal_drop, misindexed_ids):
    """
    Parse the test citations from json
    """
    
    with gzip.open(XML_path, "rt", encoding="utf8") as f:
        citations = json.load(f) 

    if journal_drop:
        citations = [citation for citation in citations if citation['journal_nlmid'] not in misindexed_ids]

    return citations


def evaluate_individual_models(cnn_predictions, voting_predictions, labels, group_thresh, journal_ids, group_ids):
    """
    Evaluate the performance of each model individually
    
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
        XML_path = resource_filename(__name__, "datasets/validation_set.json.gz")
    else:
        XML_path = resource_filename(__name__, "datasets/test_set.json.gz")
   
    citations = parse_test_citations(XML_path, journal_drop, misindexed_ids)
    voting_citations, journal_ids, _ = preprocess_data(citations)
    labels = [c["is_indexed"] for c in citations]
    voting_predictions = run_voting(args.ensemble_path, voting_citations)
    CNN_citations = get_batch_data(citations, journal_ids_path, word_indicies_path)
    cnn_predictions = run_CNN(args.CNN_path, CNN_citations)
    combined_predictions = combine_predictions(voting_predictions, cnn_predictions)
    prediction_dict = {'predictions': combined_predictions, 'journal_ids': journal_ids}
    adjusted_predictions = adjust_thresholds(prediction_dict, group_ids, group_thresh) 
    adjusted_predictions = adjust_in_scope_predictions(adjusted_predictions, prediction_dict)

    cnn_recall, cnn_precision, voting_recall, voting_precision = evaluate_individual_models(cnn_predictions, voting_predictions, labels, group_thresh, journal_ids, group_ids)

    all_pos_pred = [1 if pred == 1 or pred == 2 else 0 for pred in adjusted_predictions]
    BmCS_recall = recall_score(labels, all_pos_pred, pos_label=1)
    BmCS_precision = precision_score(labels, all_pos_pred, pos_label=1)

    in_scope_preds = [1 if pred == 1 else 0 for pred in adjusted_predictions]
    BmCS_in_scope_recall = recall_score(labels, in_scope_preds, pos_label=1)
    BmCS_in_scope_precision = precision_score(labels, in_scope_preds, pos_label=1)

    results_path = "{}/BmCS_test_results.txt".format(destination)
    with open(results_path, "a") as f:
        f.write("\n\n")
        for arg in vars(args):
            f.write("{0}: {1}\n".format(arg, vars(args)[arg]))
        f.write("""BmCS recall: {0}\nBmCS precision: {1}\nVoting recall: {2}\nVoting precision: {3}\nCNN recall: {4}\nCNN precision: {5}\nBmCS in-scope recall: {6}\nBmCS in-scope precision: {7}\n""".format(
                BmCS_recall,
                BmCS_precision,
                voting_recall,
                voting_precision,
                cnn_recall,
                cnn_precision,
                BmCS_in_scope_recall,
                BmCS_in_scope_precision))

    # Values computed using generate_validation_vs_test_vs_group_thresholds.py, not included in this repository.
    if not group_thresh and not journal_drop:
        if dataset == "validation":
            assert isclose(BmCS_recall, 0.995079049314208, abs_tol=args.tolerance), "BmCS recall does not match expected value"
            assert isclose(BmCS_precision, 0.7191827468785471, abs_tol=args.tolerance), "BmCS precision does not match expected value"
            assert isclose(BmCS_in_scope_recall, 0.7309705789969637, abs_tol=args.tolerance), "BmCS in-scope recall does not match expected value"
            assert isclose(BmCS_in_scope_precision, 0.9700569681811866, abs_tol=args.tolerance), "BmCS in-scope precision does not match expected value"
            print("Assertions passed")
        else:
            assert isclose(BmCS_recall, 0.9940239043824701, abs_tol=args.tolerance), "BmCS recall does not match expected value"
            assert isclose(BmCS_precision, 0.7133666904932094, abs_tol=args.tolerance), "BmCS precision does not match expected value"
            assert isclose(BmCS_in_scope_recall, 0.723937583001328, abs_tol=args.tolerance), "BmCS in-scope recall does not match expected value"
            assert isclose(BmCS_in_scope_precision, 0.969218802089121, abs_tol=args.tolerance), "BmCS in-scope precision does not match expected value"
            print("Assertions passed")
