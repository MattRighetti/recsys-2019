#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 21/10/2018

@author: Maurizio Ferrari Dacrema
"""

import numpy as np
import scipy.sparse as sps
from tqdm import tqdm


def precision(is_relevant, relevant_items):
    # is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

    precision_score = np.sum(is_relevant, dtype=np.float32) / len(is_relevant)

    return precision_score


def recall(is_relevant, relevant_items):
    # is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

    recall_score = np.sum(is_relevant, dtype=np.float32) / relevant_items.shape[0]

    return recall_score


def MAP(is_relevant, relevant_items):
    # is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

    # Cumulative sum: precision at 1, at 2, at 3 ...
    p_at_k = is_relevant * np.cumsum(is_relevant, dtype=np.float32) / (1 + np.arange(is_relevant.shape[0]))

    # Sum of average precision
    map_score = np.sum(p_at_k) / np.min([relevant_items.shape[0], is_relevant.shape[0]])

    return map_score

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def evaluate_algorithm(URM_test, recommender_object, at=10):
    cumulative_precision = 0.0
    cumulative_recall = 0.0
    cumulative_MAP = 0.0

    num_eval = 0

    URM_test = sps.csr_matrix(URM_test)

    n_users = URM_test.shape[0]

    printProgressBar(0, n_users, prefix='Evaluation:', suffix='Complete', length=50)
    for user_id in range(n_users):
        printProgressBar(user_id, n_users, prefix = 'Evaluation:', suffix = 'Complete', length = 50)


        start_pos = URM_test.indptr[user_id]
        end_pos = URM_test.indptr[user_id + 1]

        if end_pos - start_pos > 0:
            relevant_items = URM_test.indices[start_pos:end_pos]

            recommended_items = recommender_object.recommend(user_id, at)
            num_eval += 1

            is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

            cumulative_precision += precision(is_relevant, relevant_items)
            cumulative_recall += recall(is_relevant, relevant_items)
            cumulative_MAP += MAP(is_relevant, relevant_items)

    cumulative_precision /= num_eval
    cumulative_recall /= num_eval
    cumulative_MAP /= n_users

    #print("Recommender performance is: Precision = {:.4f}, Recall = {:.4f}, MAP = {:.4f}".format(
    # cumulative_precision, cumulative_recall, cumulative_MAP))

    result_dict = {
        "precision": cumulative_precision,
        "recall": cumulative_recall,
        "MAP": cumulative_MAP,
    }

    return result_dict

def evaluate_MAP(URM_test, recommender_object, at=10, verbose=False):
    cumulative_MAP = 0.0
    URM_test = sps.csr_matrix(URM_test)
    n_users = URM_test.shape[0]

    for user_id in tqdm(n_users):

        start_pos = URM_test.indptr[user_id]
        end_pos = URM_test.indptr[user_id + 1]

        if end_pos - start_pos > 0:
            relevant_items = URM_test.indices[start_pos:end_pos]

            recommended_items = recommender_object.recommend(user_id, at)

            is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

            cumulative_MAP += MAP(is_relevant, relevant_items)

    cumulative_MAP /= n_users

    return cumulative_MAP

def evaluate_MAP_target_users(URM_test, recommender_object, target_users, at=10, verbose=True):
    cumulative_MAP = 0.0
    num_eval = 0
    n_users = URM_test.shape[0]

    for user_id in tqdm(target_users):

        start_pos = URM_test.indptr[user_id]
        end_pos = URM_test.indptr[user_id + 1]

        if end_pos - start_pos > 0:
            relevant_items = URM_test.indices[start_pos:end_pos]

            recommended_items = recommender_object.recommend(user_id, at)
            num_eval += 1

            is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

            cumulative_MAP += MAP(is_relevant, relevant_items)

    cumulative_MAP /= n_users

    return cumulative_MAP