#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 23/10/17

@author: Maurizio Ferrari Dacrema
"""
from Utils.Toolkit import get_data
from Algorithms.Base.Evaluation.Evaluator import EvaluatorHoldout
from Algorithms.Data_manager.Split_functions.split_train_validation_leave_k_out import split_train_leave_k_out_user_wise
from Algorithms.Data_manager.DataSplitter_leave_k_out import DataSplitter_leave_k_out
from Utils.OutputWriter import write_output

from Algorithms.Base.Recommender_utils import check_matrix
from Algorithms.Base.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender

from Algorithms.Base.IR_feature_weighting import okapi_BM_25, TF_IDF
import numpy as np

from Algorithms.Base.Similarity.Compute_Similarity import Compute_Similarity


class ItemKNNCFRecommender(BaseItemSimilarityMatrixRecommender):
    """ ItemKNN recommender"""

    RECOMMENDER_NAME = "ItemKNNCFRecommender"

    FEATURE_WEIGHTING_VALUES = ["BM25", "TF-IDF", "none"]



    def __init__(self, URM_train, verbose = True):
        super(ItemKNNCFRecommender, self).__init__(URM_train, verbose = verbose)


    def fit(self, topK=50, shrink=100, similarity='cosine', normalize=True, feature_weighting = "none",
            tversky_alpha = 0.0, tversky_beta = 0.0, asymmetric_alpha = 0.0, **similarity_args):

        self.topK = topK
        self.shrink = shrink

        if feature_weighting not in self.FEATURE_WEIGHTING_VALUES:
            raise ValueError("Value for 'feature_weighting' not recognized. Acceptable values are {}, provided was '{}'".format(self.FEATURE_WEIGHTING_VALUES, feature_weighting))


        if feature_weighting == "BM25":
            self.URM_train = self.URM_train.astype(np.float64)
            self.URM_train = okapi_BM_25(self.URM_train.T).T
            self.URM_train = check_matrix(self.URM_train, 'csr')

        elif feature_weighting == "TF-IDF":
            self.URM_train = self.URM_train.astype(np.float64)
            self.URM_train = TF_IDF(self.URM_train.T).T
            self.URM_train = check_matrix(self.URM_train, 'csr')

        similarity = Compute_Similarity(self.URM_train,
                                        shrink=shrink,
                                        topK=topK,
                                        normalize=normalize,
                                        similarity=similarity,
                                        tversky_alpha=tversky_alpha,
                                        tversky_beta=tversky_beta,
                                        asymmetric_alpha=asymmetric_alpha,
                                        **similarity_args)


        self.W_sparse = similarity.compute_similarity()
        self.W_sparse = check_matrix(self.W_sparse, format='csr')

if __name__ == '__main__':

    train, test = split_train_leave_k_out_user_wise(get_data()['URM_all'], k_out=1)

    evaluator = EvaluatorHoldout(test, [10], target_users=get_data()['target_users'])

    args = {
        'topK': 12,
        'shrink': 88,
        'similarity': 'tversky',
        'normalize': True,
        'tversky_alpha': 0.12331166243379268,
        'tversky_beta': 1.9752288743799558
    }

    itemCF = ItemKNNCFRecommender(train)
    # itemCF.load_model("/Users/mattiarighetti/Developer/PycharmProjects/recsys/result_experiments/SKOPT_prova/", file_name="ItemKNNCFRecommender_tversky_best_model.zip")
    itemCF.fit(12, 88, similarity='tversky', normalize=True,
               feature_weighting='none', tversky_alpha=args['tversky_alpha'], tversky_beta=args['tversky_beta'])
    result, result_string = evaluator.evaluateRecommender(itemCF)
    # write_output(itemCF, get_data()['target_users'])
    print(f"MAP: {result[10]['MAP']:.5f}")