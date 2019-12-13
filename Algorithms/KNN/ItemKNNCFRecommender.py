#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 23/10/17

@author: Maurizio Ferrari Dacrema
"""
from Utils.Toolkit import get_data
from Algorithms.Base.Evaluation.Evaluator import EvaluatorHoldout
from Algorithms.Data_manager.Kaggle.KaggleDataReader import KaggleDataReader
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


    def fit(self, topK=50, shrink=100, similarity='cosine', normalize=True, feature_weighting = "none", **similarity_args):

        self.topK = topK
        self.shrink = shrink

        if feature_weighting not in self.FEATURE_WEIGHTING_VALUES:
            raise ValueError("Value for 'feature_weighting' not recognized. Acceptable values are {}, provided was '{}'".format(self.FEATURE_WEIGHTING_VALUES, feature_weighting))


        if feature_weighting == "BM25":
            self.URM_train = self.URM_train.astype(np.float32)
            self.URM_train = okapi_BM_25(self.URM_train.T).T
            self.URM_train = check_matrix(self.URM_train, 'csr')

        elif feature_weighting == "TF-IDF":
            self.URM_train = self.URM_train.astype(np.float32)
            self.URM_train = TF_IDF(self.URM_train.T).T
            self.URM_train = check_matrix(self.URM_train, 'csr')

        similarity = Compute_Similarity(self.URM_train, shrink=shrink, topK=topK, normalize=normalize, similarity=similarity, **similarity_args)


        self.W_sparse = similarity.compute_similarity()
        self.W_sparse = check_matrix(self.W_sparse, format='csr')

if __name__ == '__main__':

    itemCF = ItemKNNCFRecommender(get_data()['train'])
    itemCF.load_model("/Users/mattiarighetti/Developer/PycharmProjects/recsys/result_experiments/SKOPT_prova/", file_name="ItemKNNCFRecommender_tanimoto_best_model.zip")
    #itemCF.fit(29, 5, similarity='tanimoto', normalize=True, feature_weighting="none")
    itemCF.evaluate_MAP_target(get_data()['test'], get_data()['target_users'])
    #write_output(itemCF, get_data()['target_users'])