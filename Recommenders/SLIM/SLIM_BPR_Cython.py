#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 07/09/17

@author: Maurizio Ferrari Dacrema
"""

import subprocess
import os, sys

import numpy as np

from Algorithms.Base.Recommender_utils import similarityMatrixTopK
from Algorithms.Notebooks_utils.evaluation_function import evaluate_MAP_target_users
from Utils.Toolkit import get_data, feature_boost_URM
from Recommenders.BaseRecommender import BaseRecommender


class SLIM_BPR_Cython(BaseRecommender):

    def __init__(self, positive_threshold=None, final_model_sparse_weights=True,
                 train_with_sparse_weights=False, symmetric=True, epochs=400, batch_size=1000, lambda_i=0.6, lambda_j=1,
                 learning_rate=1e-4, topK=30, sgd_mode='sgd', gamma=0.995, beta_1=0.9, beta_2=0.999):

        #### Retreiving parameters for fitting #######
        super().__init__()
        self.epochs = epochs
        self.batch_size = batch_size
        self.lambda_i = lambda_i
        self.lambda_j = lambda_j
        self.learning_rate = learning_rate
        self.topK = topK
        self.sgd_mode = sgd_mode
        self.gamma = gamma
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.symmetric = symmetric
        #############################################

        self.normalize = False
        self.positive_threshold = positive_threshold

        self.train_with_sparse_weights = train_with_sparse_weights
        self.sparse_weights = final_model_sparse_weights


        if self.train_with_sparse_weights:
            self.sparse_weights = True

    def fit(self, URM_train, boost=False):

        ### Stuff to adapt code to general structure

        self.URM_train = URM_train

        if boost:
            self.URM_train = feature_boost_URM(self.URM_train.copy(), 10, min_interactions=40, kind="subclass")
            # self.URM_train = feature_boost_URM(self.URM_train.copy(), 5, min_interactions=3, kind="asset")
            # self.URM_train = feature_boost_URM(self.URM_train.copy(), 5, min_interactions=3, kind="price")
            # self.URM_train = normalize_matrix(self.URM_train, axis=1)

        self.n_users = URM_train.shape[0]
        self.n_items = URM_train.shape[1]

        URM_train_positive = self.URM_train.copy()

        self.URM_mask = self.URM_train.copy()

        if self.positive_threshold is not None:
            self.URM_mask.data = self.URM_mask.data >= self.positive_threshold
            self.URM_mask.eliminate_zeros()

        assert self.URM_mask.nnz > 0, "MatrixFactorization_Cython: URM_train_positive is empty, positive threshold is too high"

        if not self.train_with_sparse_weights:

            n_items = URM_train.shape[1]
            requiredGB = 8 * n_items ** 2 / 1e+06

            if self.symmetric:
                requiredGB /= 2

            #print("SLIM_BPR_Cython: Estimated memory required for similarity matrix of {} items is {:.2f} MB".format(
                #n_items, requiredGB))


        #### Actual fitting from here
        if self.positive_threshold is not None:
            URM_train_positive.data = URM_train_positive.data >= self.positive_threshold
            URM_train_positive.eliminate_zeros()

        from Recommenders.SLIM.SLIM_BPR_Cython_Epoch import SLIM_BPR_Cython_Epoch

        self.cythonEpoch = SLIM_BPR_Cython_Epoch(self.URM_mask,
                                                 train_with_sparse_weights=self.train_with_sparse_weights,
                                                 final_model_sparse_weights=self.sparse_weights,
                                                 topK=self.topK,
                                                 learning_rate=self.learning_rate,
                                                 li_reg=self.lambda_i,
                                                 lj_reg=self.lambda_j,
                                                 batch_size=self.batch_size,
                                                 symmetric=self.symmetric,
                                                 sgd_mode=self.sgd_mode,
                                                 gamma=self.gamma,
                                                 beta_1=self.beta_1,
                                                 beta_2=self.beta_2)


        self._initialize_incremental_model()
        self.epochs_best = 0
        currentEpoch = 0

        while currentEpoch < self.epochs:

            self._run_epoch()
            self._update_best_model()
            currentEpoch += 1

        self.get_S_incremental_and_set_W()

        sys.stdout.flush()

        self.RECS = self.URM_train.dot(self.W_sparse)
        self.W_sparse = None  # TODO ADDED TO save Memory, adjust

    def _initialize_incremental_model(self):
        self.S_incremental = self.cythonEpoch.get_S()
        self.S_best = self.S_incremental.copy()

    def _update_incremental_model(self):
        self.get_S_incremental_and_set_W()

    def _update_best_model(self):
        self.S_best = self.S_incremental.copy()

    def _run_epoch(self):
        self.cythonEpoch.epochIteration_Cython()

    def get_S_incremental_and_set_W(self):

        self.S_incremental = self.cythonEpoch.get_S()

        if self.train_with_sparse_weights:
            self.W_sparse = self.S_incremental
        else:
            if self.sparse_weights:
                self.W_sparse = similarityMatrixTopK(self.S_incremental, k=self.topK)
            else:
                self.W = self.S_incremental

    def get_expected_ratings(self, playlist_id):
        expected_ratings = self.RECS[playlist_id].todense()
        return np.squeeze(np.asarray(expected_ratings))

    def recommend(self, playlist_id, at=10):

        # compute the scores using the dot product
        scores = self.get_expected_ratings(playlist_id)
        ranking = scores.argsort()[::-1]
        unseen_items_mask = np.in1d(ranking, self.URM_train[playlist_id].indices, assume_unique=True, invert=True)
        ranking = ranking[unseen_items_mask]
        return ranking[:at]

################################################ Test ##################################################
# max_map = 0
# data = get_data()
#
# args = {
#     'topK': 85,
#     'lambda_i': 0.1,
#     'lambda_j': 1,
#     'epochs': 4000,
#     'learning_rate' : 1e-4,
#     'symmetric' : False,
#     'sgd_mode' : 'adagrad'
# }
#
# recommender = SLIM_BPR_Cython(epochs=args['epochs'],
#                               topK=args['topK'],
#                               lambda_i=args['lambda_i'],
#                               lambda_j=args['lambda_j'],
#                               positive_threshold=1,
#                               sgd_mode=args['sgd_mode'],
#                               symmetric=args['symmetric'],
#                               learning_rate=args['learning_rate'])
# recommender.fit(data['train'])
# recommender.evaluate_MAP_target(data['test'], data['target_users'])
################################################ Test ##################################################