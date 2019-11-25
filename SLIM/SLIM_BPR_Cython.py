#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 07/09/17

@author: Maurizio Ferrari Dacrema
"""

import subprocess
import os, sys

import numpy as np

from Algorithms.Base.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from Algorithms.Base.Recommender_utils import similarityMatrixTopK
from Algorithms.Notebooks_utils.evaluation_function import evaluate_MAP_target_users, evaluate_MAP
from Algorithms.Base.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping
from Utils.Toolkit import get_data


class SLIM_BPR_Cython(BaseItemSimilarityMatrixRecommender, Incremental_Training_Early_Stopping):

    RECOMMENDER_NAME = "SLIM_BPR_Recommender"

    def __init__(self, URM_train, recompile_cython=False):

        super().__init__(URM_train)
        self.URM_train = URM_train

        self.normalize = False

        if recompile_cython:
            print("Compiling in Cython")
            self.runCompilationScript()
            print("Compilation Complete")

    def fit(self, positive_threshold=None, final_model_sparse_weights=True,
                 train_with_sparse_weights=False, symmetric=True, epochs = 400,
                batch_size = 1, lambda_i = 0.6, lambda_j = 1, learning_rate = 1e-4, topK = 30,
                sgd_mode = 'sgd', gamma=0.995, beta_1=0.9, beta_2=0.999, **earlystopping_kwargs):

        if self.train_with_sparse_weights is not None:
            if self.train_with_sparse_weights:
                self.sparse_weights = True

        #### Retreiving parameters for fitting #######
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

        self.positive_threshold = positive_threshold

        self.train_with_sparse_weights = train_with_sparse_weights
        self.sparse_weights = final_model_sparse_weights

        ### Stuff to adapt code to general structure



        self.n_users = self.URM_train.shape[0]
        self.n_items = self.URM_train.shape[1]

        URM_train_positive = self.URM_train.copy()

        self.URM_mask = self.URM_train.copy()

        if self.positive_threshold is not None:
            self.URM_mask.data = self.URM_mask.data >= self.positive_threshold
            self.URM_mask.eliminate_zeros()

        assert self.URM_mask.nnz > 0, "MatrixFactorization_Cython: URM_train_positive is empty, positive threshold is too high"

        if not self.train_with_sparse_weights:

            n_items = self.URM_train.shape[1]
            requiredGB = 8 * n_items ** 2 / 1e+06

            if self.symmetric:
                requiredGB /= 2

            #print("SLIM_BPR_Cython: Estimated memory required for similarity matrix of {} items is {:.2f} MB".format(
                #n_items, requiredGB))


        #### Actual fitting from here
        if self.positive_threshold is not None:
            URM_train_positive.data = URM_train_positive.data >= self.positive_threshold
            URM_train_positive.eliminate_zeros()

        from SLIM.SLIM_BPR_Cython_Epoch import SLIM_BPR_Cython_Epoch
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
        self.W_sparse = None

        self._train_with_early_stopping(self.epochs,
                                        algorithm_name=self.RECOMMENDER_NAME,
                                        **earlystopping_kwargs)

        self.get_S_incremental_and_set_W()

        self.cythonEpoch._dealloc()

        sys.stdout.flush()

    def _prepare_model_for_validation(self):
        self.get_S_incremental_and_set_W()

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

    def runCompilationScript(self):

        # Run compile script setting the working directory to ensure the compiled file are contained in the
        # appropriate subfolder and not the project root

        compiledModuleSubfolder = ""
        # fileToCompile_list = ['Sparse_Matrix_CSR.pyx', 'SLIM_BPR_Cython_Epoch.pyx']
        fileToCompile_list = ['SLIM_BPR_Cython_Epoch.pyx']

        for fileToCompile in fileToCompile_list:

            command = ['python',
                       'compileCython.py',
                       fileToCompile,
                       'build_ext',
                       '--inplace'
                       ]

            output = subprocess.check_output(' '.join(command), shell=True, cwd=os.getcwd() + compiledModuleSubfolder)

            try:

                command = ['cython',
                           fileToCompile,
                           '-a'
                           ]

                output = subprocess.check_output(' '.join(command), shell=True,
                                                 cwd=os.getcwd() + compiledModuleSubfolder)

            except:
                pass

        print("Compiled module saved in subfolder: {}".format(compiledModuleSubfolder))

        # Command to run compilation script
        # python compileCython.py SLIM_BPR_Cython_Epoch.pyx build_ext --inplace

        # Command to generate html report
        # cython -a SLIM_BPR_Cython_Epoch.pyx

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

    def evaluate_MAP(self, URM_test, args_used):
        result = evaluate_MAP(URM_test, self)
        print("SLIM_BPR -> MAP: {:.4f} with data {}".format(result, args_used))
        return result

    def evaluate_MAP_target(self, URM_test, target_user_list, args_used):
        result = evaluate_MAP_target_users(URM_test, self, target_user_list)
        print("SLIM_BPR -> MAP: {:.4f} with data {}".format(result, args_used))
        return result

################################################ Test ##################################################
# max_map = 0
# data = get_data(test=True)
#
# args = {
#     'topK': 25,
#     'lambda_i': 0.03,
#     'lambda_j': 0.9,
#     'epochs': 3800,
#     'learning_rate' : 1e-3,
#     'sgd_mode' : 'adagrad'
# }
#
# # These are set by default (no need to add them)
# advanced = {
#     'train_with_sparse_weights' : False,
#     'final_model_sparse_weights' : True,
#     'batch_size' : 1000,
#     'symmetric' : True,
#     'gamma' : 0.995,
#     'beta_1' : 0.9,
#     'beta_2' : 0.999,
#     'positive_threshold' : 1
# }
#
# recommender = SLIM_BPR_Cython(epochs=args['epochs'],
#                               topK=args['topK'],
#                               lambda_i=args['lambda_i'],
#                               lambda_j=args['lambda_j'],
#                               positive_threshold=advanced['positive_threshold'],
#                               sgd_mode=args['sgd_mode'],
#                               learning_rate=args['learning_rate'],
#                               batch_size=advanced['batch_size'],
#                               train_with_sparse_weights=advanced['train_with_sparse_weights'])
#
# #recommender.runCompilationScript()
# recommender.fit(data['train'])
# recommender.evaluate_MAP_target(data['test'], data['target_users'], args)
################################################ Test ##################################################