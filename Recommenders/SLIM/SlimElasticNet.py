
import pathos.pools as pp
import multiprocessing
from functools import partial
import numpy as np
import scipy.sparse as sps
from sklearn.linear_model import ElasticNet

from Utils.Toolkit import get_data
from Algorithms.Base.Evaluation.Evaluator import EvaluatorHoldout
from Algorithms.Data_manager.Split_functions.split_train_validation_leave_k_out import split_train_leave_k_out_user_wise
from Algorithms.Data_manager.Kaggle.KaggleDataReader import KaggleDataReader
from Algorithms.Data_manager.DataSplitter_leave_k_out import DataSplitter_leave_k_out
from Utils.OutputWriter import write_output

import numpy as np
import scipy.sparse as sps
from Algorithms.Base.Recommender_utils import check_matrix
from sklearn.linear_model import ElasticNet
from sklearn.exceptions import ConvergenceWarning

from Algorithms.Base.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from Algorithms.Utils.seconds_to_biggest_unit import seconds_to_biggest_unit
import time, sys, warnings

from Utils.Toolkit import get_data
from Algorithms.Base.Evaluation.Evaluator import EvaluatorHoldout
from Algorithms.Data_manager.Split_functions.split_train_validation_leave_k_out import split_train_leave_k_out_user_wise
from Algorithms.Data_manager.Kaggle.KaggleDataReader import KaggleDataReader
from Algorithms.Data_manager.DataSplitter_leave_k_out import DataSplitter_leave_k_out
from Utils.OutputWriter import write_output

class SLIMElasticNetRecommender(object):
    """
    Train a Sparse Linear Methods (SLIM) item similarity model.
    NOTE: ElasticNet solver is parallel, a single intance of SLIM_ElasticNet will
          make use of half the cores available

    See:
        Efficient Top-N Recommendation by Linear Regression,
        M. Levy and K. Jack, LSRS workshop at RecSys 2013.
        https://www.slideshare.net/MarkLevy/efficient-slides

        SLIM: Sparse linear methods for top-n recommender systems,
        X. Ning and G. Karypis, ICDM 2011.
        http://glaros.dtc.umn.edu/gkhome/fetch/papers/SLIM2011icdm.pdf
    """
    def __init__(self, URM_train):
        self.URM_train = URM_train


    """ 
        Fit given to each pool thread, to fit the W_sparse 
    """

    def fit(self, l1_ratio=0.1, alpha=1.0, positive_only=True, topK=100, max_iter=40):

        self.l1_ratio = l1_ratio
        self.positive_only = positive_only
        self.topK = topK

        # Display ConvergenceWarning only once and not for every item it occurs
        warnings.simplefilter("once", category=ConvergenceWarning)

        # initialize the ElasticNet model
        self.model = ElasticNet(alpha=alpha,
                                l1_ratio=l1_ratio,
                                positive=positive_only,
                                fit_intercept=False,
                                copy_X=False,
                                precompute=True,
                                selection='random',
                                max_iter=max_iter,
                                tol=1e-4)

        URM_train = check_matrix(self.URM_train, 'csc', dtype=np.float32)

        n_items = URM_train.shape[1]

        # Use array as it reduces memory requirements compared to lists
        dataBlock = 10000000

        rows = np.zeros(dataBlock, dtype=np.int32)
        cols = np.zeros(dataBlock, dtype=np.int32)
        values = np.zeros(dataBlock, dtype=np.float32)

        numCells = 0

        start_time = time.time()
        start_time_printBatch = start_time

        # fit each item's factors sequentially (not in parallel)
        for currentItem in range(n_items):

            # get the target column
            y = URM_train[:, currentItem].toarray()

            # set the j-th column of X to zero
            start_pos = URM_train.indptr[currentItem]
            end_pos = URM_train.indptr[currentItem + 1]

            current_item_data_backup = URM_train.data[start_pos: end_pos].copy()
            URM_train.data[start_pos: end_pos] = 0.0

            # fit one ElasticNet model per column
            self.model.fit(URM_train, y)

            # self.model.coef_ contains the coefficient of the ElasticNet model
            # let's keep only the non-zero values

            # Select topK values
            # Sorting is done in three steps. Faster then plain np.argsort for higher number of items
            # - Partition the data to extract the set of relevant items
            # - Sort only the relevant items
            # - Get the original item index

            nonzero_model_coef_index = self.model.sparse_coef_.indices
            nonzero_model_coef_value = self.model.sparse_coef_.data

            local_topK = min(len(nonzero_model_coef_value) - 1, self.topK)

            relevant_items_partition = (-nonzero_model_coef_value).argpartition(local_topK)[0:local_topK]
            relevant_items_partition_sorting = np.argsort(-nonzero_model_coef_value[relevant_items_partition])
            ranking = relevant_items_partition[relevant_items_partition_sorting]

            for index in range(len(ranking)):

                if numCells == len(rows):
                    rows = np.concatenate((rows, np.zeros(dataBlock, dtype=np.int32)))
                    cols = np.concatenate((cols, np.zeros(dataBlock, dtype=np.int32)))
                    values = np.concatenate((values, np.zeros(dataBlock, dtype=np.float32)))

                rows[numCells] = nonzero_model_coef_index[ranking[index]]
                cols[numCells] = currentItem
                values[numCells] = nonzero_model_coef_value[ranking[index]]

                numCells += 1

            # finally, replace the original values of the j-th column
            URM_train.data[start_pos:end_pos] = current_item_data_backup

            elapsed_time = time.time() - start_time
            new_time_value, new_time_unit = seconds_to_biggest_unit(elapsed_time)

            if time.time() - start_time_printBatch > 300 or currentItem == n_items - 1:
                self._print("Processed {} ( {:.2f}% ) in {:.2f} {}. Items per second: {:.2f}".format(
                    currentItem + 1,
                    100.0 * float(currentItem + 1) / n_items,
                    new_time_value,
                    new_time_unit,
                    float(currentItem) / elapsed_time))

                sys.stdout.flush()
                sys.stderr.flush()

                start_time_printBatch = time.time()

        # generate the sparse weight matrix
        self.W_sparse = sps.csr_matrix((values[:numCells], (rows[:numCells], cols[:numCells])),
                                       shape=(n_items, n_items), dtype=np.float32)

    def get_expected_ratings(self, playlist_id):
        playlist_id = int(playlist_id)
        user_profile = self.URM_train[playlist_id]
        expected_ratings = user_profile.dot(self.W_sparse).toarray().ravel()

        # # EDIT
        return expected_ratings

    def recommend(self, playlist_id, at=10):
        playlist_id = int(playlist_id)
        # compute the scores using the dot product
        scores = self.get_expected_ratings(playlist_id)
        user_profile = self.URM_train[playlist_id].indices
        scores[user_profile] = 0

        # rank items
        recommended_items = np.flip(np.argsort(scores), 0)

        return recommended_items[:at]