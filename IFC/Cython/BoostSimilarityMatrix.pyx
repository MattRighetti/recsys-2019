import cython
IF UNAME_SYSNAME == "linux":
    DEF LONG_t = "long"
ELIF  UNAME_SYSNAME == "Windows":
    DEF LONG_t = "long long"
ELSE:
    DEF LONG_t = "long long"

import scipy.sparse as sps
import numpy as np
cimport numpy as np
from cpython.array cimport array, clone
from Algorithms.Base.Recommender_utils import check_matrix

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.overflowcheck(False)
cdef class BoostSimilarityMatrix:

    cdef float[:,:] boostedRMMatrix
    cdef float[:] recommended_matrix_data
    cdef int[:] recommended_matrix_indptr, recommended_matrix_indices
    cdef int[:] icm_matrix_indptr, icm_matrix_indices
    cdef int[:] user_feature_matrix_indptr
    cdef float[:] user_feature_matrix_data
    cdef int n_items, num_users, rmboosted_data_len, rmboosted_indices_len, rmboosted_indptr_len


    def __init__(self, rm_mat, icm_mat, user_feature_mat):

        self.recommended_matrix_data = rm_mat.data
        self.recommended_matrix_indices = rm_mat.indices
        self.recommended_matrix_indptr = rm_mat.indptr

        self.icm_matrix_indices = icm_mat.indices
        self.icm_matrix_indptr = icm_mat.indptr

        self.user_feature_matrix_data = user_feature_mat.data
        self.user_feature_matrix_indptr = user_feature_mat.indptr

        self.num_users = rm_mat.shape[0]
        self.n_items = rm_mat.shape[1]

        self.rmboosted_data_len = len(rm_mat.data)
        self.rmboosted_indices_len = len(rm_mat.indices)
        self.rmboosted_indptr_len = len(rm_mat.indptr)

        super(BoostSimilarityMatrix, self).__init__()

    def compute_boosted_matrix(self):
        """
        Applies feature boost to the RM
        1. Find item index
        :return: Boosted RM
        """

        cdef float[:] user_expected_recommendations
        cdef float[:] user_interacted_features_weight
        cdef int[:] item_recommended_cols
        cdef int user_id, item_id, \
                    rating_start_pos, rating_end_pos, \
                    user_feature_start_pos, user_feature_end_pos, \
                    item_start_pos, item_end_pos, counter
        cdef int feature_id, num_users, n_items
        cdef float boost_sum

        cdef double[:] boosted_ratings_data = np.zeros(self.rmboosted_data_len)
        cdef double[:] boosted_ratings_data_cols = np.zeros(self.rmboosted_indices_len)
        cdef double[:] boosted_ratings_data_rows = np.zeros(self.rmboosted_indptr_len)

        num_users = self.num_users
        n_items = self.n_items

        counter = 0
        boosted_ratings_data_rows[counter] = 0

        print("Initializing...")

        for user_id in range(num_users):

            # To partition user profiles
            rating_start_pos = self.recommended_matrix_indptr[user_id]
            rating_end_pos = self.recommended_matrix_indptr[user_id + 1]
            # Array of recommended items rating
            user_expected_recommendations = self.recommended_matrix_data[rating_start_pos:rating_end_pos]
            # Col Index of each items (supposing that some values in user_profile could be 0) |
            # in most cases should be np.arange(num_items)
            item_recommended_cols = self.recommended_matrix_indices[rating_start_pos:rating_end_pos]

            # To partition user interacted features
            user_feature_start_pos = self.user_feature_matrix_indptr[user_id]
            user_feature_end_pos = self.user_feature_matrix_indptr[user_id + 1]
            # Array of weight of interacted features (multiplied later if present in item features)
            user_interacted_features_weight = self.user_feature_matrix_data[user_feature_start_pos:user_feature_end_pos]

            for item_id in item_recommended_cols:

                boost_sum = 0

                # To partition item features
                item_start_pos = self.icm_matrix_indptr[item_id]
                item_end_pos = self.icm_matrix_indptr[item_id + 1]
                # Array of item features
                item_features_cols = self.icm_matrix_indices[item_start_pos:item_end_pos]

                for feature_index in item_features_cols:
                    boost_sum += user_interacted_features_weight[feature_index]

                boosted_ratings_data[counter] = user_expected_recommendations[item_id] * boost_sum
                boosted_ratings_data_cols[counter] = item_id
                boosted_ratings_data_rows[user_id+1] = counter

                counter += 1

        boostedRMMatrix = sps.csr_matrix((boosted_ratings_data, boosted_ratings_data_cols, boosted_ratings_data_rows),
                                         shape=(num_users, n_items), dtype=np.float32)

        return boostedRMMatrix