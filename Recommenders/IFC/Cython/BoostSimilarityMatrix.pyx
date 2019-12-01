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
cdef class Booster:

    cdef int n_cols
    cdef int n_rows
    cdef int n_features
    cdef double novelty_weight

    def __init__(self, RM, ICM, UFM, double weight_novelty):
        ICM_matrix = ICM.copy()
        UFM_matrix = UFM.copy()
        R_matrix = RM.copy()
        self.n_cols = ICM_matrix.shape[0]
        self.n_features = 1
        self.n_rows = RM.shape[0]
        self.novelty_weight = weight_novelty

    def apply_boost(self, R_matrix, ICM_matrix, UFM_matrix):
        cdef int user_id = 0

        cdef int item_RM_index = 0
        cdef int feature_index = 0

        cdef int new_features = 0
        cdef int not_new_features = 0
        cdef double feature_weight = 0.0
        cdef double final_weight = 0.0

        cdef user_profile
        cdef item_profile
        cdef user_feature_profile

        cdef int start_pos
        cdef int end_pos

        print(type(R_matrix))
        while user_id < self.n_rows:
            if user_id % 1000 == 0:
                print(user_id)
            user_profile = R_matrix[user_id].toarray().ravel()
            item_RM_index = 0
            user_feature_profile = UFM_matrix[user_id].toarray().ravel()
            # Scorri tutte le item
            while item_RM_index < self.n_cols:
                if user_profile[item_RM_index] > 0:
                    start_pos = ICM_matrix.indptr[item_RM_index]
                    end_pos = ICM_matrix.indptr[item_RM_index + 1]
                    item_profile = ICM_matrix.indices[start_pos:end_pos]

                    feature_weight = 0
                    item_feature_weight = ICM_matrix[item_RM_index, item_profile[0]]
                    if user_feature_profile[item_profile[0]] != 0:
                        feature_weight += (ICM_matrix[item_RM_index, item_profile[0]] * user_feature_profile[item_profile[0]])

                    R_matrix[user_id, item_RM_index] += feature_weight

                item_RM_index += 1
            user_id += 1

        RM_boosted = R_matrix
        return RM_boosted