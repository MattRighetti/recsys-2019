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

    def __init__(self):
        print("")

    def compute_boosted_matrix(self, rm_mat, icm_mat, user_feature_mat):
        """
        Applies feature boost to the RM
        1. Find item index
        :return: Boosted RM
        """

        cdef long data_array_len, indices_array_len, indptr_array_len, item_start_pos, item_end_pos, \
            user_feature_start_pos, user_feature_end_pos, n_users, n_items, user_id, item_id, feature_index,\
            item_in_row_counter, user_start_pos, user_end_pos
        cdef float[:] data, user_profile, user_feature_profile
        cdef long[:] indices, indptr

        cdef int[:] user_profile_items, item_feature_indices

        rm_mat = rm_mat.copy()
        icm_mat = icm_mat.copy()
        user_feature_mat = user_feature_mat.copy()

        n_users = rm_mat.shape[0]
        n_items = rm_mat.shape[1]

        data_array_len = len(rm_mat.data)
        indices_array_len = len(rm_mat.indices)
        indptr_array_len = len(rm_mat.indptr)

        # Create arrays to fill later with the boosted data
        # Must be the same dimensions as RM's
        data = np.zeros(data_array_len, dtype=np.float32)
        print(data_array_len)
        indices = np.zeros(indices_array_len, dtype=np.long)
        print(indices_array_len)
        indptr = np.zeros(indptr_array_len, dtype=np.long)
        print(indptr_array_len)

        print(n_users)
        print(n_items)

        counter = 0
        indptr[counter] = 0

        print("BoOoOoOosTiNg!")

        for user_id in range(rm_mat.shape[0]):

            if user_id % 1000 == 0:
                print(f'BoOoOoOosTeD {user_id/rm_mat.shape[0]*100:.2f}% users')

            item_in_row_counter = 0

            user_start_pos = rm_mat.indptr[user_id]
            user_end_pos = rm_mat.indptr[user_id + 1]

            user_profile = rm_mat.data[user_start_pos:user_end_pos]
            user_profile_items = rm_mat.indices[user_start_pos:user_end_pos]

            for item_id in user_profile_items:
                boost_sum = 0.0
                item_start_pos = icm_mat.indptr[item_id]
                item_end_pos = icm_mat.indptr[item_id + 1]

                item_feature_indices = icm_mat.indices[item_start_pos:item_end_pos]

                # Take the whole line to speed up multiplication
                user_feature_profile = user_feature_mat[user_id].toarray().ravel()

                for feature_index in item_feature_indices:
                    boost_sum += user_feature_profile[feature_index]/100

                data[counter] = user_profile[item_id] + boost_sum
                indices[counter] = item_id
                counter += 1
                item_in_row_counter += 1

            print(f'Saving user: {user_id+1}')
            # TODO check if out of range
            indptr[user_id + 1] = indptr[user_id] + item_in_row_counter

        boostedRMMatrix = sps.csr_matrix((data, indices, indptr), shape=(n_users, n_items), dtype=np.float64)

        return boostedRMMatrix