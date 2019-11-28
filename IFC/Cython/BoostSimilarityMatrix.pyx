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
@cython.overflowcheck(True)
cdef class Booster:

    def __init__(self):
        super(Booster, self).__init__()

    def get_boosted_recommendations(self, expected_ratings_array, user_profile_indices, user_id, icm_matrix, user_features_matrix):
        """
        Applies feature boost to the RM
        1. Find item index
        :return: Boosted RM
        """

        # Expected ratings
        cdef float[:] expected_ratings = expected_ratings_array.copy()
        # Num of indexes of the array to be returned
        cdef int num_items = icm_matrix.shape[0]
        # Array to be returned
        cdef double[:] boosted_ratings = np.zeros((num_items), dtype=np.double,)
        # User-Features array
        cdef int user_startpos = user_features_matrix.indptr[user_id]
        cdef int user_endpos = user_features_matrix.indptr[user_id + 1]
        cdef float[:] user_features_profile = user_features_matrix.data[user_startpos:user_endpos]
        cdef int[:] user_features_indices = user_features_matrix.indices[user_startpos:user_endpos]
        # Item-Features array
        cdef int item_startpos = 0
        cdef int item_endpos = 0
        cdef int[:] item_features_indices
        cdef double[:] item_features_profile


        cdef int item_index = 0
        cdef int features_index = 0
        cdef double item_rating = 0.0
        cdef double boost_value = 0.0

        for item_index in user_profile_indices:
            item_rating = expected_ratings[item_index]
            item_startpos = icm_matrix.indptr[item_index]
            item_endpos = icm_matrix.indptr[item_index + 1]
            item_features_profile = icm_matrix.data[item_startpos:item_endpos]
            item_features_indices = icm_matrix.indices[item_startpos:item_endpos]

            for feature_index in item_features_indices:
                boost_value += (user_features_profile[features_index]/1000)

            final_rating = item_rating
            boosted_ratings[item_index] = final_rating

        boosted_ratings = np.array(boosted_ratings, dtype=np.double)
        return boosted_ratings