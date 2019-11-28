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

    def get_boosted_recommendations(self, expected_ratings_array, user_id, icm_matrix, user_features_matrix):
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
        cdef float[:] boosted_ratings = np.zeros((num_items), dtype=np.float32,)
        # User-Features array
        cdef long[:] user_features_profile = user_features_matrix[user_id].toarray().ravel().astype(int)
        # Item-Features array
        cdef long[:] item_features_profile

        cdef int item_index, features_index
        cdef double item_rating
        cdef double boost_value = 0.0

        for item_index in range(len(expected_ratings)):
            item_rating = expected_ratings[item_index]
            item_features_profile = icm_matrix[item_index].toarray().ravel().astype(int)

            for feature_index in range(len(item_features_profile)):
                if item_features_profile[feature_index] != 0 and user_features_profile[feature_index] != 0:
                    boost_value += (user_features_profile[features_index]/1000)

            final_rating = item_rating + boost_value
            boosted_ratings[item_index] = final_rating

        boosted_ratings = np.array(boosted_ratings, dtype=np.float32)
        return boosted_ratings