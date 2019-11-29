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

    def get_boosted_recommendations(self, expected_ratings_array, user_profile_indices, user_id, weight, icm_matrix, user_features_matrix):
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
        boosted_ratings = np.zeros((num_items), dtype=np.double,)
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
        cdef int w = weight


        cdef int item_index = 0
        cdef int features_index = 0
        cdef double item_rating = 0.0
        cdef double boost_value

        for item_index in user_profile_indices:
            boost_value = 0.0
            item_rating = expected_ratings[item_index]
            item_startpos = icm_matrix.indptr[item_index]
            item_endpos = icm_matrix.indptr[item_index + 1]
            item_features_profile = icm_matrix.data[item_startpos:item_endpos]
            item_features_indices = icm_matrix.indices[item_startpos:item_endpos]

            # TODO fix errors
            for feature_index in item_features_indices:
                boost_value += (user_features_profile[features_index] * item_features_profile[features_index])

            if boost_value > 1:
                boost_value /= (boost_value + 0.5)


            final_rating = item_rating + boost_value
            boosted_ratings[item_index] = final_rating

        boosted_ratings = np.array(boosted_ratings, dtype=np.double)
        return boosted_ratings





    def boost(self, recommended_item_indexes, recommended_item_ratings, user_id, icm_matrix, user_features_matrix):
        """
        Apply boost on the first 10 items
        :param recommended_item_indexes: Index of first 10 items
        :param recommended_item_ratings: Ratings of first 10 items
        :param user_id: User ID
        :param icm_matrix: ICM
        :param user_features_matrix: UFM
        :return:
        """

        cdef int user_ID = user_id
        cdef int i = 0
        cdef int features_index = 0
        cdef double item_rating = 0.0
        cdef double boost_value
        cdef int counter = 0

        cdef int user_startpos = user_features_matrix.indptr[user_id]
        cdef int user_endpos = user_features_matrix.indptr[user_id+1]
        cdef int item_startpos, item_endpos
        # TODO why float?
        cdef float[:] user_features_profile_row = user_features_matrix[user_id].toarray().ravel()
        #cdef float[:] user_features_profile = user_features_matrix.data[user_startpos:user_endpos]
        cdef int[:] user_features_indices = user_features_matrix.indices[user_startpos:user_endpos]

        cdef int new_features
        cdef int not_new_features
        cdef double features_weights
        cdef int feature_index, inner_counter

        boosted_ratings = np.zeros((10), dtype=np.double,)

        if len(user_features_indices) > 0:
            for i in range(len(recommended_item_indexes)):
                boost_value = 0.0
                features_weights = 0.0
                new_features = 0
                not_new_features = 0
                item_rating = recommended_item_ratings[i]
                item_startpos = icm_matrix.indptr[recommended_item_indexes[i]]
                item_endpos = icm_matrix.indptr[recommended_item_indexes[i] + 1]
                item_features_profile = icm_matrix.data[item_startpos:item_endpos]
                item_features_indices = icm_matrix.indices[item_startpos:item_endpos]

                inner_counter = 0
                for feature_index in item_features_indices:
                    if user_features_profile_row[feature_index] != 0:
                        not_new_features += 1
                        # TODO could be a bad thing if the user_features_profile takes all features of the matrix in consideration
                        features_weights += (item_features_profile[inner_counter] * user_features_matrix[user_id, feature_index])
                        inner_counter += 1
                    else:
                        new_features += 1
                        inner_counter += 1

                if new_features == 0:
                    boost_value += features_weights
                else:
                    boost_value += (not_new_features / new_features) + features_weights

                if boost_value > 5:
                    print(f'Weight over 5!')

                final_rating = item_rating + boost_value
                boosted_ratings[counter] = final_rating
                counter += 1

            boosted_ratings = np.array(boosted_ratings, dtype=np.double)
            return boosted_ratings, True
        else:
            return recommended_item_ratings, False