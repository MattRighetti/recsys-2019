from Algorithms.Base.Similarity.Cython.Compute_Similarity_Cython import Compute_Similarity_Cython
import scipy.sparse as sps
import numpy as np


class ItemFeatureCollaborativeFiltering(object):
    def __init__(self, topK, shrink):
        self.topK = topK
        self.shrink = shrink

        self.URM_train = None
        self.ICM = None
        self.SM_item = None
        self.SM_user_feature = None
        self.RM_item = None

    def get_similarity_matrix(self, similarity='cosine'):
        similarity_object = Compute_Similarity_Cython(self.URM_train,
                                                      self.shrink,
                                                      self.topK,
                                                      normalize = True,
                                                      tversky_alpha = 1.0,
                                                      tversky_beta = 1.0,
                                                      similarity = similarity)
        return sps.csr_matrix(similarity_object.compute_similarity())


    def fit(self, URM_train, ICM):
        """
        PASS URM_TRAIN and ICM as CSR MATRICES
        :param URM_train:
        :param ICM:
        :return:
        """
        self.URM_train = URM_train
        self.ICM = ICM
        self.SM_item = self.get_similarity_matrix(URM_train)
        self.RM_item = self.URM_train.dot(self.SM_item)
        self.SM_user_feature = self.URM_train.dot(self.ICM)

    def get_expected_ratings(self, user_id):
        """
        Returns the ratings of the corresponding user
        :param user_id: ID of the User
        :return: 1D array of items containing each item rating
        """
        expected_ratings = self.RM_item[user_id].todense()
        return np.squeeze(np.asarray(expected_ratings))

    def feature_boost(self, user_id):
        expected_ratings = self.get_expected_ratings(user_id)

        for item_id in range(len(expected_ratings) - 1):
            start_pos = item_id
            end_pos = item_id + 1

            features_indices = self.ICM.indices[start_pos:end_pos]

            for feature in features_indices:
                if self.SM_user_feature[user_id, feature] != 0:
                    print("Cazzo in bocca a davide")
                    # TODO boost

        return expected_ratings

    def recommend(self, user_id):

