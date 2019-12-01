from Utils.Toolkit import get_URM_TFIDF, normalize_matrix, get_data, generate_SM_user_feature_matrix
from Algorithms.Notebooks_utils.evaluation_function import evaluate_MAP_target_users, evaluate_MAP
from Algorithms.Base.Similarity.Cython.Compute_Similarity_Cython import Compute_Similarity_Cython
from Recommenders.BaseRecommender import BaseRecommender
import scipy.sparse as sps
import numpy as np


class FeatureCollaborativeFiltering(BaseRecommender):

    RECOMMENDER_NAME = "FeatureCollaborativeFiltering"

    def __init__(self, topK, shrink):
        super().__init__()
        self.ICM = None
        self.topK = topK
        self.shrink = shrink
        self.SM_feature = None
        self.SM_user_feature = None
        self.RM = None

    def get_similarity_matrix(self, similarity='tanimoto'):
        similarity_object = Compute_Similarity_Cython(self.ICM,
                                                      self.shrink,
                                                      self.topK,
                                                      normalize = True,
                                                      tversky_alpha = 1.0,
                                                      tversky_beta = 1.0,
                                                      similarity = similarity)
        return similarity_object.compute_similarity()

    def fit(self, ICM, URM_train):
        self.ICM = ICM.tocsr()
        self.SM_user_feature = generate_SM_user_feature_matrix(URM_train, self.ICM)
        self.SM_feature = self.get_similarity_matrix()
        self.RM = self.SM_user_feature.dot(self.SM_feature)

    def recommend(self, user_id, at=10, exclude_seen=False):
        expected_ratings = self.get_expected_ratings(user_id)
        recommended_items = np.flip(np.argsort(expected_ratings), 0)

        return recommended_items[:at]

    def get_expected_ratings(self, user_id):
        expected_recommendations = self.RM[user_id].todense()
        return np.squeeze(np.asarray(expected_recommendations))

    def get_features_ratings(self, at=10):
        data = []

        for i in range(self.RM.shape[0]):
            recommended_features = np.flip(np.argsort(self.RM[i].toarray().ravel()), 0)
            data.append(recommended_features)

        data_csr = sps.csr_matrix(data)
        return data_csr[:, :at]

################################################ Test ##################################################
# data = get_data(dir_path='../../')
#
# ICM_test = data['ICM_test'].tocsr()
# ICM_train = data['ICM_train'].tocsr()
# URM = data['train'].tocsr()
#
# for topK in np.arange(10, 101, 10):
#     for shrink in np.arange(10, 601, 100):
#         featCF = FeatureCollaborativeFiltering(topK, shrink)
#         featCF.fit(ICM_train, URM)
#         featCF.evaluate_MAP(ICM_test)
################################################ Test ##################################################