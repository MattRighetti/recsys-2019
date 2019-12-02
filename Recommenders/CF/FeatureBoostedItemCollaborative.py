from Algorithms.Base.Similarity.Cython.Compute_Similarity_Cython import Compute_Similarity_Cython
from Algorithms.Notebooks_utils.evaluation_function import evaluate_MAP_target_users, evaluate_MAP
from Recommenders.BaseRecommender import BaseRecommender
from Utils.Toolkit import get_URM_TFIDF, normalize_matrix, get_data, feature_boost_URM
from Utils.OutputWriter import write_output
import numpy as np
from multiprocessing import Process


class FeatureBoostedItemCollaborativeFiltering(BaseRecommender):

    RECOMMENDER_NAME = "FeatureBoostedItemCollaborativeFiltering"

    def __init__(self, topK, shrink):
        super().__init__()
        self.URM_train = None
        self.topK = topK
        self.shrink = shrink
        self.SM_item = None
        self.RM = None

    def get_similarity_matrix(self, similarity='tanimoto'):
        similarity_object = Compute_Similarity_Cython(self.URM_train,
                                                      self.shrink,
                                                      self.topK,
                                                      normalize = True,
                                                      tversky_alpha = 1.0,
                                                      tversky_beta = 1.0,
                                                      similarity = similarity)
        return similarity_object.compute_similarity()

    def fit(self, URM_train, boost=False):
        self.URM_train = URM_train.tocsr()

        if boost:
            self.URM_train = feature_boost_URM(URM_train, 5, min_interactions=10)
            #self.URM_train = normalize_matrix(self.URM_train, axis=1)
            #self.URM_train = get_URM_TFIDF(self.URM_train.transpose())
            #self.URM_train = self.URM_train.transpose().tocsr()

        self.SM_item = self.get_similarity_matrix()
        self.RM = self.URM_train.dot(self.SM_item)

    def recommend(self, user_id, at=10, exclude_seen=True):
        expected_ratings = self.get_expected_ratings(user_id)
        recommended_items = np.flip(np.argsort(expected_ratings), 0)

        if exclude_seen:
            unseen_items_mask = np.in1d(recommended_items, self.URM_train[user_id].indices, assume_unique=True, invert=True)
            recommended_items = recommended_items[unseen_items_mask]

        return recommended_items[:at]

    def get_expected_ratings(self, user_id):
        expected_recommendations = self.RM[user_id].todense()
        return np.squeeze(np.asarray(expected_recommendations))

################################################ Test ##################################################
data = get_data(dir_path='../../')

URM = data['train'].tocsr()
URM_test = data['test'].tocsr()
URM_final = URM_test + URM

FBICF = FeatureBoostedItemCollaborativeFiltering(29, 5)
FBICF.fit(URM, boost=True)
FBICF.evaluate_MAP_target(URM_test, data['target_users'])
#FBICF.fit(URM_final, boost=True)
#write_output(FBICF, data['target_users'])
