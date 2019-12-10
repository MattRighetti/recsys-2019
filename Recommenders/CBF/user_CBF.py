from Algorithms.Base.Similarity.Cython.Compute_Similarity_Cython import Compute_Similarity_Cython
from Algorithms.Notebooks_utils.evaluation_function import evaluate_algorithm, evaluate_MAP, evaluate_MAP_target_users
from Utils.Toolkit import DataReader, normalize, get_URM_BM_25, get_URM_TFIDF, get_data
from Recommenders.BaseRecommender import BaseRecommender
import numpy as np
import scipy.sparse as sps



class UserContentBasedRecommender(BaseRecommender):

    RECOMMENDER_NAME = "UserContentBasedRecommender"

    def __init__(self, topK, shrink):
        super().__init__()
        self.topK = topK
        self.shrink = shrink
        self.URM_train = None
        self.UCM = None
        self.SM = None
        self.RM = None

    def compute_similarity(self, UCM, topK, shrink):
        similarity_object = Compute_Similarity_Cython(UCM, shrink, topK, True, similarity='cosine')
        return sps.csr_matrix(similarity_object.compute_similarity())

    def recommend(self, user_id, at=10, exclude_seen=True):
        expected_ratings = self.get_expected_ratings(user_id)
        recommended_items = np.flip(np.argsort(expected_ratings), 0)

        if exclude_seen:
            unseen_items_mask = np.in1d(recommended_items, self.URM_train[user_id].indices, assume_unique=True, invert=True)
            recommended_items = recommended_items[unseen_items_mask]

        return recommended_items[:at]

    def fit(self, URM_train, UCM):
        # PRICE IS NOT INCLUDED INTENTIONALLY
        self.URM_train = URM_train.copy()
        self.UCM = UCM.copy()
        self.UCM = get_URM_TFIDF(self.UCM)
        self.SM = self.compute_similarity(self.UCM.T, self.topK, self.shrink)
        self.RM = self.SM.dot(self.URM_train)
        self.RM = self.RM.tocsr()

    def get_expected_ratings(self, user_id):
        expected_ratings = self.RM[user_id].todense()
        return np.squeeze(np.asarray(expected_ratings))


################################################ Test ##################################################
if __name__ == '__main__':
    max_map = 0
    data = get_data()

    for topK in range(1500, 2501, 100):
        for shrink in [190]:

            args = {
                'topK':1000,
                'shrink':7900
            }

            userCF = UserContentBasedRecommender(args['topK'], args['shrink'])

            userCF.fit(data['train'].tocsr(), data['UCM'].tocsr())
            result = userCF.evaluate_MAP_target(data['test'].tocsr(), data['target_users'])

            if result['MAP'] > max_map:
                max_map = result['MAP']
                print(f'Best values {args}')
################################################ Test ##################################################