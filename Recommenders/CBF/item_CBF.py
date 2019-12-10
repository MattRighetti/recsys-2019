from Algorithms.Base.Similarity.Cython.Compute_Similarity_Cython import Compute_Similarity_Cython
from Algorithms.Notebooks_utils.evaluation_function import evaluate_algorithm, evaluate_MAP, evaluate_MAP_target_users
from Utils.Toolkit import DataReader, normalize, get_URM_BM_25, get_URM_TFIDF, get_data
from Recommenders.BaseRecommender import BaseRecommender
import numpy as np
import scipy.sparse as sps


class ItemContentBasedRecommender(BaseRecommender):

    RECOMMENDER_NAME = "ItemContentBasedRecommender"

    def __init__(self, topK, shrink):
        super().__init__()
        self.topK = topK
        self.shrink = shrink

        self.URM_train = None
        self.ICM = None
        self.SM = None
        self.RM = None

    def compute_similarity(self, ICM, topK, shrink):
        similarity_object = Compute_Similarity_Cython(ICM.transpose(), shrink, topK, True, similarity='cosine')
        return sps.csr_matrix(similarity_object.compute_similarity())

    def fit(self, URM_train, ICM):
        # PRICE IS NOT INCLUDED INTENTIONALLY
        self.URM_train = URM_train.copy()
        self.ICM = ICM.copy()

        self.SM = self.compute_similarity(self.ICM, self.topK, self.shrink)
        self.RM = self.URM_train.dot(self.SM)

    def get_expected_ratings(self, user_id):
        expected_ratings = self.RM[user_id].todense()
        return np.squeeze(np.asarray(expected_ratings))

    def recommend(self, user_id, at=10, exclude_seen=True):
        expected_ratings = self.get_expected_ratings(user_id)
        recommended_items = np.flip(np.argsort(expected_ratings), 0)

        if exclude_seen:
            unseen_items_mask = np.in1d(recommended_items, self.URM_train[user_id].indices, assume_unique=True, invert=True)
            recommended_items = recommended_items[unseen_items_mask]

        return recommended_items[:at]


################################################ Test ##################################################
if __name__ == '__main__':
    max_map = 0
    data = get_data()

    for topK in [10, 12, 15, 20]:
        for shrink in [5, 10]:

            args = {
                'topK': topK,
                'shrink': shrink
            }

            itemCBF = ItemContentBasedRecommender(args['topK'], args['shrink'])

            itemCBF.fit(data['train'], data['ICM_subclass'])
            result = itemCBF.evaluate_MAP_target(data['test'], data['target_users'])

            if result['MAP'] > max_map:
                max_map = result['MAP']
                print(f'Best values {args}')
################################################ Test ##################################################