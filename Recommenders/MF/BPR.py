from implicit.bpr import BayesianPersonalizedRanking
from Utils.Toolkit import get_data
from Recommenders.BaseRecommender import BaseRecommender
import numpy as np

class BPRRecommender(BaseRecommender):

    def __init__(self):
        super().__init__()
        self.URM = None
        self.item_factors = None
        self.user_factors = None
        self.model = BayesianPersonalizedRanking(factors=1000, num_threads=8, verify_negative_samples=True)

    def fit(self, URM):
        self.URM = URM
        URM_transpose = self.URM.T

        self.model.fit(URM_transpose)

        self.user_factors = self.model.user_factors
        self.item_factors = self.model.item_factors

    def get_expected_ratings(self, user_id):
        scores = np.dot(self.user_factors[user_id], self.item_factors.T)
        return np.squeeze(scores)

    def recommend(self, user_id, at=10):
        expected_ratings = self.get_expected_ratings(user_id)

        recommended_items = np.flip(np.argsort(expected_ratings), 0)

        unseen_items_mask = np.in1d(recommended_items, self.URM[user_id].indices, assume_unique=True, invert=True)
        recommended_items = recommended_items[unseen_items_mask]

        return recommended_items[:at]

################################################ Test ##################################################
if __name__ == '__main__':
    data = get_data()

    BPR = BPRRecommender()
    BPR.fit(data['train'].tocsr())
    BPR.evaluate_MAP_target(data['test'].tocsr(), data['target_users'])
################################################ Test ##################################################