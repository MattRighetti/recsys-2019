from CF.user_cf import UserBasedCollaborativeFiltering
from CF.item_cf import ItemBasedCollaborativeFiltering
import numpy as np

class HybridRecommender(object):

    def __init__(self, URM_train):
        self.URM_train = URM_train

        self.userCF = UserBasedCollaborativeFiltering(self.URM_train.copy(), topK=7, shrink=700)
        self.itemCF = ItemBasedCollaborativeFiltering(self.URM_train.copy(), topK=20, shrink=100)

        self.itemCF.fit()
        self.userCF.fit(similarity="pearson")

        self.userCF_w = None
        self.itemCF_w = None

        self.userCF_scores = None
        self.itemCF_scores = None


    def fit(self, userCF_w=1, itemCF_w=1):
        self.userCF_w = userCF_w
        self.itemCF_w = itemCF_w

    def recommend(self, user_id, at=10, exclude_seen=True):
        self.userCF_scores = self.userCF.get_scores(user_id)
        self.itemCF_scores = self.itemCF.get_scores(user_id)

        scores = (self.userCF_scores * self.userCF_w) + (self.itemCF_scores * self.itemCF_w)

        if exclude_seen:
            scores = self.filter_seen(user_id, scores)

        ranking = scores.argsort()[::-1]

        return ranking[:at]

    def filter_seen(self, user_id, scores):
        start_pos = self.URM_train.indptr[user_id]
        end_pos = self.URM_train.indptr[user_id + 1]
        user_profile = self.URM_train.indices[start_pos:end_pos]

        scores[user_profile] = -np.inf

        return scores
