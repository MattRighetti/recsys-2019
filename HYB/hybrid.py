from CF.user_cf import UserBasedCollaborativeFiltering
from CF.item_cf import ItemBasedCollaborativeFiltering
from multiprocessing import Process
import numpy as np

class HybridRecommender(object):

    def __init__(self, URM_train):
        self.URM_train = URM_train

        self.userBCF = UserBasedCollaborativeFiltering(self.URM_train.copy(), topK=10, shrink=500)
        self.itemBCF = ItemBasedCollaborativeFiltering(self.URM_train.copy(), topK=20, shrink=10)

        p1 = Process(target=self.userBCF.fit)
        p2 = Process(target=self.itemBCF.fit)
        p1.start()
        p2.start()
        p1.join()
        p2.join()

        self.userBCF.fit(similarity="pearson")
        self.itemBCF.fit()

        self.userBCF_w = None
        self.itemBCF_w = None

        self.userBCF_scores = None
        self.itemBCF_scores = None


    def fit(self, userCBF_w=1, itemCBF_w=1):
        self.userBCF_w = userCBF_w
        self.itemBCF_w = itemCBF_w

    def recommend(self, user_id, at=10, exclude_seen=True):
        self.userBCF_scores = self.userBCF.get_scores(user_id)
        self.itemBCF_scores = self.itemBCF.get_scores(user_id)

        scores = (self.userBCF_scores * self.userBCF_w) + (self.itemBCF_scores * self.itemBCF_w)

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
