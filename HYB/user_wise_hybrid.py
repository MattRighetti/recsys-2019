from Algorithms.Notebooks_utils.evaluation_function import evaluate_MAP_target_users
from NonPersonalized.top_pop import TopPop
from CF.item_cf import ItemBasedCollaborativeFiltering
import numpy as np


class UserWiseHybridRecommender(object):
    def __init__(self):
        self.URM_train = None
        self.TopPop = TopPop()
        self.topK = 10
        self.shrink = 20
        self.itemCF = None
        self.user_profile_lengths = None

    def fit(self, URM_train):
        if self is None:
            self.itemCF = ItemBasedCollaborativeFiltering(topK=self.topK, shrink=self.shrink)
        self.URM_train = URM_train.tocsr()
        self.TopPop.fit(URM_train.copy().tocsr())
        self.itemCF.fit(URM_train.copy().tocsr())
        self.user_profile_lengths = np.ediff1d(URM_train.indptr)

    def recommend(self, user_id, at=10):
        if self.user_profile_lengths[user_id] < 1:
            return self.TopPop.recommend(user_id)
        else:
            return self.itemCF.recommend(user_id, at=10)

    def evaluate_MAP_target(self, URM_test, target_user_list):
        result = evaluate_MAP_target_users(URM_test, self, target_user_list)
        print("HybUserWise -> MAP: {:.4f}".format(result))
        return result

    def set_topK(self, topK):
        self.topK = topK

    def set_shrink(self, shrink):
        self.shrink = shrink

    def set_item_cf(self, itemCF):
        self.itemCF = itemCF
