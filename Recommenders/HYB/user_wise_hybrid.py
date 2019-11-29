from Algorithms.Notebooks_utils.evaluation_function import evaluate_MAP_target_users
from Recommenders.NonPersonalized.top_pop import TopPop
from Recommenders.CF.item_cf import ItemBasedCollaborativeFiltering
from Recommenders.BaseRecommender import BaseRecommender
import numpy as np


class UserWiseHybridRecommender(BaseRecommender):
    def __init__(self):
        super().__init__()
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

    def recommend(self, user_id):
        if self.user_profile_lengths[user_id] < 1:
            return self.TopPop.recommend(user_id)
        else:
            return self.itemCF.recommend(user_id, at=10)