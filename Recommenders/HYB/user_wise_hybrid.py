from Algorithms.Notebooks_utils.evaluation_function import evaluate_MAP_target_users
from Recommenders.NonPersonalized.top_pop import TopPop
from Recommenders.CF.item_cf import ItemBasedCollaborativeFiltering
from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.CBF.user_CBF import UserContentBasedRecommender
import numpy as np
from Utils.Toolkit import get_data



class UserWiseHybridRecommender(BaseRecommender):

    RECOMMENDER_NAME = "UW HYB"

    def __init__(self, ICF_topK, ICF_shrink, UCBF_topK, UCBF_shrink):
        super().__init__()
        self.URM_train = None
        self.TopPop = TopPop()
        self.ICF_topK = ICF_topK
        self.ICF_shrink = ICF_shrink
        self.UCBF_topK = UCBF_topK
        self.UCBF_shrink = UCBF_shrink
        self.UCBF = None
        self.itemCF = None
        self.user_profile_lengths = None

    def fit(self, URM_train, UCM):
        self.itemCF = ItemBasedCollaborativeFiltering(topK=self.ICF_topK, shrink=self.ICF_shrink)
        self.UCBF = UserContentBasedRecommender(self.UCBF_topK, self.UCBF_shrink)
        self.URM_train = URM_train.tocsr()
        self.TopPop.fit(URM_train.copy().tocsr())
        self.UCBF.fit(URM_train.copy().tocsr(), UCM)
        self.itemCF.fit(URM_train.copy().tocsr())
        self.user_profile_lengths = np.ediff1d(URM_train.indptr)

    def recommend(self, user_id, at):
        if self.user_profile_lengths[user_id] < 1:
            return self.UCBF.recommend(user_id, at=at)
        else:
            return self.itemCF.recommend(user_id, at=at)

################################################ Test ##################################################
# data = get_data(dir_path='../../')
# hyb = UserWiseHybridRecommender(29, 5, 750, 1000)
# hyb.fit(data['train'], data['UCM'])
# hyb.evaluate_MAP_target(data['test'], data['target_users'])
################################################ Test ##################################################