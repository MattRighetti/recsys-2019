from Algorithms.Notebooks_utils.evaluation_function import evaluate_MAP, evaluate_MAP_target_users
from Algorithms.GraphBased.P3alphaRecommender import P3alphaRecommender
from Recommenders.CF.item_cf import ItemBasedCollaborativeFiltering
from Recommenders.CF.user_cf import UserBasedCollaborativeFiltering
from Recommenders.CBF.item_CBF import ItemContentBasedRecommender
from Recommenders.CBF.user_CBF import UserContentBasedRecommender
from Recommenders.MF.ALS import AlternatingLeastSquare
from Recommenders.SLIM.SLIM_BPR_Cython import SLIM_BPR_Cython
from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.NonPersonalized.top_pop import TopPop
from Utils.Toolkit import get_data, feature_boost_URM
from Utils.OutputWriter import write_output
import numpy as np


class HybridRecommender(BaseRecommender):

    RECOMMENDER_NAME = "HYB"

    def __init__(self, weights=None, userCF_args=None, itemCBF_args=None,
                 itemCF_args=None, SLIM_BPR_args=None, userCBF_args=None):

        super().__init__()
        ######################## URM ########################
        self.URM_train = None
        ######################## Weights ########################
        self.weight = weights
        ######################## Args ########################
        self.userCF_args = userCF_args
        self.itemCBF_args = itemCBF_args
        self.itemCF_args = itemCF_args
        self.SLIM_BPR_args = SLIM_BPR_args
        self.userCBF_args = userCBF_args
        ######################## Scores ########################
        self.userCF_scores = None
        self.itemCF_scores = None
        self.SLIM_BPR_scores = None
        self.itemCBF_scores = None
        self.ALS_scores = None
        ######################## Collaborative Filtering ########################
        self.userCF = UserBasedCollaborativeFiltering(topK=self.userCF_args['topK'], shrink=self.userCF_args['shrink'])
        self.itemCF = ItemBasedCollaborativeFiltering(topK=self.itemCF_args['topK'], shrink=self.itemCF_args['shrink'])
        self.userCBF = UserContentBasedRecommender(topK=self.userCBF_args['topK'], shrink=self.userCBF_args['shrink'])

        self.itemCBF = ItemContentBasedRecommender(topK=self.itemCBF_args['topK'],
                                                   shrink=self.itemCBF_args['shrink'])

        self.SLIM_BPR = SLIM_BPR_Cython(epochs=self.SLIM_BPR_args['epochs'],
                                topK=self.SLIM_BPR_args['topK'],
                                lambda_i=self.SLIM_BPR_args['lambda_i'],
                                lambda_j=self.SLIM_BPR_args['lambda_j'],
                                positive_threshold=1,
                                sgd_mode=self.SLIM_BPR_args['sgd_mode'],
                                symmetric=self.SLIM_BPR_args['symmetric'],
                                learning_rate=self.SLIM_BPR_args['learning_rate'],
                                batch_size=1000)
        self.ALS = AlternatingLeastSquare()

    def fit(self, URM_train, ICM, UCM):
        self.URM_train = URM_train.copy()
        ########### FITTING ##########
        self.userCF.fit(self.URM_train.copy())
        self.itemCF.fit(self.URM_train.copy(), ICM)
        self.userCBF.fit(self.URM_train.copy(), UCM)
        self.SLIM_BPR.fit(self.URM_train.copy())
        self.itemCBF.fit(self.URM_train.copy(), ICM)
        self.ALS.fit(self.URM_train.copy())

    def recommend(self, user_id, at=10, exclude_seen=True):
        self.userCF_scores = self.userCF.get_expected_ratings(user_id)
        self.itemCF_scores = self.itemCF.get_expected_ratings(user_id)
        self.SLIM_BPR_scores = self.SLIM_BPR.get_expected_ratings(user_id)
        self.itemCBF_scores = self.itemCBF.get_expected_ratings(user_id)
        self.ALS_scores = self.ALS.get_expected_ratings(user_id)

        start_pos = self.URM_train.indptr[user_id]
        end_pos = self.URM_train.indptr[user_id + 1]

        if len(self.URM_train.indices[start_pos:end_pos]) < 1:
            scores = self.userCBF.get_expected_ratings(user_id)
        else:
            scores = (self.userCF_scores * self.weight['user_cf']) + \
                    (self.itemCF_scores * self.weight['item_cf']) + \
                    (self.SLIM_BPR_scores * self.weight['SLIM_BPR']) + \
                    (self.itemCBF_scores * self.weight['item_cbf']) + \
                    (self.ALS_scores * self.weight['ALS'])

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

################################################ Test ##################################################
max_map = 0
data = get_data()

userCF_args = {
    'topK' : 102,
    'shrink' : 7
}

userCBF_args = {
    'topK' : 1000,
    'shrink' : 7900
}

itemCF_args = {
    'topK' : 29,
    'shrink' : 5
}
itemCBF_args = {
    'topK' : 29,
    'shrink' : 5
}

SLIM_BPR_args = {
    'topK': 20,
    'lambda_i': 5.0,
    'lambda_j': 7.0,
    'epochs': 5000,
    'learning_rate' : 1e-4,
    'symmetric' : True,
    'sgd_mode' : 'adam'
}

weights = {
    'user_cf' : 0,
    'item_cf' : 1.55,
    'SLIM_BPR' : 1.52,
    'item_cbf' : 0,
    'ALS' : 0.6
}

hyb = HybridRecommender(weights=weights,
                        userCF_args=userCF_args,
                        SLIM_BPR_args=SLIM_BPR_args,
                        itemCF_args=itemCF_args,
                        itemCBF_args=itemCBF_args,
                        userCBF_args=userCBF_args)

#hyb.fit(data['train'].tocsr(), data['ICM_subclass'].tocsr(), data['UCM'].tocsr())
#result = hyb.evaluate_MAP_target(data['test'], data['target_users'])
print(weights)
# #
URM_final = data['train'] + data['test']
URM_final = URM_final.tocsr()
# #
# #print(type(URM_final))
hyb.fit(URM_final, data['ICM_subclass'].tocsr(), data['UCM'].tocsr())
write_output(hyb, target_user_list=data['target_users'])
################################################ Test ##################################################