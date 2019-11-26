from Algorithms.Notebooks_utils.evaluation_function import evaluate_MAP, evaluate_MAP_target_users
from CBF.item_CBF import ItemContentBasedRecommender
from CF.user_cf import UserBasedCollaborativeFiltering
from CF.item_cf import ItemBasedCollaborativeFiltering
import numpy as np

from NonPersonalized.top_pop import TopPop
from SLIM.SLIM_BPR_Cython import SLIM_BPR_Cython
from Utils.OutputWriter import write_output
from Utils.Toolkit import get_data


class HybridRecommender(object):
    def __init__(self, weights=None, userCF_args=None, itemCBF_args=None, itemCF_args=None, SLIM_BPR_args=None, with_top_pop=False):
        self.URM_train = None
        self.with_top_pop = with_top_pop
        ######################## DEFAULT VALUES ########################
        self.weight = {
            'user_cf' : 0.5,
            'item_cf' : 0.5,
            'SLIM_BPR' : 0.5,
            'item_cbf' : 0.3
        }

        self.itemCF_args = {
            'topK' : 31,
            'shrink' : 27
        }

        self.userCF_args = {
            'topK' : 10,
            'shrink' : 100
        }

        self.itemCBF_args = {
            'topK_asset' : 100,
            'shrink_asset' : 10,
            'topK_subclass': 100,
            'shrink_subclass': 10
        }

        self.SLIM_BPR_args = {
            'topK': 25,
            'lambda_i': 0.03,
            'lambda_j': 0.9,
            'epochs': 3800,
            'learning_rate' : 1e-3,
            'sgd_mode' : 'adagrad'
        }

        ######################## Weights ########################
        if weights is not None:
            self.weight = weights
        if userCF_args is not None:
            self.userCF_args = userCF_args
        if itemCBF_args is not None:
            self.itemCBF_args = itemCBF_args
        if itemCF_args is not None:
            self.itemCF_args = itemCF_args
        if SLIM_BPR_args is not None:
            self.SLIM_BPR_args = SLIM_BPR_args

        ######################## Collaborative Filtering ########################
        self.userCF = UserBasedCollaborativeFiltering(topK=self.userCF_args['topK'], shrink=self.userCF_args['shrink'])
        self.itemCF = ItemBasedCollaborativeFiltering(topK=self.itemCF_args['topK'], shrink=self.itemCF_args['shrink'])
        self.itemCBF = ItemContentBasedRecommender(topK_asset=self.itemCBF_args['topK_asset'],
                                                   topK_subclass=self.itemCBF_args['topK_subclass'],
                                                   shrink_asset=self.itemCBF_args['shrink_asset'],
                                                   shrink_subclass=self.itemCBF_args['shrink_subclass'])
        self.SLIM_BPR = SLIM_BPR_Cython(epochs=self.SLIM_BPR_args['epochs'],
                                topK=self.SLIM_BPR_args['topK'],
                               lambda_i=self.SLIM_BPR_args['lambda_i'],
                               lambda_j=self.SLIM_BPR_args['lambda_j'],
                               positive_threshold=1,
                               sgd_mode=self.SLIM_BPR_args['sgd_mode'],
                               learning_rate=self.SLIM_BPR_args['learning_rate'],
                               batch_size=1000)
        self.topPop = TopPop()

        self.userCF_scores = None
        self.itemCF_scores = None
        self.SLIM_BPR_scores = None
        self.itemCBF_scores = None

    def fit(self, URM_train, ICM_asset, ICM_subclass):
        self.URM_train = URM_train

        ########### FITTING ##########
        self.userCF.fit(self.URM_train.copy())
        self.itemCF.fit(self.URM_train.copy())
        self.topPop.fit(self.URM_train.copy())
        self.SLIM_BPR.fit(self.URM_train.copy())
        self.itemCBF.fit(self.URM_train.copy(), ICM_asset, ICM_subclass)

    def recommend(self, user_id, at=10, exclude_seen=True):
        self.userCF_scores = self.userCF.get_expected_recommendations(user_id)
        self.itemCF_scores = self.itemCF.get_expected_recommendations(user_id)
        self.SLIM_BPR_scores = self.SLIM_BPR.get_expected_ratings(user_id)
        self.itemCBF_scores = self.itemCBF.get_expected_ratings(user_id)
        # TODO get_expected_recommendations for TopPop

        start_pos = self.URM_train.indptr[user_id]
        end_pos = self.URM_train.indptr[user_id + 1]

        if self.with_top_pop and len(self.URM_train.indices[start_pos:end_pos]) < 1:
            scores = self.topPop.recommend(user_id)
            ranking = scores
        else:
            scores = (self.userCF_scores * self.weight['user_cf']) + \
                    (self.itemCF_scores * self.weight['item_cf']) + \
                    (self.SLIM_BPR_scores * self.weight['SLIM_BPR']) + \
                    (self.itemCBF_scores * self.weight['item_cbf'])

        if exclude_seen:
            scores = self.filter_seen(user_id, scores)

        ranking = scores.argsort()[::-1]

        return ranking[:at]

    def filter_seen(self, user_id, scores):
        start_pos = self.URM_train.indptr[user_id]
        end_pos = self.URM_train.indptr[user_id + 1]
        user_profile = self.URM_train.indices[start_pos:end_pos]

        scores[user_profile] = -50

        return scores

    def evaluate_MAP(self, URM_test):
        result_map = evaluate_MAP(URM_test, self)
        print("HYB -> MAP: {:.4f}"
              .format(result_map))
        return result_map

    def evaluate_MAP_target(self, URM_test, target_user_list):
        result_map = evaluate_MAP_target_users(URM_test, self, target_user_list)
        print("HYB -> MAP: {:.4f}"
              .format(result_map))
        return result_map


################################################ Test ##################################################
max_map = 0
data = get_data(dir_path='../')

userCF_args = {
    'topK' : 204,
    'shrink' : 850
}

itemCF_args = {
    'topK' : 29,
    'shrink' : 5
}

SLIM_BPR_args = {
    'topK': 20,
    'lambda_i': 5.0,
    'lambda_j': 7.0,
    'epochs': 5000,
    'learning_rate' : 1e-4,
    'sgd_mode' : 'adam'
}

weights = {
    'user_cf' : 0.22,
    'item_cf' : 2,
    'SLIM_BPR' : 1,
    'item_cbf' : 0.45
}

hyb = HybridRecommender(weights=weights, userCF_args=userCF_args, itemCF_args=itemCF_args)
hyb.fit(data['train'].tocsr(), data['ICM_asset'].tocsr(), data['ICM_subclass'].tocsr())
result = hyb.evaluate_MAP_target(data['test'], data['target_users'])
print(weights)
#
URM_final = data['train'] + data['test']
URM_final = URM_final.tocsr()
ICM_asset = data['ICM_asset'].tocsr()
ICM_subclass = data['ICM_subclass'].tocsr()
#
# #print(type(URM_final))
hyb.fit(URM_final, ICM_asset, ICM_subclass)
write_output(hyb, target_user_list=data['target_users'])
################################################ Test ##################################################