from Algorithms.Notebooks_utils.evaluation_function import evaluate_MAP, evaluate_MAP_target_users
from CF.user_cf import UserBasedCollaborativeFiltering
from CF.item_cf import ItemBasedCollaborativeFiltering
import numpy as np

from NonPersonalized.top_pop import TopPop
from Utils.OutputWriter import write_output
from Utils.Toolkit import get_data


class HybridRecommender(object):
    def __init__(self, weights=None, userCF_args=None, itemCF_args=None):
        self.URM_train = None

        ######################## DEFAULT VALUES ########################
        self.weight = {
            'user_cf' : 0.5,
            'item_cf' : 0.5
        }

        self.itemCF_args = {
            'topK' : 31,
            'shrink' : 27
        }

        self.userCF_args = {
            'topK' : 10,
            'shrink' : 100
        }

        ######################## Weights ########################
        if weights is not None:
            self.weight = weights
        if userCF_args is not None:
            self.userCF_args = userCF_args
        if itemCF_args is not None:
            self.itemCF_args = itemCF_args

        ######################## Collaborative Filtering ########################
        self.userCF = UserBasedCollaborativeFiltering(topK=self.userCF_args['topK'], shrink=self.userCF_args['shrink'])
        self.itemCF = ItemBasedCollaborativeFiltering(topK=self.itemCF_args['topK'], shrink=self.itemCF_args['shrink'])
        self.topPop = TopPop()

        self.userCF_scores = None
        self.itemCF_scores = None

    def fit(self, URM_train):
        self.URM_train = URM_train

        ########### FITTING ##########
        self.userCF.fit(self.URM_train.copy())
        self.itemCF.fit(self.URM_train.copy())
        self.topPop.fit(self.URM_train.copy())

    def recommend(self, user_id, at=10, exclude_seen=True):
        self.userCF_scores = self.userCF.get_expected_recommendations(user_id)
        self.itemCF_scores = self.itemCF.get_expected_recommendations(user_id)
        #TODO get_expected_recommendations for TopPop

        start_pos = self.URM_train.indptr[user_id]
        end_pos = self.URM_train.indptr[user_id + 1]

        if len(self.URM_train.indices[start_pos:end_pos]) < 1:
            scores = self.topPop.recommend(user_id)
            ranking = scores
        else:
            scores = (self.userCF_scores * self.weight['user_cf']) + \
                 (self.itemCF_scores * self.weight['item_cf'])

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
        print("HYB -> MAP: {:.4f} with UserCF TopK = {} "
              "& UserCF Shrink = {}, ItemCF TopK = {} & ItemCF Shrink = {} \t"
              .format(result_map, self.userCF.topK, self.userCF.shrink, self.itemCF.topK, self.itemCF.shrink))
        return result_map

    def evaluate_MAP_target(self, URM_test, target_user_list):
        result_map = evaluate_MAP_target_users(URM_test, self, target_user_list)
        print("HYB -> MAP: {:.4f} with UserCF TopK = {} "
              "& UserCF Shrink = {}, ItemCF TopK = {} & ItemCF Shrink = {} | Weights item: {}, user {} \t"
              .format(result_map,
                      self.userCF.topK,
                      self.userCF.shrink,
                      self.itemCF.topK,
                      self.itemCF.shrink,
                      self.weight['user_cf'],
                      self.weight['item_cf'])
              )
        return result_map


################################################ Test ##################################################
max_map = 0
data = get_data()

userCF_args = {
    'topK' : 102,
    'shrink' : 7
}

itemCF_args = {
    'topK' : 31,
    'shrink' : 27
}

hyb = HybridRecommender(weights={'user_cf':0.23, 'item_cf':0.77}, userCF_args=userCF_args, itemCF_args=itemCF_args)
hyb.fit(data['train'])
result = hyb.evaluate_MAP_target(data['test'], data['target_users'])

URM_final = data['train'] + data['test']
URM_final = URM_final.tocsr()

print(type(URM_final))
hyb.fit(URM_final)
write_output(hyb, target_user_list=data['target_users'])
################################################ Test ##################################################