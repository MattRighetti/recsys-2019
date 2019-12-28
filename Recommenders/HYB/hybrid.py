from Algorithms.Notebooks_utils.evaluation_function import evaluate_MAP, evaluate_MAP_target_users
from Algorithms.GraphBased.P3alphaRecommender import P3alphaRecommender
from Recommenders.CF.item_cf import ItemBasedCollaborativeFiltering
from Recommenders.CF.user_cf import UserBasedCollaborativeFiltering
from Recommenders.CBF.item_CBF import ItemContentBasedRecommender
from Recommenders.CBF.user_CBF import UserContentBasedRecommender
from Recommenders.MF.ALS import AlternatingLeastSquare
from Recommenders.Graph.P3GraphRecommender import P3alphaRecommender
from Algorithms.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.SLIM.SlimElasticNet import SLIMElasticNetRecommender
from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.NonPersonalized.top_pop import TopPop
from Utils.Toolkit import get_data, get_target_users_group
from Utils.OutputWriter import write_output
import numpy as np


class HybridRecommender(BaseRecommender):

    RECOMMENDER_NAME = "HYB"

    def __init__(self, weights=None, userCF_args=None, itemCF_args=None, P3alpha_args=None,
                 userCBF_args=None, RP3alpha_args=None, SlimEl_args=None):
        super().__init__()
        ######################## URM ########################
        self.URM_train = None

        ######################## Weights ########################
        self.weight_initial = weights[0]
        self.weight_middle = weights[1]
        self.weight_end = weights[2]

        ######################## Args ########################
        self.userCF_args = userCF_args
        self.itemCF_args = itemCF_args
        self.userCBF_args = userCBF_args
        self.P3alpha_args = P3alpha_args
        self.RP3alpha_args = RP3alpha_args
        self.SlimEl_args = SlimEl_args

        ######################## Scores ########################
        self.userCF_scores = None
        self.itemCF_scores = None
        self.ALS_scores = None
        self.P3alpha_scores =  None
        self.RP3alpha_scores = None
        self.SlimEl_scores = None

        ######################## Collaborative Filtering ########################
        self.itemCF = ItemBasedCollaborativeFiltering(topK=self.itemCF_args['topK'],
                                                      shrink=self.itemCF_args['shrink'],
                                                      feature_weighting=self.itemCF_args['fw'],
                                                      similarity=self.itemCF_args['similarity'],
                                                      tversky_alpha=self.itemCF_args['tversky_alpha'],
                                                      tversky_beta=self.itemCF_args['tversky_beta'],
                                                      asymmetric_alpha=self.itemCF_args['asymmetric_alpha'])

        self.userCBF = UserContentBasedRecommender(topK=self.userCBF_args['topK'], shrink=self.userCBF_args['shrink'])

        self.P3alpha = P3alphaRecommender(topK=self.P3alpha_args['topK'],
                                          alpha=self.P3alpha_args['alpha'],
                                          normalize_similarity=self.P3alpha_args['normalize'])

        self.ALS = AlternatingLeastSquare()

        self.RP3alpha = None

        self.SlimElasticNet = None

    def fit(self, URM_train, UCM):
        self.URM_train = URM_train.copy()

        ########### FITTING ##########
        print("Fitting ItemCF...")
        self.itemCF.fit(self.URM_train.copy())

        print("Fitting UserCBF...")
        self.userCBF.fit(self.URM_train.copy(), UCM)

        print("Fitting ALS...")
        self.ALS.fit(self.URM_train.copy())

        print("Fitting P3Alpha")
        self.P3alpha.fit(self.URM_train.copy())

        print("Fitting RP3Alpha")
        self.RP3alpha = RP3betaRecommender(self.URM_train.copy(), verbose=False)
        self.RP3alpha.fit(alpha=0.032949920239451876, beta=0.14658580479486563, normalize_similarity=True, topK=75)

        print("Fitting SlimElasticNet")
        self.SlimElasticNet = SLIMElasticNetRecommender(self.URM_train)
        self.SlimElasticNet.fit(topK=150, l1_ratio=0.00622, alpha=0.00308, positive_only=True, max_iter=40)

        print("Done fitting models...")

    def recommend(self, user_id, at=10, exclude_seen=True):
        self.itemCF_scores = self.itemCF.get_expected_ratings(user_id)
        self.ALS_scores = self.ALS.get_expected_ratings(user_id)
        self.P3alpha_scores = self.P3alpha.get_expected_ratings(user_id)
        self.RP3alpha_scores = self.RP3alpha.get_expected_ratings(user_id)
        self.SlimEl_scores = self.SlimElasticNet.get_expected_ratings(user_id)

        start_pos = self.URM_train.indptr[user_id]
        end_pos = self.URM_train.indptr[user_id + 1]

        score = 0

        if len(self.URM_train.indices[start_pos:end_pos]) == 0:
            score = self.userCBF.get_expected_ratings(user_id)

        elif 0 < len(self.URM_train.indices[start_pos:end_pos]) <= 2:
            score = self.itemCF_scores * self.weight_initial['item_cf']
            score += self.ALS_scores * self.weight_initial['ALS']
            score += self.P3alpha_scores * self.weight_initial['P3Alpha']
            score += self.RP3alpha_scores * self.weight_initial['RP3Alpha']
            score += self.SlimEl_scores * self.weight_initial['SlimElasticNet']

        elif 2 < len(self.URM_train.indices[start_pos:end_pos]) <= 5:
            score = self.itemCF_scores * self.weight_middle['item_cf']
            score += self.ALS_scores * self.weight_middle['ALS']
            score += self.P3alpha_scores * self.weight_middle['P3Alpha']
            score += self.RP3alpha_scores * self.weight_middle['RP3Alpha']
            score += self.SlimEl_scores * self.weight_middle['SlimElasticNet']

        elif 5 < len(self.URM_train.indices[start_pos:end_pos]):
            score = self.itemCF_scores * self.weight_end['item_cf']
            score += self.ALS_scores * self.weight_end['ALS']
            score += self.P3alpha_scores * self.weight_end['P3Alpha']
            score += self.RP3alpha_scores * self.weight_end['RP3Alpha']
            score += self.SlimEl_scores * self.weight_end['SlimElasticNet']

        if exclude_seen:
            score = self._filter_seen(user_id, score)

        ranking = score.argsort()[::-1]

        return ranking[:at]

################################################ Test ##################################################
if __name__ == '__main__':
    test = True
    split_users = True
    max_map = 0
    data = get_data()

    group_cold = None
    group_one = None
    group_two = None
    group_three = None

    userCBF_args = {
        'topK' : 1000,
        'shrink' : 7950
    }

    P3alpha_args = {
        'topK': 66,
        'alpha': 0.2731573847973295,
        'normalize': True
    }

    itemCF_args = {
        'topK': 12,
        'shrink': 88,
        'similarity': 'tversky',
        'normalize': True,
        'fw': 'none',
        'tversky_alpha': 0.12331166243379268,
        'tversky_beta': 1.9752288743799558,
        'asymmetric_alpha': 0.0
    }

    weights_initial = {
        'user_cf' : 0,
        'item_cf' : 0.9,
        'ALS' : 0.1,
        'P3Alpha' : 0.9,
        'RP3Alpha': 0.9,
        'SlimElasticNet':1
    }

    weights_middle = {
        'user_cf' : 0,
        'item_cf' : 0.9,
        'ALS' : 0.1,
        'P3Alpha': 0.9,
        'RP3Alpha': 0.9,
        'SlimElasticNet':1
    }

    weights_end = {
        'user_cf' : 0,
        'item_cf' : 0.9,
        'ALS' : 0.1,
        'P3Alpha' : 0.9,
        'RP3Alpha': 0.9,
        'SlimElasticNet':1
    }

    hyb = HybridRecommender(weights=[weights_initial, weights_middle, weights_end],
                            itemCF_args=itemCF_args,
                            userCBF_args=userCBF_args,
                            P3alpha_args=P3alpha_args)

    if test:

        hyb.fit(data['train'].tocsr(), data['UCM'].tocsr())

        if split_users:

            group_cold, group_one, group_two, group_three = get_target_users_group(data['target_users'], data['train'])

            result_cold = hyb.evaluate_MAP_target(data['test'], group_cold)
            result_one = hyb.evaluate_MAP_target(data['test'], group_one)
            result_two = hyb.evaluate_MAP_target(data['test'], group_two)
            result_three = hyb.evaluate_MAP_target(data['test'], group_three)

            print(f'Total MAP: {result_cold["MAP"] + result_one["MAP"] + result_two["MAP"] + result_three["MAP"]:.5f}')

        elif not split_users:
            hyb.evaluate_MAP_target(data['test'], data['target_users'])

        print("\nInitial {}".format(weights_initial))
        print("Middle {}".format(weights_middle))
        print("End {}".format(weights_end))

    else:
        URM_final = data['train'] + data['test']
        URM_final = URM_final.tocsr()
        hyb.fit(URM_final, data['UCM'].tocsr())
        write_output(hyb, target_user_list=data['target_users'])
################################################ Test ##################################################