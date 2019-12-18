from Algorithms.Notebooks_utils.evaluation_function import evaluate_MAP, evaluate_MAP_target_users
from Algorithms.GraphBased.P3alphaRecommender import P3alphaRecommender
from Recommenders.CF.item_cf import ItemBasedCollaborativeFiltering
from Recommenders.CF.user_cf import UserBasedCollaborativeFiltering
from Recommenders.CBF.item_CBF import ItemContentBasedRecommender
from Recommenders.CBF.user_CBF import UserContentBasedRecommender
from Recommenders.MF.ALS import AlternatingLeastSquare
from Recommenders.Graph.P3GraphRecommender import P3alphaRecommender
from Recommenders.SLIM.SLIM_BPR_Cython import SLIM_BPR_Cython
from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.NonPersonalized.top_pop import TopPop
from Utils.Toolkit import get_data, get_target_users_group
from Utils.OutputWriter import write_output
import numpy as np


class HybridRecommender(BaseRecommender):

    RECOMMENDER_NAME = "HYB"

    def __init__(self, weights=None, userCF_args=None, itemCBF_args=None,
                 itemCF_args=None, SLIM_BPR_args=None, P3alpha_args=None, userCBF_args=None):

        super().__init__()
        ######################## URM ########################
        self.URM_train = None

        ######################## Weights ########################
        self.weight_initial = weights[0]
        self.weight_middle = weights[1]
        self.weight_end = weights[2]

        ######################## Args ########################
        self.userCF_args = userCF_args
        self.itemCBF_args = itemCBF_args
        self.itemCF_args = itemCF_args
        self.SLIM_BPR_args = SLIM_BPR_args
        self.userCBF_args = userCBF_args
        self.P3alpha_args = P3alpha_args

        ######################## Scores ########################
        self.userCF_scores = None
        self.itemCF_scores = None
        self.SLIM_BPR_scores = None
        self.itemCBF_scores = None
        self.ALS_scores = None
        self.P3alpha_scores =  None

        ######################## Collaborative Filtering ########################
        #self.userCF = UserBasedCollaborativeFiltering(topK=self.userCF_args['topK'], shrink=self.userCF_args['shrink'])
        self.itemCF = ItemBasedCollaborativeFiltering(topK=self.itemCF_args['topK'],
                                                      shrink=self.itemCF_args['shrink'],
                                                      feature_weighting=self.itemCF_args['fw'],
                                                      similarity=self.itemCF_args['similarity'],
                                                      tversky_alpha=self.itemCF_args['alpha'],
                                                      tversky_beta=self.itemCF_args['beta'],
                                                      asymmetric_alpha=self.itemCF_args['a_alpha'])

        self.userCBF = UserContentBasedRecommender(topK=self.userCBF_args['topK'], shrink=self.userCBF_args['shrink'])

        #self.itemCBF = ItemContentBasedRecommender(topK=self.itemCBF_args['topK'], shrink=self.itemCBF_args['shrink'])

        self.SLIM_BPR = SLIM_BPR_Cython(epochs=self.SLIM_BPR_args['epochs'],
                                topK=self.SLIM_BPR_args['topK'],
                                lambda_i=self.SLIM_BPR_args['lambda_i'],
                                lambda_j=self.SLIM_BPR_args['lambda_j'],
                                positive_threshold=1,
                                sgd_mode=self.SLIM_BPR_args['sgd_mode'],
                                symmetric=self.SLIM_BPR_args['symmetric'],
                                learning_rate=self.SLIM_BPR_args['learning_rate'],
                                batch_size=1000)

        self.P3alpha = P3alphaRecommender(topK=self.P3alpha_args['topK'],
                                          alpha=self.P3alpha_args['alpha'],
                                          normalize_similarity=self.P3alpha_args['normalize'])

        self.ALS = AlternatingLeastSquare()

    def fit(self, URM_train, ICM, UCM):
        self.URM_train = URM_train.copy()

        ########### FITTING ##########
        #self.userCF.fit(self.URM_train.copy())

        print("Fitting ItemCF...")
        self.itemCF.fit(self.URM_train.copy())

        print("Fitting UserCBF...")
        self.userCBF.fit(self.URM_train.copy(), UCM)

        print("Fitting SLIM...")
        self.SLIM_BPR.fit(self.URM_train.copy())
        #self.itemCBF.fit(self.URM_train.copy(), ICM)

        print("Fitting ALS...")
        self.ALS.fit(self.URM_train.copy())

        print("Fitting P3Alpha")
        self.P3alpha.fit(self.URM_train.copy())

        print("Done fitting models...")

    def recommend(self, user_id, at=10, exclude_seen=True):
        #self.userCF_scores = self.userCF.get_expected_ratings(user_id)
        self.itemCF_scores = self.itemCF.get_expected_ratings(user_id)
        self.SLIM_BPR_scores = self.SLIM_BPR.get_expected_ratings(user_id)
        #self.itemCBF_scores = self.itemCBF.get_expected_ratings(user_id)
        self.ALS_scores = self.ALS.get_expected_ratings(user_id)
        self.P3alpha_scores = self.P3alpha.get_expected_ratings(user_id)

        start_pos = self.URM_train.indptr[user_id]
        end_pos = self.URM_train.indptr[user_id + 1]

        score = 0

        if len(self.URM_train.indices[start_pos:end_pos]) == 0:
            score = self.userCBF.get_expected_ratings(user_id)

        elif 0 < len(self.URM_train.indices[start_pos:end_pos]) <= 2:
            score = self.itemCF_scores * self.weight_initial['item_cf']
            score += self.SLIM_BPR_scores * self.weight_initial['SLIM_BPR']
            score += self.ALS_scores * self.weight_initial['ALS']
            score += self.P3alpha_scores * self.weight_initial['P3Alpha']
            # scores += self.userCF_scores * self.weight['user_cf']
            # scores += self.itemCBF_scores * self.weight['item_cbf']

        elif 2 < len(self.URM_train.indices[start_pos:end_pos]) <= 5:
            score = self.itemCF_scores * self.weight_middle['item_cf']
            score += self.SLIM_BPR_scores * self.weight_middle['SLIM_BPR']
            score += self.ALS_scores * self.weight_middle['ALS']
            score += self.P3alpha_scores * self.weight_middle['P3Alpha']
            # scores += self.userCF_scores * self.weight['user_cf']
            # scores += self.itemCBF_scores * self.weight['item_cbf']

        elif 5 < len(self.URM_train.indices[start_pos:end_pos]):
            score = self.itemCF_scores * self.weight_end['item_cf']
            score += self.SLIM_BPR_scores * self.weight_end['SLIM_BPR']
            score += self.ALS_scores * self.weight_end['ALS']
            score += self.P3alpha_scores * self.weight_end['P3Alpha']
            # scores += self.userCF_scores * self.weight['user_cf']
            # scores += self.itemCBF_scores * self.weight['item_cbf']

        if exclude_seen:
            score = self._filter_seen(user_id, score)

        ranking = score.argsort()[::-1]

        return ranking[:at]

################################################ Test ##################################################
if __name__ == '__main__':
    test = False
    split_users = True
    max_map = 0
    data = get_data()

    group_cold = None
    group_one = None
    group_two = None
    group_three = None

    userCF_args = {
        'topK' : 102,
        'shrink' : 7
    }

    userCBF_args = {
        'topK' : 1000,
        'shrink' : 7950
    }

    P3alpha_args = {
        'topK' : 66,
        'alpha': 0.2731573847973295,
        'normalize' : True
    }

    itemCF_args = {
        'topK': 15,
        'shrink': 986,
        'fw': 'TF-IDF',
        'similarity': 'asymmetric',
        'a_alpha': 0.30904474725892556,
        'alpha': 0.0,
        'beta': 0.0
    }

    itemCBF_args = {
        'topK' : 10,
        'shrink' : 986
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

    weights_initial = {
        'user_cf' : 0,
        'item_cf' : 1.55,
        'SLIM_BPR' : 1.45,
        'item_cbf' : 0,
        'ALS' : 0.6,
        'P3Alpha' : 1.5
    }

    weights_middle = {
        'user_cf' : 0,
        'item_cf' : 1.55,
        'SLIM_BPR' : 1.62,
        'item_cbf' : 0,
        'ALS' : 0.6,
        'P3Alpha': 0.9
    }

    weights_end = {
        'user_cf' : 0,
        'item_cf' : 1.55,
        'SLIM_BPR' : 0.4,
        'item_cbf' : 0,
        'ALS' : 0.1,
        'P3Alpha' : 2
    }

    hyb = HybridRecommender(weights=[weights_initial, weights_middle, weights_end],
                            userCF_args=userCF_args,
                            SLIM_BPR_args=SLIM_BPR_args,
                            itemCF_args=itemCF_args,
                            itemCBF_args=itemCBF_args,
                            userCBF_args=userCBF_args,
                            P3alpha_args=P3alpha_args)

    if test:

        hyb.fit(data['train'].tocsr(), data['ICM_subclass'].tocsr(), data['UCM'].tocsr())

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
        hyb.fit(URM_final, data['ICM_subclass'].tocsr(), data['UCM'].tocsr())
        write_output(hyb, target_user_list=data['target_users'])
################################################ Test ##################################################