from Utils.Toolkit import get_data
from Algorithms.Base.Evaluation.Evaluator import EvaluatorHoldout
from Algorithms.Data_manager.Split_functions.split_train_validation_leave_k_out import split_train_leave_k_out_user_wise
from Algorithms.Data_manager.Kaggle.KaggleDataReader import KaggleDataReader
from Algorithms.Data_manager.DataSplitter_leave_k_out import DataSplitter_leave_k_out
from Utils.OutputWriter import write_output

from Algorithms.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Algorithms.KNN.UserKNNCBFRecommender import UserKNNCBFRecommender
from Algorithms.MatrixFactorization.ALSRecommender import ALSRecommender
from Algorithms.SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from Algorithms.GraphBased.P3alphaRecommender import P3alphaRecommender
from Algorithms.GraphBased.RP3betaRecommender import RP3betaRecommender

from Algorithms.Base.Recommender_utils import check_matrix
from Algorithms.Base.BaseRecommender import BaseRecommender

from Algorithms.Base.IR_feature_weighting import okapi_BM_25, TF_IDF
import numpy as np

from Algorithms.Base.Similarity.Compute_Similarity import Compute_Similarity


class HybridRecommender(BaseRecommender):

    RECOMMENDER_NAME = "HybridRecommender"

    def __init__(self, URM_train, UCM, verbose=True):
        super(HybridRecommender, self).__init__(URM_train, verbose)

        self.verbose = verbose
        ############## UTILS ##############
        self.UCM = UCM

        self.cold_users = np.arange(URM_train.shape[0])[self._cold_user_mask]
        ############## RECOMMENDERS ##############
        self.itemCF = None
        self.RP3 = None
        self.P3 = None
        self.userCBF = None
        self.ALS = None


    def fit(self, weight_itemcf = 0.0, weight_p3 = 0.0, weight_rp3 = 0.0, weight_als = 0.0):

        self.weight_itemcf = weight_itemcf
        self.weight_p3 = weight_p3
        self.weight_rp3 = weight_rp3
        self.weight_als = weight_als

        ###################### DEFAULT VALUES #########################
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

        P3_args = {
            'topK': 66,
            'alpha': 0.2731573847973295,
            'normalize': True
        }

        RP3_args = {
            'topK': 75,
            'alpha': 0.032949920239451876,
            'beta': 0.14658580479486563,
            'normalize_similarity': True
        }

        userCBF_args = {
            'topK': 1000,
            'shrink': 7950
        }

        ALS_args = {
            'n_factors' : 300,
            'regularization' : 0.55,
            'iterations' : 30,
            'alpha_val' : 15
        }

        ############################ INIT ############################
        if self.verbose:
            print("Initialising models...", end='\r')

        self.itemCF = ItemKNNCFRecommender(self.URM_train, verbose=False)
        self.RP3 = RP3betaRecommender(self.URM_train, verbose=False)
        self.P3 = P3alphaRecommender(self.URM_train, verbose=False)
        self.userCBF = UserKNNCBFRecommender(self.URM_train, self.UCM, verbose=False)
        self.ALS = ALSRecommender(self.URM_train, verbose = False)

        ############################ FIT #############################
        if self.verbose:
            print("Fitting Item CF", end='\r')

        self.itemCF.fit(topK=itemCF_args['topK'],
                        shrink=itemCF_args['shrink'],
                        similarity=itemCF_args['similarity'],
                        feature_weighting=itemCF_args['fw'],
                        tversky_alpha=itemCF_args['tversky_alpha'],
                        tversky_beta=itemCF_args['tversky_beta'],
                        asymmetric_alpha=itemCF_args['asymmetric_alpha'])

        if self.verbose:
            print("Fitting P3", end='\r')

        self.P3.fit(topK=P3_args['topK'],
                    alpha=P3_args['alpha'],
                    normalize_similarity=P3_args['normalize'])

        if self.verbose:
            print("Fitting RP3", end='\r')

        self.RP3.fit(alpha=RP3_args['alpha'],
                     beta=RP3_args['beta'],
                     topK=RP3_args['topK'],
                     normalize_similarity=RP3_args['normalize_similarity'])

        if self.verbose:
            print("Fitting UserCBF", end='\r')

        self.userCBF.fit(topK=userCBF_args['topK'], shrink=userCBF_args['shrink'])

        self.ALS.fit(n_factors=ALS_args['n_factors'],
                     regularization=ALS_args['regularization'],
                     iterations=ALS_args['iterations'],
                     alpha_val=ALS_args['alpha_val'])

    def _compute_weighted_scores(self, user_id_array):
        itemCF_scores = self.itemCF._compute_item_score(user_id_array)
        P3_scores = self.P3._compute_item_score(user_id_array)
        RP3_scores = self.RP3._compute_item_score(user_id_array)
        ALS_scores = self.ALS._compute_item_score(user_id_array)

        scores = itemCF_scores * self.weight_itemcf
        scores += P3_scores * self.weight_p3
        scores += RP3_scores * self.weight_rp3
        scores += ALS_scores * self.weight_als

        return scores

    def _compute_item_score(self, user_id_array, items_to_compute = None):
        # If input is a single int
        if np.isscalar(user_id_array):
            if user_id_array in self.cold_users:
                return self.userCBF._compute_item_score(user_id_array)
            else:
                return self._compute_weighted_scores(user_id_array)
        # If input is an array of int
        else:

            rank_list = []

            for user_id in user_id_array:
                if user_id in self.cold_users:
                    score = self.userCBF._compute_item_score(user_id)
                    score = score.ravel()
                    rank_list.append(score)
                else:
                    score = self._compute_weighted_scores(user_id)
                    score = score.ravel()
                    rank_list.append(score)

            return np.asarray(rank_list, dtype=np.float32)

    def save_model(self, folder_path, file_name = None):
        print("Saving not implemented...")




if __name__ == '__main__':

    weight_itemcf = 0.06469128422082827
    weight_p3 = 0.04997541987671707
    weight_rp3 = 0.030600333541027876
    weight_als = 0.0

    train, test = split_train_leave_k_out_user_wise(get_data()['URM_all'], k_out=1)
    ucm = get_data()['UCM']

    evaluator = EvaluatorHoldout(test, [10], target_users=get_data()['target_users'])

    hybrid = HybridRecommender(train, ucm)
    hybrid.fit(weight_itemcf=weight_itemcf, weight_p3=weight_p3, weight_rp3=weight_rp3, weight_als=weight_als)

    result, result_string = evaluator.evaluateRecommender(hybrid)
    print(f"MAP: {result[10]['MAP']:.5f}")