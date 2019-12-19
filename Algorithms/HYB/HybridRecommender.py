from Utils.Toolkit import get_data
from Algorithms.Base.Evaluation.Evaluator import EvaluatorHoldout
from Algorithms.Data_manager.Split_functions.split_train_validation_leave_k_out import split_train_leave_k_out_user_wise
from Algorithms.Data_manager.Kaggle.KaggleDataReader import KaggleDataReader
from Algorithms.Data_manager.DataSplitter_leave_k_out import DataSplitter_leave_k_out
from Utils.OutputWriter import write_output

from Algorithms.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Algorithms.KNN.UserKNNCBFRecommender import UserKNNCBFRecommender
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
        self.SLIM_BPR = None
        self.RP3 = None
        self.P3 = None
        self.userCBF = None


    def fit(self, itemCF_args = None, SLIM_args = None, RP3_args = None, P3_args = None, weight_itemcf = 0.0,
            weight_slim = 0.0, weight_p3 = 0.0, weight_rp3 = 0.0):

        self.weight_itemcf = weight_itemcf
        self.weight_slim = weight_slim
        self.weight_p3 = weight_p3
        self.weight_rp3 = weight_rp3

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

        SLIM_args = {
            'topK': 20,
            'lambda_i': 0.01,
            'lambda_j': 0.007,
            'epochs': 100,
            'learning_rate': 1e-4,
            'symmetric': True,
            'sgd_mode': 'adagrad'
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

        ############################ INIT ############################
        if self.verbose:
            print("Initialising models...")

        self.itemCF = ItemKNNCFRecommender(self.URM_train, verbose=False)
        self.SLIM_BPR = SLIM_BPR_Cython(self.URM_train, verbose=False)
        self.RP3 = RP3betaRecommender(self.URM_train, verbose=False)
        self.P3 = P3alphaRecommender(self.URM_train, verbose=False)
        self.userCBF = UserKNNCBFRecommender(self.URM_train, self.UCM, verbose=False)

        ############################ FIT #############################
        if self.verbose:
            print("Fitting Item CF")

        self.itemCF.fit(topK=itemCF_args['topK'],
                        shrink=itemCF_args['shrink'],
                        similarity=itemCF_args['similarity'],
                        feature_weighting=itemCF_args['fw'],
                        tversky_alpha=itemCF_args['tversky_alpha'],
                        tversky_beta=itemCF_args['tversky_beta'],
                        asymmetric_alpha=itemCF_args['asymmetric_alpha'])

        if self.verbose:
            print("Fitting SLIM BPR")

        self.SLIM_BPR.fit(epochs=SLIM_args['epochs'],
                          topK=SLIM_args['topK'],
                          lambda_i=SLIM_args['lambda_i'],
                          lambda_j=SLIM_args['lambda_j'],
                          sgd_mode=SLIM_args['sgd_mode'],
                          symmetric=SLIM_args['symmetric'],
                          learning_rate=SLIM_args['learning_rate'])

        if self.verbose:
            print("Fitting P3")

        self.P3.fit(topK=P3_args['topK'],
                    alpha=P3_args['alpha'],
                    normalize_similarity=P3_args['normalize'])

        if self.verbose:
            print("Fitting RP3")

        self.RP3.fit(alpha=RP3_args['alpha'],
                     beta=RP3_args['beta'],
                     topK=RP3_args['topK'],
                     normalize_similarity=RP3_args['normalize_similarity'])

        self.userCBF.fit(topK=userCBF_args['topK'], shrink=userCBF_args['shrink'])

    def _compute_weighted_scores(self, user_id_array):
        itemCF_scores = self.itemCF._compute_item_score(user_id_array)
        SLIM_scores = self.SLIM_BPR._compute_item_score(user_id_array)
        P3_scores = self.P3._compute_item_score(user_id_array)
        RP3_scores = self.RP3._compute_item_score(user_id_array)

        scores = itemCF_scores * self.weight_itemcf
        scores += SLIM_scores * self.weight_slim
        scores += P3_scores * self.weight_p3
        scores += RP3_scores * self.weight_rp3

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

    def is_cold_user(self, user_id):
        return user_id in self.cold_users

if __name__ == '__main__':

    itemCF_args = {
        'topK': 12,
        'shrink': 88,
        'similarity': 'tversky',
        'normalize': True,
        'fw' : 'none',
        'tversky_alpha': 0.12331166243379268,
        'tversky_beta': 1.9752288743799558,
        'asymmetric_alpha' : 0.0
    }

    SLIM_args = {
        'topK': 20,
        'lambda_i': 0.01,
        'lambda_j': 0.007,
        'epochs': 100,
        'learning_rate': 1e-4,
        'symmetric': True,
        'sgd_mode': 'adagrad'
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

    weight_itemcf = 3.7076880415191917
    weight_slim = 1.013619998005147
    weight_p3 = 2.2500050504760125
    weight_rp3 = 3.3606074884992196

    train, test = split_train_leave_k_out_user_wise(get_data()['URM_all'], k_out=1)
    ucm = get_data()['UCM']

    evaluator = EvaluatorHoldout(test, [10], target_users=get_data()['target_users'])

    hybrid = HybridRecommender(train, ucm)
    hybrid.fit(weight_itemcf=weight_itemcf, weight_slim=weight_slim, weight_p3=weight_p3, weight_rp3=weight_rp3)

    result, result_string = evaluator.evaluateRecommender(hybrid)
    print(f"MAP: {result[10]['MAP']:.5f}")