from Utils.Toolkit import get_data
from Algorithms.Base.Evaluation.Evaluator import EvaluatorHoldout
from Algorithms.Data_manager.Split_functions.split_train_validation_leave_k_out import split_train_leave_k_out_user_wise
from Algorithms.Data_manager.Kaggle.KaggleDataReader import KaggleDataReader
from Algorithms.Data_manager.DataSplitter_leave_k_out import DataSplitter_leave_k_out
from Utils.OutputWriter import write_output

from Algorithms.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
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

    def __init__(self, URM_train, weights = None, verbose=True):
        super(HybridRecommender, self).__init__(URM_train, verbose)

        ############## RECOMMENDERS ##############
        self.itemCF = None
        self.SLIM_BPR = None

        self.weights = weights


    def fit(self, itemCF_args = None, SLIM_args = None, RP3_args = None, P3_args = None):

        ############################ INIT ############################
        self.itemCF = ItemKNNCFRecommender(self.URM_train, verbose=False)
        self.SLIM_BPR = SLIM_BPR_Cython(self.URM_train, verbose=False)
        self.RP3 = RP3betaRecommender(self.URM_train, verbose=False)
        self.P3 = P3alphaRecommender(self.URM_train, verbose=False)

        ############################ FIT #############################
        self.itemCF.fit(topK=itemCF_args['topK'],
                        shrink=itemCF_args['shrink'],
                        similarity=itemCF_args['similarity'],
                        feature_weighting=itemCF_args['fw'],
                        tversky_alpha=itemCF_args['tversky_alpha'],
                        tversky_beta=itemCF_args['tversky_beta'],
                        asymmetric_alpha=itemCF_args['asymmetric_alpha'])

        self.SLIM_BPR.fit(epochs=SLIM_args['epochs'],
                          topK=SLIM_args['topK'],
                          lambda_i=SLIM_args['lambda_i'],
                          lambda_j=SLIM_args['lambda_j'],
                          sgd_mode=SLIM_args['sgd_mode'],
                          symmetric=SLIM_args['symmetric'],
                          learning_rate=SLIM_args['learning_rate'])

        self.P3.fit(topK=P3_args['topK'],
                    alpha=P3_args['alpha'],
                    normalize_similarity=P3_args['normalize'])

        self.RP3.fit(alpha=RP3_args['alpha'],
                     beta=RP3_args['beta'],
                     topK=RP3_args['topK'],
                     normalize_similarity=RP3_args['normalize_similarity'])


    def _compute_item_score(self, user_id_array, items_to_compute = None):
        itemCF_scores = self.itemCF._compute_item_score(user_id_array)
        SLIM_scores = self.SLIM_BPR._compute_item_score(user_id_array)
        P3_scores = self.P3._compute_item_score(user_id_array)
        RP3_scores = self.RP3._compute_item_score(user_id_array)

        scores = itemCF_scores * self.weights['item_cf']
        scores += SLIM_scores * self.weights['SLIM_BPR']
        scores += P3_scores * self.weights['P3']
        scores += RP3_scores * self.weights['RP3']

        return scores


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

    weights = {
        'item_cf': 1.55,
        'SLIM_BPR': 0.4,
        'P3' : 0.9,
        'RP3' : 0.9
    }

    train, test = split_train_leave_k_out_user_wise(get_data()['URM_all'], k_out=1)

    evaluator = EvaluatorHoldout(test, [10], target_users=get_data()['target_users'])

    hybrid = HybridRecommender(train, weights=weights)
    hybrid.fit(itemCF_args=itemCF_args, SLIM_args=SLIM_args, P3_args=P3_args, RP3_args=RP3_args)

    result, result_string = evaluator.evaluateRecommender(hybrid)
    print(f"MAP: {result[10]['MAP']:.5f}")