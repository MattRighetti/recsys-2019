from Utils.Toolkit import get_static_data
import os.path
from tqdm import tqdm
from Algorithms.Base.Evaluation.Evaluator import EvaluatorHoldout
from Algorithms.Data_manager.Split_functions.split_train_validation_leave_k_out import split_train_leave_k_out_user_wise
from Algorithms.Data_manager.DataSplitter_leave_k_out import DataSplitter_leave_k_out
from Utils.OutputWriter import write_output

from Algorithms.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Algorithms.KNN.UserKNNCBFRecommender import UserKNNCBFRecommender
from Algorithms.MatrixFactorization.ALSRecommender import ALSRecommender
from Algorithms.GraphBased.P3alphaRecommender import P3alphaRecommender
from Algorithms.GraphBased.RP3betaRecommender import RP3betaRecommender
from Algorithms.SLIM_ElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender

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
        self.slimEl = None


    def fit(self, weight_itemcf = 0.0, weight_p3 = 0.0, weight_rp3 = 0.0, weight_als = 0.0, weight_slimel = 0.0):

        saved_model_path = '/Users/mattiarighetti/Developer/PycharmProjects/recsys/Algorithms/HYB/saved_models/'

        self.weight_itemcf = weight_itemcf
        self.weight_p3 = weight_p3
        self.weight_rp3 = weight_rp3
        self.weight_als = weight_als
        self.weight_slimel = weight_slimel

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

        SLIMElasticNet_args = {
            'topK': 852,
            'l1_ratio': 1.0351291192456847e-05,
            'alpha': 0.002222232489404766
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
            'n_factors': 433,
            'iterations': 29,
            'regularization': 1.707545716729426e-05,
            'alpha_val': 5
        }

        ############################ INIT ############################
        if self.verbose:
            print("Initialising models...", end='\r')

        self.itemCF = ItemKNNCFRecommender(self.URM_train, verbose=False)
        self.RP3 = RP3betaRecommender(self.URM_train, verbose=False)
        self.P3 = P3alphaRecommender(self.URM_train, verbose=False)
        self.userCBF = UserKNNCBFRecommender(self.URM_train, self.UCM, verbose=False)
        self.ALS = ALSRecommender(self.URM_train, verbose = False)
        self.slimEl = SLIMElasticNetRecommender(self.URM_train, verbose = False)

        ############################ FIT #############################
        if os.path.isfile(f'{saved_model_path}{self.itemCF.RECOMMENDER_NAME}.zip'):
            self.itemCF.load_model(saved_model_path, f'{self.itemCF.RECOMMENDER_NAME}.zip')
        else:
            if self.verbose:
                print("Fitting Item CF", end='\r')
            self.itemCF.fit(topK=itemCF_args['topK'],
                            shrink=itemCF_args['shrink'],
                            similarity=itemCF_args['similarity'],
                            feature_weighting=itemCF_args['fw'],
                            tversky_alpha=itemCF_args['tversky_alpha'],
                            tversky_beta=itemCF_args['tversky_beta'],
                            asymmetric_alpha=itemCF_args['asymmetric_alpha'])

            self.itemCF.save_model(saved_model_path, self.itemCF.RECOMMENDER_NAME)

        if os.path.isfile(
                f'{saved_model_path}{self.P3.RECOMMENDER_NAME}.zip'):
            self.P3.load_model(saved_model_path, f'{self.P3.RECOMMENDER_NAME}.zip')
        else:
            if self.verbose:
                print("Fitting P3", end='\r')
            self.P3.fit(topK=P3_args['topK'],
                        alpha=P3_args['alpha'],
                        normalize_similarity=P3_args['normalize'])
            self.P3.save_model(saved_model_path, self.P3.RECOMMENDER_NAME)

        if os.path.isfile(f'{saved_model_path}{self.RP3.RECOMMENDER_NAME}.zip'):
            self.RP3.load_model(saved_model_path, f'{self.RP3.RECOMMENDER_NAME}.zip')
        else:
            if self.verbose:
                print("Fitting RP3", end='\r')
            self.RP3.fit(alpha=RP3_args['alpha'],
                         beta=RP3_args['beta'],
                         topK=RP3_args['topK'],
                         normalize_similarity=RP3_args['normalize_similarity'])
            self.RP3.save_model(saved_model_path, self.RP3.RECOMMENDER_NAME)

        if os.path.isfile(f'{saved_model_path}{self.userCBF.RECOMMENDER_NAME}.zip'):
            self.userCBF.load_model(saved_model_path, f'{self.userCBF.RECOMMENDER_NAME}.zip')
        else:
            if self.verbose:
                print("Fitting UserCBF", end='\r')
            self.userCBF.fit(topK=userCBF_args['topK'], shrink=userCBF_args['shrink'])
            self.userCBF.save_model(saved_model_path, self.userCBF.RECOMMENDER_NAME)

        if os.path.isfile(f'{saved_model_path}{self.ALS.RECOMMENDER_NAME}.zip'):
            self.ALS.load_model(saved_model_path, f'{self.ALS.RECOMMENDER_NAME}.zip')
        else:
            self.ALS.fit(n_factors=ALS_args['n_factors'],
                         regularization=ALS_args['regularization'],
                         iterations=ALS_args['iterations'],
                         alpha_val=ALS_args['alpha_val'])
            self.ALS.save_model(saved_model_path, self.ALS.RECOMMENDER_NAME)

        if os.path.isfile(f'{saved_model_path}{self.slimEl.RECOMMENDER_NAME}.zip'):
            self.slimEl.load_model(saved_model_path, f'{self.slimEl.RECOMMENDER_NAME}.zip')
        else:
            if self.verbose:
                print("Fitting SlimElasticNet", end='\r')
            self.slimEl.fit(l1_ratio=SLIMElasticNet_args['l1_ratio'], topK=SLIMElasticNet_args['topK'])
            self.slimEl.save_model(saved_model_path, self.slimEl.RECOMMENDER_NAME)

    def _compute_item_score(self, user_id_array, items_to_compute=None):

        result = [None] * len(user_id_array)

        for i in tqdm(range(len(user_id_array))):
            if user_id_array[i] in self.cold_users:
                result[i] = self.userCBF._compute_item_score(user_id_array[i]).ravel().tolist()
            else:
                itemCF_scores = self.itemCF._compute_item_score(user_id_array[i])
                P3_scores = self.P3._compute_item_score(user_id_array[i])
                RP3_scores = self.RP3._compute_item_score(user_id_array[i])
                ALS_scores = self.ALS._compute_item_score(user_id_array[i])
                SLIMElasticNet_scores = self.slimEl._compute_item_score(user_id_array[i])

                scores = itemCF_scores * self.weight_itemcf
                scores += P3_scores * self.weight_p3
                scores += RP3_scores * self.weight_rp3
                scores += ALS_scores * self.weight_als
                scores += SLIMElasticNet_scores * self.weight_slimel
                scores = scores.ravel().tolist()

                result[i] = scores

        return result

    def save_model(self, folder_path, file_name = None):
        print("Saving not implemented...")


    def _check_if_loadable(self, model):
        if os.path.isfile(f'/Users/mattiarighetti/Developer/PycharmProjects/recsys/Algorithms/HYB/saved_models/{model.RECOMMENDER_NAME}.zip'):
            model.load_model('/Users/mattiarighetti/Developer/PycharmProjects/recsys/Algorithms/HYB/saved_models/',
                             f'{model.RECOMMENDER_NAME}.zip')



if __name__ == '__main__':
    evaluate = False

    weight_itemcf = 0.06469128422082827
    weight_p3 = 0.04997541987671707
    weight_rp3 = 0.030600333541027876
    weight_als = 1.506525953103991
    weight_slimEl = 0.0

    data = get_static_data(5)
    train = data['train']
    test = data['test']
    ucm = data['UCM']

    if evaluate:
        evaluator = EvaluatorHoldout(test, [10], target_users=data['target_users'])

        hybrid = HybridRecommender(train, ucm)
        hybrid.fit(weight_itemcf=weight_itemcf, weight_p3=weight_p3, weight_rp3=weight_rp3, weight_als=weight_als, weight_slimel=weight_slimEl)

        result, result_string = evaluator.evaluateRecommender(hybrid)
        print(f"MAP: {result[10]['MAP']:.5f}")

    else:
        urm_all = train + test
        hybrid = HybridRecommender(urm_all, ucm)
        hybrid.fit(weight_itemcf=weight_itemcf, weight_p3=weight_p3, weight_rp3=weight_rp3, weight_als=weight_als, weight_slimel=weight_slimEl)
        write_output(hybrid, data['target_users'])