from Algorithms.Base.BaseRecommender import BaseRecommender
from implicit.als import AlternatingLeastSquares
import numpy as np

from Utils.Toolkit import get_data
from Algorithms.Base.Evaluation.Evaluator import EvaluatorHoldout
from Algorithms.Data_manager.Split_functions.split_train_validation_leave_k_out import split_train_leave_k_out_user_wise
from Algorithms.Data_manager.Kaggle.KaggleDataReader import KaggleDataReader
from Algorithms.Data_manager.DataSplitter_leave_k_out import DataSplitter_leave_k_out
from Utils.OutputWriter import write_output

class ALSRecommender(BaseRecommender):

    def __init__(self, URM_train, verbose=True):
        super().__init__(URM_train, verbose)


    def fit(self, n_factors=300, regularization=0.55, iterations=30, alpha_val=15):
        W_sparse = self.URM_train.T

        model = AlternatingLeastSquares(factors=n_factors, regularization=regularization,
                                        iterations=iterations)

        data_conf = (W_sparse * alpha_val).astype('double')

        model.fit(data_conf)

        self.user_factors = model.user_factors
        self.item_factors = model.item_factors

    def _compute_item_score(self, user_id_array, items_to_compute = None):

        if np.isscalar(user_id_array):
            score = np.dot(self.user_factors[user_id_array], self.item_factors.T)
            score = np.squeeze(score)
            return score

        scores_list = []

        for user_id in user_id_array:
            scores = np.dot(self.user_factors[user_id], self.item_factors.T)
            scores = np.squeeze(scores)
            scores_list.append(scores)

        return np.asarray(scores_list, dtype=np.float32)


if __name__ == '__main__':

    train, test = split_train_leave_k_out_user_wise(get_data()['URM_all'], k_out=1)
    evaluator = EvaluatorHoldout(test, [10], target_users=get_data()['target_users'])

    als = ALSRecommender(train)
    als.fit()

    result, result_string = evaluator.evaluateRecommender(als)
    print(f"MAP: {result[10]['MAP']:.5f}")