from Algorithms.Base.BaseRecommender import BaseRecommender
from implicit.als import AlternatingLeastSquares
import numpy as np

from Utils.Toolkit import get_data
from Algorithms.Base.DataIO import DataIO
from Algorithms.Base.Evaluation.Evaluator import EvaluatorHoldout
from Algorithms.Data_manager.Split_functions.split_train_validation_leave_k_out import split_train_leave_k_out_user_wise
from Algorithms.Data_manager.DataSplitter_leave_k_out import DataSplitter_leave_k_out
from Utils.OutputWriter import write_output

class ALSRecommender(BaseRecommender):

    RECOMMENDER_NAME = "ALSRecommender"

    def __init__(self, URM_train, verbose=True):
        super().__init__(URM_train, verbose)
        self.user_factors = None
        self.item_factors = None


    def fit(self, n_factors=433, regularization=1.707545716729426e-05, iterations=29, alpha_val=5):

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

        return np.asarray(scores_list)

    def save_model(self, folder_path, file_name = None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        data_dict_to_save = {"user_factors": self.user_factors, "item_factors": self.item_factors}

        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save=data_dict_to_save)

        self._print("Saving complete")


if __name__ == '__main__':

    ALS_args = {
        'n_factors': 433,
        'iterations': 29,
        'regularization': 1.707545716729426e-05,
        'alpha_val' : 5
    }

    train, test = split_train_leave_k_out_user_wise(get_data()['URM_all'], k_out=1)
    evaluator = EvaluatorHoldout(test, [10], target_users=get_data()['target_users'])

    als = ALSRecommender(train)
    als.fit(n_factors=ALS_args['n_factors'],
            regularization=ALS_args['regularization'],
            iterations=ALS_args['iterations'],
            alpha_val=ALS_args['alpha_val'])

    result, result_string = evaluator.evaluateRecommender(als)
    print(f"MAP: {result[10]['MAP']:.5f}")