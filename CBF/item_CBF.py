from Algorithms.Notebooks_utils.Compute_Similarity_Python import Compute_Similarity_Python
from Algorithms.Notebooks_utils.evaluation_function import evaluate_algorithm
import numpy as np
import pandas as pd
import scipy.sparse as sps


class ItemContentBasedRecommender(object):

    def __init__(self, URM_train, topk, shrink):
        item_subclass_file_path = "./data/data_ICM_sub_class.csv"
        df = pd.read_csv(item_subclass_file_path)
        self.value = list(df['data'])
        self.itemList = list(df['row'])
        self.featureList = list(df['col'])

        self.ICM = sps.coo_matrix((self.value, (self.itemList, self.featureList)))
        self.URM_train = URM_train
        self.W_sparse = None
        self.topk = None
        self.shrink = None

    def fit(self, normalize=True):
        similarity_object = Compute_Similarity_Python(self.ICM.T, shrink=self.shrink, topK=self.topk, normalize=normalize)

        self.W_sparse = similarity_object.compute_similarity()

    def recommend(self, user_id , at=None, exclude_seen=True):
        # compute the scores using the dot product
        playlist_profile = self.URM[user_id]

        scores = playlist_profile.dot(self.W_sparse).toarray().ravel()

        if exclude_seen:
            scores = self.filter_seen(user_id, scores)

        # rank items
        ranking = scores.argsort()[::-1]

        return ranking[:at]

    def get_scores(self, user_id):
        user_profile = self.URM[user_id]

        scores = user_profile.dot(self.W_sparse).toarray().ravel()

        maximum = np.amax(scores)

        normalized_scores = np.true_divide(scores, maximum)

        return normalized_scores

    def filter_seen(self, user_id, scores):
        start_pos = self.URM.indptr[user_id]
        end_pos = self.URM.indptr[user_id + 1]

        playlist_profile = self.URM.indices[start_pos:end_pos]

        scores[playlist_profile] = -np.inf

        return scores

    def evaluate(self, URM_test):
        result_dict = evaluate_algorithm(URM_test, self)
        map_result = result_dict['MAP']
        print("Item CBF -> MAP: {:.4f} with TopK = {} "
              "& Shrink = {}\t".format(map_result, self.get_topK(), self.get_shrink()))

    def get_topK(self):
        return self.topK

    def get_shrink(self):
        return self.shrink
