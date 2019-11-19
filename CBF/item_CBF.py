from Algorithms.Notebooks_utils.Compute_Similarity_Python import Compute_Similarity_Python
from Algorithms.Notebooks_utils.evaluation_function import evaluate_algorithm
import numpy as np

class ItemContentBasedRecommender(object):

    def __init__(self, ICM, URM_train, topK, shrink):
        self.ICM = ICM
        self.URM_train = URM_train
        self.W_sparse = None
        self.topK = topK
        self.shrink = shrink

    def fit(self, normalize=True):
        similarity_object = Compute_Similarity_Python(self.ICM.T, shrink=self.shrink, topK=self.topK, normalize=normalize)
        self.W_sparse = similarity_object.compute_similarity()

    def recommend(self, user_id , at=None, exclude_seen=True):
        # compute the scores using the dot product
        user_profile = self.URM_train[user_id]

        scores = user_profile.dot(self.W_sparse).toarray().ravel()

        if exclude_seen:
            scores = self.filter_seen(user_id, scores)

        # rank items
        ranking = scores.argsort()[::-1]

        return ranking[:at]

    def get_scores(self, user_id):
        user_profile = self.URM_train[user_id]

        scores = user_profile.dot(self.W_sparse).toarray().ravel()

        maximum = np.amax(scores)

        normalized_scores = np.true_divide(scores, maximum)

        return normalized_scores

    def filter_seen(self, user_id, scores):
        start_pos = self.URM_train.indptr[user_id]
        end_pos = self.URM_train.indptr[user_id + 1]

        playlist_profile = self.URM_train.indices[start_pos:end_pos]

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
