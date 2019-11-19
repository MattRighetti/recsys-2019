import numpy as np
from Algorithms.Base.Similarity.Compute_Similarity_Python import Compute_Similarity_Python
from Algorithms.Notebooks_utils.evaluation_function import evaluate_algorithm


class UserBasedCollaborativeFiltering(object):
    """
    UserBasedCollaborativeFiltering recommender system
    """

    def __init__(self, URM_train, topK, shrink):
        """
        :param URM: URM_TRAIN!
        :param topK:
        :param shrink:
        """
        self.URM_train = URM_train
        self.topK = topK
        self.shrink = shrink
        self.W_sparse = None

    def set_URM(self, URM):
        self.URM = URM

    def set_topK(self, topK):
        self.topK = topK

    def set_shrink(self, shrink):
        self.shrink = shrink

    def get_topK(self):
        return self.topK

    def get_shrink(self):
        return self.shrink

    def fit(self, normalize=True, similarity="cosine"):
        similarity_object = Compute_Similarity_Python(self.URM_train.T, self.shrink, self.topK, normalize=normalize, similarity=similarity)
        # Compute the similarity matrix (express with a score the similarity between two items
        self.W_sparse = similarity_object.compute_similarity()

    def recommend(self, user_id, at=None, exclude_seen=True):
        # Use dot product to compute the scores of the items
        scores = self.W_sparse[user_id, :].dot(self.URM_train).toarray().ravel()

        if exclude_seen:
            scores = self.filter_seen(user_id, scores)

        #rank items
        ranking = scores.argsort()[::-1]

        return ranking[:at]

    def get_scores(self, user_id):
        user_profile = self.W_sparse[user_id, :]

        scores = user_profile.dot(self.URM_train).toarray().ravel()

        return scores

    def filter_seen(self, user_id, scores):
        start_pos = self.URM_train.indptr[user_id]
        end_pos =self.URM_train.indptr[user_id + 1]

        user_profile = self.URM_train.indices[start_pos:end_pos]
        scores[user_profile] = -np.inf

        return scores

    def evaluate(self, URM_test):
        result_dict = evaluate_algorithm(URM_test, self)
        map_result = result_dict['MAP']
        print("UserCF -> MAP: {:.4f} with TopK = {} "
              "& Shrink = {}\t".format(map_result, self.get_topK(), self.get_shrink()))