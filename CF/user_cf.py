import numpy as np
from Algorithms.Base.Similarity.Cython.Compute_Similarity_Cython import Compute_Similarity_Cython
from Algorithms.Notebooks_utils.evaluation_function import evaluate_MAP, evaluate_MAP_target_users


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
        similarity_object = Compute_Similarity_Cython(self.URM_train.T, self.shrink, self.topK, normalize=normalize, similarity=similarity)
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

        max_value = np.amax(scores)

        normalized_scores = np.true_divide(scores, max_value)

        return normalized_scores

    def filter_seen(self, user_id, scores):
        start_pos = self.URM_train.indptr[user_id]
        end_pos =self.URM_train.indptr[user_id + 1]

        user_profile = self.URM_train.indices[start_pos:end_pos]
        scores[user_profile] = -np.inf

        return scores

    def evaluate_MAP(self, URM_test):
        result = evaluate_MAP(URM_test, self)
        print("UserCF -> MAP: {:.4f} with TopK = {} "
              "& Shrink = {}\t".format(result, self.get_topK(), self.get_shrink()))

    def evaluate_MAP_target(self, URM_test, target_user_list):
        result = evaluate_MAP_target_users(URM_test, self, target_user_list)
        print("UserCF -> MAP: {:.4f} with TopK = {} "
              "& Shrink = {}\t".format(result, self.get_topK(), self.get_shrink()))