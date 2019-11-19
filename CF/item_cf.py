import numpy as np
from scipy.sparse import csr_matrix
from Algorithms.Base.Similarity.Compute_Similarity_Python import Compute_Similarity_Python
from Algorithms.Notebooks_utils.Cosine_Similarity_Cython import Cosine_Similarity
from Algorithms.Notebooks_utils.evaluation_function import evaluate_algorithm


class ItemBasedCollaborativeFiltering(object):
    """
    ItemBasedCollaborativeFiltering
    """

    def __init__(self, URM_train, topK, shrink):
        self.URM_train = URM_train
        self.topK = topK
        self.shrink = shrink
        self.W_sparse = None

    def set_URM_train(self, URM):
        self.URM_train = URM

    def set_topK(self, topK):
        self.topK = topK

    def set_shrink(self, shrink):
        self.shrink = shrink

    def get_topK(self):
        return self.topK

    def get_shrink(self):
        return self.shrink

    def fit(self, normalize=True, similarity='cosine'):
        if similarity == 'cosine':
            similarity_object = Cosine_Similarity(self.URM_train, self.topK)
        else:
            similarity_object = Compute_Similarity_Python(self.URM_train, self.topK, self.shrink, normalize=normalize, similarity=similarity)

        self.W_sparse = similarity_object.compute_similarity()

    def recommend(self, user_id, at=None, exclude_seen=True):
        # Compute the scores using the dot product
        user_profile = self.URM_train[user_id]
        scores = user_profile.dot(self.W_sparse).toarray().ravel()

        if exclude_seen:
            scores = self.filter_seen(user_id, scores)

        # Rank items
        ranking = scores.argsort()[::-1]

        return ranking[:at]

    def get_scores(self, user_id):
        user_profile = self.URM_train[user_id]

        scores = user_profile.dot(self.W_sparse).toarray().ravel()

        return scores

    def filter_seen(self, user_id, scores):
        """
        Function that removes items already seen by the user
        :param user_id: User ID corresponding to each row index of the URM
        :param scores: Every rating of the corresponding User ID
        :return: Scored without already seen items
        """
        target_row = user_id
        start_pos = self.URM_train.indptr[target_row]
        end_pos = self.URM_train.indptr[target_row + 1]

        user_profile = self.URM_train.indices[start_pos:end_pos]

        scores[user_profile] = -np.inf

        return scores

    def evaluate(self, URM_test):
        result_dict = evaluate_algorithm(URM_test, self)
        map_result = result_dict['MAP']
        print("UserCF -> MAP: {:.4f} with TopK = {} "
              "& Shrink = {}\t".format(map_result, self.get_topK(), self.get_shrink()))