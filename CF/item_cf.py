import numpy as np
from scipy.sparse import csr_matrix
from Algorithms.Base.Similarity.Compute_Similarity_Python import Compute_Similarity_Python

class ItemBasedCollaborativeFiltering(object):
    """
    ItemBasedCollaborativeFiltering
    """

    def __init__(self, URM, topK, shrink):
        self.URM = URM
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

    def fit(self, normalize=True, similarity='cosine'):
        similarity_object = Compute_Similarity_Python(self.URM, self.topK, self.shrink, normalize=normalize, similarity=similarity)

        self.W_sparse = similarity_object.compute_similarity()

    def recommend(self, user_id, at=None, exclude_seen=True):
        # Compute the scores using the dot product
        user_profile = self.URM[user_id]
        scores = user_profile.dot(self.W_sparse).toarray().ravel()

        if exclude_seen:
            scores = self.filter_seen(user_id, scores)

        # Rank items
        ranking = scores.argsort()[::-1]

        return ranking[:at]

    def filter_seen(self, user_id, scores):
        """
        Function that removes items already seen by the user
        :param user_id: User ID corresponding to each row index of the URM
        :param scores: Every rating of the corresponding User ID
        :return: Scored without already seen items
        """
        target_row = user_id
        start_pos = self.URM.indptr[target_row]
        end_pos = self.URM.indptr[target_row + 1]

        user_profile = self.URM.indices[start_pos:end_pos]

        scores[user_profile] = -np.inf

        return scores