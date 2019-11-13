import numpy as np
from Algorithms.Notebooks_utils.Compute_Similarity_Python import Compute_Similarity_Python


class UserBasedCollaborativeFiltering(object):
    """
    UserBasedCollaborativeFiltering recommender system
    """

    def __init__(self, URM, topK, shrink):
        print("UserBasedCF created!")
        self.URM = URM
        self.topK = topK
        self.shrink = shrink

    def fit(self, normalize=True, similarity="cosine"):
        similarity_object = Compute_Similarity_Python(self.URM.T,
                                                      self.shrink,
                                                      self.topK,
                                                      normalize=normalize,
                                                      similarity=similarity)
        # Compute the similarity matrix (express with a score the similarity between two items
        self.W_sparse = similarity_object.compute_similarity()

    def recommend(self, user_id, at=None, exclude_seen=True):
        # Use dot product to compute the scores of the items
        scores = self.W_sparse[user_id, :].dot(self.URM).toarray().ravel()

        if exclude_seen:
            scores = self.filter_seen(user_id, scores)

        #rank items
        ranking = scores.argsort()[::-1]

        return ranking[:at]

    def filter_seen(self, user_id, scores):
        start_pos = self.URM.indptr[user_id]
        end_pos =self.URM.indptr[user_id + 1]

        user_profile = self.URM.indices[start_pos:end_pos]
        scores[user_profile] = -np.inf

        return scores
