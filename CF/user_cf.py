import numpy as np
from Algorithms.Base.Similarity.Cython.Compute_Similarity_Cython import Compute_Similarity_Cython
from Algorithms.Notebooks_utils.evaluation_function import evaluate_MAP, evaluate_MAP_target_users


class UserBasedCollaborativeFiltering(object):
    """
    UserBasedCollaborativeFiltering recommender system
    """

    def __init__(self, topK, shrink):
        """
        Initialises a UserCF Recommender
        URM_train MUST BE IN CSR
        SM = Similarity Matrix
        RM = Recommended Matrix
        :param topK: topK Value
        :param shrink: shrink Value
        """
        self.URM_train = None
        self.topK = topK
        self.shrink = shrink
        self.SM_users = None
        self.RM = None

    def get_similarity_matrix(self, similarity='cosine'):
        # Similarity on URM_train.transpose()
        similarity_object = Compute_Similarity_Cython(self.URM_train.T,
                                                      self.shrink,
                                                      self.topK,
                                                      normalize=True,
                                                      tversky_alpha=1.0,
                                                      tversky_beta=1.0,
                                                      similarity=similarity)
        return similarity_object.compute_similarity()

    def fit(self, URM_train):
        """
        Fits model, calculates Similarity Matrix and Recommended Matrix
        :param URM_train: URM_train MUST BE IN CSR
        :return:
        """
        self.URM_train = URM_train
        self.SM_users = self.get_similarity_matrix()
        self.RM = self.SM_users.dot(self.URM_train)

    def get_expected_recommendations(self, user_id):
        expected_recommendations = self.RM[user_id].todense()
        return np.squeeze(np.asarray(expected_recommendations))

    def recommend(self, user_id, at=None, exclude_seen=True):
        expected_ratings = self.get_expected_recommendations(user_id)
        recommended_items = np.flip(np.argsort(expected_ratings), 0)

        if exclude_seen:
            unseen_items_mask = np.in1d(recommended_items, self.URM_train[user_id].indices, assume_unique=True, invert=True)
            recommended_items = recommended_items[unseen_items_mask]

        return recommended_items[:at]

    def evaluate_MAP(self, URM_test):
        result = evaluate_MAP(URM_test, self)
        print("UserCF -> MAP: {:.4f} with TopK = {} & Shrink = {}\t".format(result, self.topK, self.shrink))

    def evaluate_MAP_target(self, URM_test, target_user_list):
        result = evaluate_MAP_target_users(URM_test, self, target_user_list)
        print("UserCF -> MAP: {:.4f} with TopK = {} & Shrink = {}\t".format(result, self.topK, self.shrink))