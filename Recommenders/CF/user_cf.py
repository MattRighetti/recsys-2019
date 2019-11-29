import numpy as np
from Algorithms.Base.Similarity.Cython.Compute_Similarity_Cython import Compute_Similarity_Cython
from Algorithms.Notebooks_utils.evaluation_function import evaluate_MAP, evaluate_MAP_target_users
from Recommenders.BaseRecommender import BaseRecommender
from Utils.Toolkit import get_data, get_URM_TFIDF


class UserBasedCollaborativeFiltering(BaseRecommender):
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
        super().__init__()
        self.URM_train = None
        self.topK = topK
        self.shrink = shrink
        self.SM_users = None
        self.RM = None

    def get_similarity_matrix(self, similarity='cosine'):
        # Similarity on URM_train.transpose()
        similarity_object = Compute_Similarity_Cython(self.URM_train.T,
                                                      topK = self.topK,
                                                      shrink=self.shrink,
                                                      normalize = True,
                                                      asymmetric_alpha = 0.5,
                                                      tversky_alpha = 1.0,
                                                      tversky_beta = 1.0,
                                                      similarity = "cosine",
                                                      row_weights = None)
        return similarity_object.compute_similarity()

    def fit(self, URM_train):
        """
        Fits model, calculates Similarity Matrix and Recommended Matrix
        :param URM_train: URM_train MUST BE IN CSR
        :return:
        """
        self.URM_train = URM_train.copy()
        self.URM_train = get_URM_TFIDF(self.URM_train)
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
        return result

    def evaluate_MAP_target(self, URM_test, target_user_list):
        result = evaluate_MAP_target_users(URM_test, self, target_user_list)
        print("UserCF -> MAP: {:.4f} with TopK = {} & Shrink = {}\t".format(result, self.topK, self.shrink))
        return result


################################################ Test ##################################################
# best_values = {'topK': 94, 'shrink': 19}
# max_map = 0
# data = get_data(dir_path='../')
#
# for topK in range(1):
#     for shrink in range(1):
#
#         args = {
#             'topK' : 902,
#             'shrink' : 1111
#         }
#
#         userCF = UserBasedCollaborativeFiltering(args['topK'], args['shrink'])
#         userCF.fit(data['train'])
#         result = userCF.evaluate_MAP_target(data['test'], data['target_users'])
#
#         if result > max_map:
#             max_map = result
#             print(f'Best values {args}')

#URM_final = data['train'] + data['test']
#URM_final = URM_final.tocsr()

#print(type(URM_final))
#hyb.fit(URM_final)
#write_output(hyb, target_user_list=data['target_users'])
################################################ Test ##################################################