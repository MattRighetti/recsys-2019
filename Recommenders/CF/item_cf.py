from Utils.Toolkit import get_URM_TFIDF, normalize_matrix, get_data, rerank_based_on_ICM, generate_SM_user_feature_matrix
from Algorithms.Notebooks_utils.evaluation_function import evaluate_MAP_target_users, evaluate_MAP
from Algorithms.Base.Similarity.Cython.Compute_Similarity_Cython import Compute_Similarity_Cython
from Recommenders.BaseRecommender import BaseRecommender
import numpy as np


class ItemBasedCollaborativeFiltering(BaseRecommender):

    RECOMMENDER_NAME = "ItemBasedCollaborativeFiltering"

    def __init__(self, topK, shrink):
        super().__init__()
        self.URM_train = None
        self.topK = topK
        self.shrink = shrink
        self.SM_item = None
        self.RM = None
        self.UFM = None

    def get_similarity_matrix(self, similarity='tanimoto'):
        similarity_object = Compute_Similarity_Cython(self.URM_train,
                                                      self.shrink,
                                                      self.topK,
                                                      normalize = True,
                                                      tversky_alpha = 1.0,
                                                      tversky_beta = 1.0,
                                                      similarity = similarity)
        return similarity_object.compute_similarity()

    def fit(self, URM_train, ICM):
        self.URM_train = URM_train.tocsr()
        #self.UFM = generate_SM_user_feature_matrix(self.URM_train.copy(), ICM)
        self.SM_item = self.get_similarity_matrix()
        self.RM = self.URM_train.dot(self.SM_item)

    def recommend(self, user_id, at=10, exclude_seen=True):
        expected_ratings = self.get_expected_ratings(user_id)
        recommended_items = np.flip(np.argsort(expected_ratings), 0)
        expected_ratings = expected_ratings[recommended_items]

        if exclude_seen:
            unseen_items_mask = np.in1d(recommended_items, self.URM_train[user_id].indices, assume_unique=True, invert=True)
            recommended_items = recommended_items[unseen_items_mask]

        #recommended_items = self.rerank_items(user_id, recommended_items, expected_ratings)

        return recommended_items[:at]

    def get_expected_ratings(self, user_id):
        expected_recommendations = self.RM[user_id].todense()
        return np.squeeze(np.asarray(expected_recommendations))

    def rerank_items(self, user_id, recommended_items, expected_ratings):
        recommended_items = recommended_items[:20]
        expected_ratings = expected_ratings[:20]
        recommended_items = rerank_based_on_ICM(self.UFM, recommended_items, expected_ratings, user_id)
        return recommended_items


################################################ Test ##################################################
# best_values_3 = {'topK': 26, 'shrink': 20}
# best_values_2 = {'topK': 26, 'shrink': 10}
# best_values_1 = {'topK': 29, 'shrink': 5}
# max_map = 0
# data = get_data()
# ICM = data['ICM_subclass'].tocsr()
#
# for topK in [29]:
#     for shrink in [5]:
#
#         args = {
#             'topK':topK,
#             'shrink':shrink
#         }
#
#         itemCF = ItemBasedCollaborativeFiltering(args['topK'], args['shrink'])
#         itemCF.fit(data['train'], ICM)
#         result = itemCF.evaluate_MAP_target(data['test'], data['target_users'])
#
#         if result['MAP'] > max_map:
#             max_map = result['MAP']
#             print(f'Best values {args}')

#URM_final = data['train'] + data['test']
#URM_final = URM_final.tocsr()

#print(type(URM_final))
#hyb.fit(URM_final)
#write_output(hyb, target_user_list=data['target_users'])
################################################ Test ##################################################