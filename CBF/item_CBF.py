from Algorithms.Base.Similarity.Cython.Compute_Similarity_Cython import Compute_Similarity_Cython
from Algorithms.Notebooks_utils.evaluation_function import evaluate_algorithm, evaluate_MAP, evaluate_MAP_target_users
import numpy as np

from Utils.Toolkit import DataReader, normalize, get_URM_BM_25, get_URM_TFIDF, get_data


class ItemContentBasedRecommender(object):

    def __init__(self, topK_asset, topK_subclass, shrink_asset, shrink_subclass, weight_subclass=0.5):
        self.topK_asset = topK_asset
        self.topK_subclass = topK_subclass
        self.shrink_asset = shrink_asset
        self.shrink_subclass = shrink_subclass
        self.weight_subclass = weight_subclass

        self.URM_train = None
        self.ICM_asset = None
        self.ICM_price = None
        self.ICM_subclass = None
        self.SM_subclass = None
        self.SM_asset = None

    def compute_similarity(self, ICM, topK, shrink):
        similarity_object = Compute_Similarity_Cython(ICM.transpose(), shrink, topK, True, similarity='cosine')
        return similarity_object.compute_similarity()

    def recommend(self, user_id, at=10):
        user_id = int(user_id)
        expected_ratings = self.get_expected_ratings(user_id)

        recommended_items = np.flip(np.argsort(expected_ratings), 0)
        return recommended_items[:at]

    def fit(self, URM_train, ICM_asset, ICM_subclass):
        # PRICE IS NOT INCLUDED INTENTIONALLY
        self.URM_train = URM_train
        self.ICM_asset = ICM_asset
        #self.ICM_asset = get_URM_BM_25(self.ICM_asset)
        self.ICM_subclass = ICM_subclass
        #self.ICM_subclass = get_URM_TFIDF(self.ICM_subclass)
        #self.ICM_subclass = normalize(self.ICM_subclass)

        self.SM_subclass = self.compute_similarity(self.ICM_asset, self.topK_asset, self.shrink_asset)
        self.SM_asset = self.compute_similarity(self.ICM_subclass, self.topK_subclass, self.shrink_subclass)

    def get_expected_ratings(self, user_id):
        user_id = int(user_id)
        interactions = self.URM_train[user_id]
        expected_ratings_asset = interactions.dot(self.SM_asset).toarray().ravel()
        expected_ratings_subclass = interactions.dot(self.SM_subclass).toarray().ravel()

        subclass_weight = self.weight_subclass
        asset_weight = 1 - subclass_weight
        expected_ratings = (expected_ratings_asset * asset_weight) + (expected_ratings_subclass * subclass_weight)
        expected_ratings[interactions.indices] = -10
        return expected_ratings

    def evaluate_MAP(self, URM_test):
        result = evaluate_MAP(URM_test, self)
        print("ItemCBF -> MAP: {:.4f}".format(result))
        return result

    def evaluate_MAP_target(self, URM_test, target_user_list):
        result = evaluate_MAP_target_users(URM_test, self, target_user_list)
        print("ItemCBF -> MAP: {:.4f}".format(result))
        return result


################################################ Test ##################################################
# max_map = 0
# data = get_data(dir_path='../')
#
# for topK_price in range(90, 101, 2):
#     for shrink_price in range(1, 20, 2):
#
#         args = {
#             'topK_price':topK_price,
#             'shrink_price':shrink_price,
#             'topK_asset': topK_price,
#             'shrink_asset': shrink_price,
#             'topK_subclass': topK_price,
#             'shrink_subclass': shrink_price,
#             'weight_subclass' : 0.5
#         }
#
#         userCF = ItemContentBasedRecommender(args['topK_asset'],
#                                              args['topK_subclass'],
#                                              args['shrink_asset'],
#                                              args['shrink_subclass'],
#                                              args['weight_subclass'])
#
#         userCF.fit(data['train'], data['ICM_asset'], data['ICM_subclass'])
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