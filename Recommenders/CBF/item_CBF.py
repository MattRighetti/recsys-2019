from Algorithms.Base.Similarity.Cython.Compute_Similarity_Cython import Compute_Similarity_Cython
from Algorithms.Notebooks_utils.evaluation_function import evaluate_algorithm, evaluate_MAP, evaluate_MAP_target_users
from Utils.Toolkit import DataReader, normalize, get_URM_BM_25, get_URM_TFIDF, get_data
from Recommenders.BaseRecommender import BaseRecommender
import numpy as np
import scipy.sparse as sps


class ItemContentBasedRecommender(BaseRecommender):

    RECOMMENDER_NAME = "ItemContentBasedRecommender"

    def __init__(self, topK, shrink):
        super().__init__()
        self.topK = topK
        self.shrink = shrink

        self.URM_train = None
        self.ICM = None
        self.SM = None

    def compute_similarity(self, ICM, topK, shrink):
        similarity_object = Compute_Similarity_Cython(ICM.transpose(), shrink, topK, True, similarity='cosine')
        return sps.csr_matrix(similarity_object.compute_similarity())

    def recommend(self, user_id, at=10):
        user_id = int(user_id)
        expected_ratings = self.get_expected_ratings(user_id)

        recommended_items = np.flip(np.argsort(expected_ratings), 0)
        return recommended_items[:at]

    def fit(self, URM_train, ICM):
        # PRICE IS NOT INCLUDED INTENTIONALLY
        self.URM_train = URM_train.copy()
        self.ICM = ICM.copy()
        self.ICM = get_URM_BM_25(self.ICM)
        #self.ICM = get_URM_TFIDF(self.ICM)
        self.ICM = normalize(self.ICM)

        self.SM = self.compute_similarity(self.ICM, self.topK, self.shrink)

    def get_expected_ratings(self, user_id):
        interactions = self.URM_train[user_id]
        expected_ratings = interactions.dot(self.SM).toarray().ravel()

        start_pos = self.URM_train.indptr[user_id]
        end_pos = self.URM_train.indptr[user_id + 1]

        user_profile = self.URM_train.indices[start_pos:end_pos]

        expected_ratings[user_profile] = -np.inf

        expected_ratings[interactions.indices] = -10
        return expected_ratings


################################################ Test ##################################################
# max_map = 0
# data = get_data(dir_path='../')
#
# for topK in range(1):
#     for shrink in range(1):
#
#         args = {
#             'topK':topK,
#             'shrink':shrink
#         }
#
#         userCF = ItemContentBasedRecommender(900,
#                                              100)
#
#         userCF.fit(data['train'], data['ICM_subclass'])
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