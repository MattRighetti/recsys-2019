from Algorithms.Base.Similarity.Cython.Compute_Similarity_Cython import Compute_Similarity_Cython
from Algorithms.Notebooks_utils.evaluation_function import evaluate_algorithm
import numpy as np

from Utils.Toolkit import DataReader, normalize, get_URM_BM_25, get_URM_TFIDF


class UserContentBasedRecommender(object):

    def __init__(self, topK_age, topK_region, shrink_age, shrink_region, weight_region=0.5):
        self.topK_age = topK_age
        self.topK_region = topK_region
        self.shrink_age = shrink_age
        self.shrink_region = shrink_region
        self.weight_region = weight_region

        self.URM_train = None
        self.UCM_age = None
        self.UCM_region = None
        self.SM_region = None
        self.SM_age = None

    def compute_similarity(self, UCM, topK, shrink):
        similarity_object = Compute_Similarity_Cython(UCM.transpose(), shrink, topK, True, similarity='cosine')
        return similarity_object.compute_similarity()

    def recommend(self, user_id, at=10):
        user_id = int(user_id)
        expected_ratings = self.get_expected_ratings(user_id)

        recommended_items = np.flip(np.argsort(expected_ratings), 0)
        return recommended_items[:at]

    def fit(self, URM_train):
        # PRICE IS NOT INCLUDED INTENTIONALLY
        self.URM_train = URM_train
        self.UCM_age = DataReader().UCM_age_COO().tocsr()
        #self.UCM_age = get_URM_BM_25(self.UCM_age)
        self.UCM_region = DataReader().UCM_region_COO().tocsr()
        #self.UCM_region = get_URM_TFIDF(self.UCM_region)
        #self.UCM_region = normalize(self.UCM_region)

        self.SM_age = self.compute_similarity(self.UCM_age, self.topK_age, self.shrink_age)
        self.SM_region = self.compute_similarity(self.UCM_region, self.topK_region, self.shrink_region)

    def get_expected_ratings(self, user_id):
        user_id = int(user_id)
        features = self.URM_train[user_id]
        expected_ratings_age = features.dot(self.SM_age).toarray().ravel()
        expected_ratings_region = features.dot(self.SM_region).toarray().ravel()

        region_weight = self.weight_region
        age_weight = 1 - region_weight
        expected_ratings = (expected_ratings_age * age_weight) + (expected_ratings_region * region_weight)
        expected_ratings[features.indices] = -10
        return expected_ratings

    def evaluate(self, URM_test):
        result_dict = evaluate_algorithm(URM_test, self)
        map_result = result_dict['MAP']
        print("Item CBF -> MAP: {:.4f}".format(map_result))