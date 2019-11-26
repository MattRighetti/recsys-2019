from Algorithms.Base.Similarity.Cython.Compute_Similarity_Cython import Compute_Similarity_Cython
from Algorithms.Notebooks_utils.evaluation_function import evaluate_algorithm, evaluate_MAP, evaluate_MAP_target_users
import numpy as np

from Utils.Toolkit import DataReader, normalize, get_URM_BM_25, get_URM_TFIDF, get_data


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

    def fit(self, URM_train, UCM_age, UCM_region):
        # PRICE IS NOT INCLUDED INTENTIONALLY
        self.URM_train = URM_train.copy()
        self.UCM_age = UCM_age
        #self.UCM_age = get_URM_BM_25(self.UCM_age)
        self.UCM_region = UCM_region
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

    def evaluate_MAP(self, URM_test):
        result = evaluate_MAP(URM_test, self)
        print("ItemCBF -> MAP: {:.4f}".format(result))
        return result

    def evaluate_MAP_target(self, URM_test, target_user_list):
        result = evaluate_MAP_target_users(URM_test, self, target_user_list)
        print("ItemCBF -> MAP: {:.4f}".format(result))
        return result


################################################ Test ##################################################
max_map = 0
data = get_data(test=True)

for topK_age in range(90, 101, 2):
    for shrink_age in range(1, 20, 2):

        args = {
            'topK_age':topK_age,
            'shrink_age':shrink_age,
            'topK_region': topK_age,
            'shrink_region': shrink_age,
            'weight_region' : 0.5
        }

        userCF = UserContentBasedRecommender(args['topK_age'],
                                             args['topK_region'],
                                             args['shrink_age'],
                                             args['shrink_region'],
                                             args['weight_region'])

        userCF.fit(data['train'], data['UCM_age'], data['UCM_region'])
        result = userCF.evaluate_MAP_target(data['test'], data['target_users'])

        if result > max_map:
            max_map = result
            print(f'Best values {args}')

#URM_final = data['train'] + data['test']
#URM_final = URM_final.tocsr()

#print(type(URM_final))
#hyb.fit(URM_final)
#write_output(hyb, target_user_list=data['target_users'])
################################################ Test ##################################################