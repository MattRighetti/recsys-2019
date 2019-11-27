from Algorithms.Base.Similarity.Cython.Compute_Similarity_Cython import Compute_Similarity_Cython
from Algorithms.Notebooks_utils.evaluation_function import evaluate_algorithm, evaluate_MAP, evaluate_MAP_target_users
import numpy as np

from Utils.Toolkit import DataReader, normalize, get_URM_BM_25, get_URM_TFIDF, get_data


class UserContentBasedRecommender(object):

    def __init__(self, topK, shrink):
        self.topK = topK
        self.shrink = shrink
        self.URM_train = None
        self.UCM = None
        self.SM = None

    def compute_similarity(self, UCM, topK, shrink):
        similarity_object = Compute_Similarity_Cython(UCM, shrink, topK, True, similarity='cosine')
        return similarity_object.compute_similarity()

    def recommend(self, user_id, at=10):
        user_id = int(user_id)
        expected_ratings = self.get_expected_ratings(user_id)

        recommended_items = np.flip(np.argsort(expected_ratings), 0)
        return recommended_items[:at]

    def fit(self, URM_train, UCM):
        # PRICE IS NOT INCLUDED INTENTIONALLY
        self.URM_train = URM_train.copy()
        self.UCM = UCM
        #self.UCM_region = get_URM_TFIDF(self.UCM_region)
        #self.UCM_region = normalize(self.UCM_region)

        self.SM = self.compute_similarity(self.UCM.T, self.topK, self.shrink)

    def get_expected_ratings(self, user_id):
        features = self.URM_train.T[user_id]
        expected_ratings = features.dot(self.SM).toarray().ravel()
        expected_ratings[features.indices] = -10
        return expected_ratings

    def evaluate_MAP(self, URM_test):
        result = evaluate_MAP(URM_test, self)
        print("UserCBF -> MAP: {:.4f}".format(result))
        return result

    def evaluate_MAP_target(self, URM_test, target_user_list):
        result = evaluate_MAP_target_users(URM_test, self, target_user_list)
        print("UserCBF -> MAP: {:.4f}".format(result))
        return result


################################################ Test ##################################################
max_map = 0
data = get_data(dir_path='../')

for topK in range(90, 101, 2):
    for shrink in range(1, 20, 2):

        args = {
            'topK':topK,
            'shrink':shrink
        }

        userCF = UserContentBasedRecommender(args['topK'],
                                             args['topK'])

        userCF.fit(data['train'], data['UCM'])
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