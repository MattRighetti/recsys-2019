from Utils.Toolkit import get_URM_TFIDF, normalize_matrix, get_data, TF_IDF
from Utils.OutputWriter import write_output
from Algorithms.Base.Recommender_utils import check_matrix
from Algorithms.Notebooks_utils.evaluation_function import evaluate_MAP_target_users, evaluate_MAP
from Algorithms.Base.Similarity.Cython.Compute_Similarity_Cython import Compute_Similarity_Cython
from Algorithms.Base.Similarity.Compute_Similarity import Compute_Similarity
from Recommenders.BaseRecommender import BaseRecommender
import numpy as np


class ItemBasedCollaborativeFiltering(BaseRecommender):

    RECOMMENDER_NAME = "ItemBasedCollaborativeFiltering"

    def __init__(self, topK, shrink, feature_weighting='TF-IDF'):
        super().__init__()
        self.URM_train = None
        self.topK = topK
        self.shrink = shrink
        self.feature_weighting = feature_weighting
        self.SM_item = None
        self.RM = None
        self.UFM = None

    def get_similarity_matrix(self, similarity='tanimoto'):
        similarity_object = Compute_Similarity(self.URM_train, shrink=self.shrink, topK=self.topK, normalize=True, similarity=similarity)
        return similarity_object.compute_similarity()

    def fit(self, URM_train):
        self.URM_train = URM_train.tocsr()

        if self.feature_weighting == "TF-IDF":
            self.URM_train = self.URM_train.astype(np.float32)
            self.URM_train = TF_IDF(self.URM_train.T).T
            self.URM_train = check_matrix(self.URM_train, 'csr')

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
if __name__ == '__main__':
    max_map = 0
    data = get_data()

    test = True

    itemCF = ItemBasedCollaborativeFiltering(10, 986, feature_weighting='TF-IDF')

    if test:

        itemCF.fit(data['train'])
        result = itemCF.evaluate_MAP_target(data['test'], data['target_users'])

    else:
        URM_final = get_data()['URM_all'].tocsr()
        itemCF.fit(URM_final)
        write_output(itemCF, target_user_list=data['target_users'])
################################################ Test ##################################################