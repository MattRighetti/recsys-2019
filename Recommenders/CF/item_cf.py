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

    def __init__(self, topK, shrink, feature_weighting='TF-IDF', tversky_alpha=1.0, tversky_beta=1.0, asymmetric_alpha=1.0, similarity='cosine'):
        super().__init__()
        self.URM_train = None
        self.topK = topK
        self.shrink = shrink
        self.feature_weighting = feature_weighting
        self.tversky_alpha = tversky_alpha
        self.tversky_beta = tversky_beta
        self.asymmetric_alpha = asymmetric_alpha
        self.similarity=similarity

        self.SM_item = None
        self.RM = None
        self.UFM = None

    def get_similarity_matrix(self, similarity='asymmetric'):
        similarity_object = Compute_Similarity(self.URM_train,
                                               shrink=self.shrink,
                                               topK=self.topK,
                                               normalize=True,
                                               tversky_alpha=1.0,
                                               tversky_beta=1.0,
                                               asymmetric_alpha=self.asymmetric_alpha,
                                               similarity=similarity)

        return similarity_object.compute_similarity()

    def fit(self, URM_train):
        self.URM_train = URM_train.tocsr()

        if self.feature_weighting == "TF-IDF":
            self.URM_train = self.URM_train.astype(np.float32)
            self.URM_train = TF_IDF(self.URM_train.T).T
            self.URM_train = check_matrix(self.URM_train, 'csr')

        self.SM_item = self.get_similarity_matrix(similarity=self.similarity)
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

    best_asymmetric = {
        'topK': 15,
        'shrink': 986,
        'fw': 'TF-IDF',
        'similarity': 'asymmetric',
        'a_alpha' : 0.30904474725892556,
        'alpha': 0.0,
        'beta': 0.0
    }

    best_cosine = {
        'topK': 14,
        'shrink': 999,
        'similarity': 'cosine',
        'fw': 'TF-IDF',
        'a_alpha': 0.30904474725892556,
        'alpha': 0.0,
        'beta': 0.0
    }

    best_tversky = {
        'topK': 23,
        'shrink': 26,
        'similarity' : 'tversky',
        'fw': 'TF-IDF',
        'a_alpha': 0.30904474725892556,
        'alpha' : 0.12835746708802967,
        'beta' : 1.995921498038378
    }

    max_map = 0
    data = get_data()

    test = True

    args = best_asymmetric

    itemCF = ItemBasedCollaborativeFiltering(args['topK'],
                                             args['shrink'],
                                             feature_weighting=args['fw'],
                                             similarity=args['similarity'],
                                             tversky_alpha=args['alpha'],
                                             tversky_beta=args['beta'],
                                             asymmetric_alpha=args['a_alpha'])

    if test:

        itemCF.fit(data['train'])
        result = itemCF.evaluate_MAP_target(data['test'], data['target_users'])

    else:
        URM_final = get_data()['URM_all'].tocsr()
        itemCF.fit(URM_final)
        write_output(itemCF, target_user_list=data['target_users'])
################################################ Test ##################################################