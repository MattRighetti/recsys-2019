from Algorithms.Base.Similarity.Cython.Compute_Similarity_Cython import Compute_Similarity_Cython
from Recommenders.IFC.Cython.BoostSimilarityMatrix import Booster
from Algorithms.Base.Recommender_utils import check_matrix
from Algorithms.Notebooks_utils.evaluation_function import evaluate_MAP, evaluate_MAP_target_users
from Utils.Toolkit import get_data, normalize_matrix, get_URM_BM_25, get_URM_TFIDF
from Recommenders.BaseRecommender import BaseRecommender
import numpy as np


class ItemFeatureCollaborativeFiltering(BaseRecommender):

    def __init__(self, topK, shrink):
        super().__init__()
        self.topK = topK
        self.shrink = shrink
        self.feature_boost = False
        self.last_ten_boost = False

        self.booster = None

        self.URM_train = None
        self.ICM = None
        self.SM_item = None
        self.SM_user_feature = None
        self.RM_item = None

    def get_similarity_matrix(self, similarity='cosine'):
        similarity_object = Compute_Similarity_Cython(self.URM_train,
                                                      self.shrink,
                                                      self.topK,
                                                      normalize=True,
                                                      tversky_alpha=1.0,
                                                      tversky_beta=1.0,
                                                      similarity='tanimoto')
        return similarity_object.compute_similarity().tocsr()

    def fit(self, URM_train, ICM, feature_boost=True, last_ten_boost=True):
        """
        PASS URM_TRAIN and ICM as CSR MATRICES
        :param URM_train:
        :param ICM:
        :return:
        """
        self.feature_boost = feature_boost
        self.last_ten_boost = last_ten_boost
        self.booster = Booster()

        self.URM_train = URM_train.copy()
        self.ICM = ICM.copy()
        #self.ICM = get_URM_BM_25(self.ICM)
        self.SM_item = self.get_similarity_matrix(URM_train)
        self.RM_item = self.URM_train.dot(self.SM_item).tocsr()
        self.SM_user_feature = self.URM_train.dot(self.ICM).tocsr()
        self.ICM = get_URM_TFIDF(self.ICM)
        self.ICM = normalize_matrix(self.ICM)

        self.SM_item = check_matrix(self.SM_item, format='csr')
        self.RM_item = check_matrix(self.RM_item, format='csr')
        self.SM_user_feature = check_matrix(self.SM_user_feature, format='csr')

    def get_expected_ratings(self, user_id):
        """
        Returns the ratings of the corresponding user
        :param user_id: ID of the User
        :return: 1D array of items containing each item rating
        """
        expected_ratings = self.RM_item[user_id].toarray().ravel()
        return np.squeeze(np.asarray(expected_ratings))

    def apply_boost(self, expected_ratings, user_id):
        startpos = self.RM_item.indptr[user_id]
        endpos = self.RM_item.indptr[user_id + 1]
        indices = self.RM_item.indices[startpos:endpos]
        boosted_ratings = self.booster.get_boosted_recommendations(expected_ratings,
                                                                   indices,
                                                                   user_id,
                                                                   0.1,
                                                                   self.ICM,
                                                                   self.SM_user_feature)
        return boosted_ratings

    def get_boosted_ratings(self, user_id):
        expected_ratings = self.get_expected_ratings(user_id)
        boosted_ratings = self.apply_boost(expected_ratings, user_id)
        return boosted_ratings

    def get_last_ten_boost(self, recommended_items, recommended_items_ratings, user_id):
        boosted_ratings, evaluated_correctly = self.booster.boost(recommended_items,
                                             recommended_items_ratings,
                                             user_id,
                                             self.ICM,
                                             self.SM_user_feature)
        #print(f'Before:{recommended_items}, After: {boosted_ratings}')
        return boosted_ratings, evaluated_correctly

    def recommend(self, user_id, at=10, exclude_seen=True):
        if self.feature_boost:
            expected_ratings = self.get_boosted_ratings(user_id)
        else:
            expected_ratings = self.get_expected_ratings(user_id)

        # Index items con rating più alto
        recommended_items = np.flip(np.argsort(expected_ratings), 0)
        # Corrispondenti rating delle items sopra
        recommended_items_ratings = expected_ratings[recommended_items]

        # TODO remove seen of above
        if exclude_seen:
            unseen_items_mask = np.in1d(recommended_items, self.URM_train[user_id].indices, assume_unique=True, invert=True)
            recommended_items = recommended_items[unseen_items_mask]
            recommended_items_ratings = recommended_items_ratings[unseen_items_mask]
            print(f'Before {recommended_items[:10]} + {recommended_items_ratings[:10]}', end=' | ')

        if self.last_ten_boost:
            recommended_items_boost, evaluated_correctly = self.get_last_ten_boost(recommended_items[:at], recommended_items_ratings[:at], user_id)
            if evaluated_correctly:
                # TODO reorder
                order_mask = np.flip(np.argsort(recommended_items_boost), 0)
                recommended_items = recommended_items[order_mask]

            print(f'After {recommended_items[:10]}  {recommended_items_boost[:10]}', end='\n\n\n')
            return recommended_items[:at]
        else:
            recommended_items = np.flip(np.argsort(expected_ratings), 0)
            return recommended_items[:at]

################################ TEST #######################################

data = get_data(dir_path='../../')

args = {
    'topK' : 29,
    'shrink' : 5
}

itemFeatureCF = ItemFeatureCollaborativeFiltering(args['topK'], args['shrink'])
#itemFeatureCF.fit(data['train'].tocsr(), data['ICM'].tocsr(), feature_boost=False)
#itemFeatureCF.evaluate_MAP_target(data['test'].tocsr(), data['target_users'])
itemFeatureCF.fit(data['train'].tocsr(), data['ICM_subclass'].tocsr(), feature_boost=False, last_ten_boost=True)
itemFeatureCF.evaluate_MAP_target(data['test'].tocsr(), data['target_users'])

################################ TEST #######################################