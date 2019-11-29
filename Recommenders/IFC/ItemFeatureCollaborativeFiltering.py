from Algorithms.Base.Similarity.Cython.Compute_Similarity_Cython import Compute_Similarity_Cython
from Recommenders.IFC.Cython.BoostSimilarityMatrix import Booster
from Algorithms.Base.Recommender_utils import check_matrix
from Algorithms.Notebooks_utils.evaluation_function import evaluate_MAP, evaluate_MAP_target_users
from Utils.Toolkit import get_data, normalize_matrix, get_URM_BM_25, get_URM_TFIDF
from Recommenders.BaseRecommender import BaseRecommender
import numpy as np
import scipy.sparse as sps
from tqdm import tqdm


class ItemFeatureCollaborativeFiltering(BaseRecommender):

    RECOMMENDER_NAME = "ItemFeatureCollaborativeFiltering"

    def __init__(self, topK, shrink):
        super().__init__()
        self.topK = topK
        self.shrink = shrink
        self.last_ten_boost = False

        self.booster = None

        self.URM_train = None
        self.ICM = None
        self.SM_item = None
        self.SM_user_feature = None
        self.RM_item = None

    def generate_SM_user_feature_matrix(self):

        # Matrix will be a USER x ITEMFEATURES
        SM_user_feature_matrix = np.zeros((self.URM_train.shape[0], self.ICM.shape[1]), dtype=int)

        for user_id in tqdm(range(self.URM_train.shape[0]), desc="Evaluating SM_user_feature_matrix"):
            u_start_pos = self.URM_train.indptr[user_id]
            u_end_pos = self.URM_train.indptr[user_id + 1]

            mask = self.URM_train.indices[u_start_pos:u_end_pos]

            if len(mask) > 0:
                features_matrix = self.ICM[mask,:].sum(axis=0)
                user_features = np.squeeze(np.asarray(features_matrix))
            else:
                user_features = np.zeros(self.ICM.shape[1], dtype=int)

            SM_user_feature_matrix[user_id] = user_features

        SM_user_feature_matrix = sps.csr_matrix(SM_user_feature_matrix)
        print(f'Generated UFM with shape {SM_user_feature_matrix.shape}')
        return SM_user_feature_matrix

    def get_similarity_matrix(self, similarity='cosine'):
        similarity_object = Compute_Similarity_Cython(self.URM_train,
                                                      self.shrink,
                                                      self.topK,
                                                      normalize=True,
                                                      tversky_alpha=1.0,
                                                      tversky_beta=1.0,
                                                      similarity=similarity)
        return similarity_object.compute_similarity().tocsr()

    def fit(self, URM_train, ICM, last_ten_boost=True):
        """
        PASS URM_TRAIN and ICM as CSR MATRICES
        :param URM_train:
        :param ICM:
        :return:
        """
        self.last_ten_boost = last_ten_boost
        self.booster = Booster()

        self.URM_train = URM_train.copy()
        self.ICM = ICM.copy()
        self.SM_item = self.get_similarity_matrix()
        self.RM_item = self.URM_train.dot(self.SM_item).tocsr()
        self.SM_user_feature = self.generate_SM_user_feature_matrix().tocsr()
        #self.ICM = get_URM_BM_25(self.ICM)
        #self.ICM = get_URM_TFIDF(self.ICM)
        #self.ICM = normalize_matrix(self.ICM)

        self.SM_item = check_matrix(self.SM_item, format='csr')
        self.RM_item = check_matrix(self.RM_item, format='csr')
        self.SM_user_feature = check_matrix(self.SM_user_feature, format='csr')
        self.ICM = check_matrix(self.ICM, format='csr')

    def get_expected_ratings(self, user_id):
        """
        Returns the ratings of the corresponding user
        :param user_id: ID of the User
        :return: 1D array of items containing each item rating
        """
        expected_ratings = self.RM_item[user_id].toarray().ravel()
        return np.squeeze(np.asarray(expected_ratings))

    def get_last_ten_boost(self, recommended_items, recommended_items_ratings, user_id):
        boosted_ratings, evaluated_correctly = self.booster.boost(recommended_items,
                                             recommended_items_ratings,
                                             user_id,
                                             5,
                                             self.ICM,
                                             self.SM_user_feature)
        #print(f'Before:{recommended_items}, After: {boosted_ratings}')
        return boosted_ratings, evaluated_correctly

    def recommend(self, user_id, at=10, exclude_seen=True):
        expected_ratings = self.get_expected_ratings(user_id)

        # Index items con rating più alto
        recommended_items = np.flip(np.argsort(expected_ratings), 0)
        # Corrispondenti rating delle items sopra
        recommended_items_ratings = expected_ratings[recommended_items]

        # TODO remove seen of above
        if exclude_seen:
            unseen_items_mask = np.in1d(recommended_items, self.URM_train[user_id].indices, assume_unique=True, invert=True)
            recommended_items = recommended_items[unseen_items_mask][:10]
            recommended_items_ratings = recommended_items_ratings[unseen_items_mask][:10]
            #print(f'Before {recommended_items[:10]}')

        if self.last_ten_boost:
            recommended_items_boost, evaluated_correctly = self.get_last_ten_boost(recommended_items, recommended_items_ratings, user_id)
            if evaluated_correctly:
                # TODO reorder
                order_mask = np.flip(np.argsort(recommended_items_boost), 0)
                recommended_items = recommended_items[order_mask]

            #print(f'After {recommended_items}')
            return recommended_items
        else:
            return recommended_items

################################ TEST #######################################

data = get_data(dir_path='../../')

args = {
    'topK' : 31,
    'shrink' : 9
}

itemFeatureCF = ItemFeatureCollaborativeFiltering(args['topK'], args['shrink'])
itemFeatureCF.fit(data['train'].tocsr(), data['ICM_asset'].tocsr(), last_ten_boost=True)
itemFeatureCF.evaluate_MAP_target(data['test'].tocsr(), data['target_users'])

################################ TEST #######################################