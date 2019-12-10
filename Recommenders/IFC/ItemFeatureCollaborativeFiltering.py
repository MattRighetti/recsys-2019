from Utils.Toolkit import get_data, normalize_matrix, get_URM_BM_25, get_URM_TFIDF, generate_SM_user_feature_matrix
from Algorithms.Notebooks_utils.evaluation_function import evaluate_MAP, evaluate_MAP_target_users
from Algorithms.Base.Similarity.Cython.Compute_Similarity_Cython import Compute_Similarity_Cython
from Recommenders.IFC.Cython.BoostSimilarityMatrix import Booster
from Utils.Toolkit import BoosterObject
from Algorithms.Base.Recommender_utils import check_matrix
from Recommenders.BaseRecommender import BaseRecommender
from Utils.OutputWriter import write_output
import scipy.sparse as sps
from tqdm import tqdm
import numpy as np


class ItemFeatureCollaborativeFiltering(BaseRecommender):

    RECOMMENDER_NAME = "ItemFeatureCollaborativeFiltering"

    def __init__(self, topK, shrink):
        super().__init__()
        self.topK = topK
        self.shrink = shrink
        self.last_ten_boost = False

        self.URM_train = None
        self.ICM = None
        self.SM_item = None
        self.SM_user_feature = None
        self.RM_item = None

    def get_similarity_matrix(self, similarity='tanimoto'):
        similarity_object = Compute_Similarity_Cython(self.URM_train,
                                                      self.shrink,
                                                      self.topK,
                                                      normalize=True,
                                                      tversky_alpha=1.0,
                                                      tversky_beta=1.0,
                                                      similarity=similarity)
        return similarity_object.compute_similarity().tocsr()

    def fit(self, URM_train, ICM, boost=True):
        """
        PASS URM_TRAIN and ICM as CSR MATRICES
        :param URM_train:
        :param ICM:
        :return:
        """
        self.boost = boost

        self.URM_train = URM_train.copy()
        self.ICM = ICM.copy()

        self.SM_item = self.get_similarity_matrix()
        self.RM_item = self.URM_train.dot(self.SM_item).tocsr()

        self.SM_user_feature = generate_SM_user_feature_matrix(self.URM_train, self.ICM)
        self.ICM = get_URM_TFIDF(self.ICM)
        self.ICM = normalize_matrix(self.ICM)
        self.SM_user_feature = normalize_matrix(self.SM_user_feature)
        self.RM_item = Booster(self.RM_item, self.ICM, self.SM_user_feature, 0).apply_boost(self.RM_item, self.ICM, self.SM_user_feature)
        print(type(self.RM_item))
        print("Done")

    def get_expected_ratings_pre_boost(self, user_id):
        """
        Returns the ratings of the corresponding user
        :param user_id: ID of the User
        :return: 1D array of items containing each item rating
        """
        expected_ratings = self.RM_item[user_id].toarray().ravel()
        return np.squeeze(np.asarray(expected_ratings))

    def get_expected_ratings(self, user_id):
        """
        CONVENIENCE METHOD FOR HYBRID
        :param user_id:
        :param at:
        :return:
        """
        expected_ratings = self.get_expected_ratings_pre_boost(user_id)
        return expected_ratings


    def recommend(self, user_id, at=10, exclude_seen=True):
        expected_ratings = self.get_expected_ratings(user_id)
        # Index items con rating più alto
        recommended_items = np.flip(np.argsort(expected_ratings), 0)
        # Corrispondenti rating delle items sopra

        # remove seen of above
        if exclude_seen:
            unseen_items_mask = np.in1d(recommended_items, self.URM_train[user_id].indices, assume_unique=True,
                                        invert=True)
            recommended_items = recommended_items[unseen_items_mask][:at]

        return recommended_items


################################ TEST #######################################
if __name__ == '__main__':
    data = get_data()

    args = {
        'topK' : 29,
        'shrink' : 9
    }

    itemFeatureCF = ItemFeatureCollaborativeFiltering(args['topK'], args['shrink'])
    itemFeatureCF.fit(data['train'].tocsr(), data['ICM_subclass'].tocsr(), boost=True)
    itemFeatureCF.evaluate_MAP_target(data['test'].tocsr(), data['target_users'])
################################ TEST #######################################