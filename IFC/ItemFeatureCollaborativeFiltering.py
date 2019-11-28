from Algorithms.Base.Similarity.Cython.Compute_Similarity_Cython import Compute_Similarity_Cython
from IFC.Cython.BoostSimilarityMatrix import BoostSimilarityMatrix
import scipy.sparse as sps
import numpy as np

from Algorithms.Base.Recommender_utils import check_matrix
from Algorithms.Notebooks_utils.evaluation_function import evaluate_MAP, evaluate_MAP_target_users
from Utils.Toolkit import get_data


class ItemFeatureCollaborativeFiltering(object):
    def __init__(self, topK, shrink, feature_boost=True):
        self.topK = topK
        self.shrink = shrink
        self.feature_boost = feature_boost

        self.URM_train = None
        self.ICM = None
        self.SM_item = None
        self.SM_user_feature = None
        self.RM_item = None
        self.RM_boosted = None

    def get_similarity_matrix(self, similarity='cosine'):
        similarity_object = Compute_Similarity_Cython(self.URM_train,
                                                      self.shrink,
                                                      self.topK,
                                                      normalize=True,
                                                      tversky_alpha=1.0,
                                                      tversky_beta=1.0,
                                                      similarity='cosine')
        return similarity_object.compute_similarity().tocsr()


    def fit(self, URM_train, ICM):
        """
        PASS URM_TRAIN and ICM as CSR MATRICES
        :param URM_train:
        :param ICM:
        :return:
        """
        self.URM_train = URM_train
        self.ICM = ICM
        self.SM_item = self.get_similarity_matrix(URM_train)
        self.RM_item = self.URM_train.dot(self.SM_item).tocsr()
        self.SM_user_feature = sps.csr_matrix(self.URM_train.dot(self.ICM))
        self.SM_item = check_matrix(self.SM_item, format='csr')
        self.RM_item = check_matrix(self.RM_item, format='csr')
        self.SM_user_feature = check_matrix(self.SM_user_feature, format='csr')
        self.RM_boosted = self.get_boosted_RM(self.RM_item, self.ICM, self.SM_user_feature)
        self.RM_boosted = check_matrix(self.RM_boosted, format='csr')

    def get_expected_ratings(self, user_id):
        """
        Returns the ratings of the corresponding user
        :param user_id: ID of the User
        :return: 1D array of items containing each item rating
        """
        expected_ratings = self.RM_boosted[user_id].toarray().ravel()
        return np.squeeze(np.asarray(expected_ratings))

    def get_boost_ratings(self, user_id):
        expected_ratings = self.get_expected_ratings(user_id)
        return expected_ratings

    def get_boosted_RM(self, recommended_matrix, icm_matrix, user_feature_matrix):
        booster = BoostSimilarityMatrix()
        return booster.compute_boosted_matrix(recommended_matrix, icm_matrix, user_feature_matrix).tocsr()

    def recommend(self, user_id, at=10, exclude_seen=True):
        if self.feature_boost:
            expected_ratings = self.get_boost_ratings(user_id)
        else:
            expected_ratings = self.get_expected_ratings(user_id)

        recommended_items = np.flip(np.argsort(expected_ratings), 0)

        if exclude_seen:
            unseen_items_mask = np.in1d(recommended_items, self.URM_train[user_id].indices, assume_unique=True, invert=True)
            recommended_items = recommended_items[unseen_items_mask]

        return recommended_items[:at]

    def evaluate_MAP(self, URM_test):
        result = evaluate_MAP(URM_test, self)
        print("ItemCF -> MAP: {:.4f} with TopK = {} "
              "& Shrink = {}\t".format(result, self.topK, self.shrink))
        return result

    def evaluate_MAP_target(self, URM_test, target_user_list):
        result = evaluate_MAP_target_users(URM_test, self, target_user_list)
        print("ItemCF -> MAP: {:.4f} with TopK = {} "
              "& Shrink = {}\t".format(result, self.topK, self.shrink))
        return result

################################ TEST #######################################

data = get_data(dir_path='../')

args = {
    'topK' : 29,
    'shrink' : 5
}

itemFeatureCF = ItemFeatureCollaborativeFiltering(args['topK'], args['shrink'], feature_boost=True)
itemFeatureCF.fit(data['train'].tocsr(), data['ICM_subclass'].tocsr())
itemFeatureCF.evaluate_MAP_target(data['test'].tocsr(), data['target_users'])

################################ TEST #######################################