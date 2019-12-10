from Utils.Toolkit import generate_SM_user_feature_matrix, get_data
from Recommenders.BaseRecommender import BaseRecommender
import numpy as np
import scipy.sparse as sps

class TopPopFeatureUser(BaseRecommender):
    def __init__(self):
        super().__init__()
        self.SM_user_feature = None
        self.RM_user_feature = None
        self.RM_user_feature_ratings = None

    def fit(self, URM, ICM, at=4):
        self.SM_user_feature = generate_SM_user_feature_matrix(URM, ICM)
        self.get_features_ratings(at)

    def get_RM_scores(self):
        return self.RM_user_feature_ratings

    def get_RM_features(self):
        return self.RM_user_feature

    def get_features_ratings(self, at=10):
        data = []
        scores = []

        for i in range(self.SM_user_feature.shape[0]):
            recommended_features = np.zeros(self.SM_user_feature.shape[1])
            recommended_features_indexes = np.flip(np.argsort(self.SM_user_feature[i].toarray().ravel()), 0)
            ordered_features = self.SM_user_feature[i].toarray().ravel()[recommended_features_indexes]
            ordered_features_mul = np.where(ordered_features > 0, 1, ordered_features)
            recommended_features = recommended_features + recommended_features_indexes
            recommended_features *= ordered_features_mul

            data.append(recommended_features)
            scores.append(ordered_features)

        data_csr = sps.csr_matrix(data, dtype=int)
        scores_csr = sps.csr_matrix(scores, dtype=int)
        self.RM_user_feature = data_csr[:, :at]
        self.RM_user_feature_ratings = scores_csr[:, :at]

    def recommend(self, user_id, at=10):
        top_features = self.RM_user_feature[user_id].data
        top_ratings = self.RM_user_feature_ratings[user_id].data
        print(top_features)
        print(top_ratings)

################################ TEST #######################################
if __name__ == '__main__':
    data = get_data()
    URM = data['train'].tocsr()
    ICM = data['ICM_subclass'].tocsr()

    tf = TopPopFeatureUser()
    tf.fit(URM, ICM)
    tf.get_features_ratings()

    for i in range(10):
        tf.recommend(i)
################################ TEST #######################################