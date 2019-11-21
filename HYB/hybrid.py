from Algorithms.Notebooks_utils.evaluation_function import evaluate_MAP, evaluate_MAP_target_users
import numpy as np

class HybridRecommender(object):
    def __init__(self, URM_train, userCF, itemCF):
        self.URM_train = URM_train

        self.userCF = userCF
        self.itemCF = itemCF

        self.userCF_w = None
        self.itemCF_w = None

        self.userCF_scores = None
        self.itemCF_scores = None

    def fit(self, userCF_w=1, itemCF_w=1):
        self.userCF_w = userCF_w
        self.itemCF_w = itemCF_w

    def recommend(self, user_id, at=10, exclude_seen=True):
        self.userCF_scores = self.userCF.get_scores(user_id)
        self.itemCF_scores = self.itemCF.get_scores(user_id)

        scores = (self.userCF_scores * self.userCF_w) + (self.itemCF_scores * self.itemCF_w)

        if exclude_seen:
            scores = self.filter_seen(user_id, scores)

        ranking = scores.argsort()[::-1]

        return ranking[:at]

    def filter_seen(self, user_id, scores):
        start_pos = self.URM_train.indptr[user_id]
        end_pos = self.URM_train.indptr[user_id + 1]
        user_profile = self.URM_train.indices[start_pos:end_pos]

        scores[user_profile] = -np.inf

        return scores

    def evaluate_MAP(self, URM_test):
        result = evaluate_MAP(URM_test, self)
        print("HYB -> MAP: {:.4f} with UserCF TopK = {} "
              "& UserCF Shrink = {}, ItemCF TopK = {} & ItemCF Shrink = {} \t".format(result, self.userCF.get_topK,
                                                                                      self.userCF.get_shrink(),
                                                                                      self.itemCF.get_topK,
                                                                                      self.itemCF.get_shrink))

    def evaluate_MAP_target(self, URM_test, target_user_list):
        result = evaluate_MAP_target_users(URM_test, self, target_user_list)
        print("HYB -> MAP: {:.4f} with UserCF TopK = {} "
              "& UserCF Shrink = {}, ItemCF TopK = {} & ItemCF Shrink = {} \t".format(result, self.userCF.get_topK,
                                                                                      self.userCF.get_shrink(),
                                                                                      self.itemCF.get_topK,
                                                                                      self.itemCF.get_shrink))