from Utils.Evaluator_new import evaluate_MAP_target_users, evaluate_MAP
import numpy as np
from tqdm import tqdm

class BaseRecommender(object):

    RECOMMENDER_NAME = "BaseRecommender"

    def __init__(self):
        super(BaseRecommender, self).__init__()

    def fit(self, URM_train=None, **kwargs):
        pass

    def recommend(self, user_id, at=10, exclude_seen=True):
        pass

    def _filter_seen(self, user_id, scores):
        start_pos = self.URM_train.indptr[user_id]
        end_pos = self.URM_train.indptr[user_id + 1]
        user_profile = self.URM_train.indices[start_pos:end_pos]

        scores[user_profile] = -np.inf

        return scores

    def evaluate_MAP(self, URM_test):
        result = evaluate_MAP(URM_test, self)
        print("{} -> MAP: {:.4f} with TopK = {} "
              "& Shrink = {}\t".format(self.RECOMMENDER_NAME,
                                       result,
                                       self.topK,
                                       self.shrink))
        return result

    def evaluate_MAP_target(self, URM_test, target_user_list):
        result = evaluate_MAP_target_users(URM_test, self, target_user_list)
        if False:

            print("{} -> MAP: {:.4f} with TopK = {} "
              "& Shrink = {}\tTOTAL MISS={}\tRelevant={}".format(self.RECOMMENDER_NAME,
                                                                result['MAP'],
                                                                self.topK,
                                                                self.shrink,
                                                                 result['TOT_MISS'],
                                                                 result['RELEVANT']))
        else:
            print("{} -> MAP: {:.4f}\tTOTAL MISS={}\tRelevant={}".format(self.RECOMMENDER_NAME,
                                                                     result['MAP'],
                                                                     result['TOT_MISS'],
                                                                     result['RELEVANT']))

        return result