from Utils.Evaluator_new import evaluate_MAP_target_users, evaluate_MAP
from tqdm import tqdm

class BaseRecommender(object):
    def __init__(self):
        super(BaseRecommender, self).__init__()

    def fit(self, URM_train=None, **kwargs):
        pass

    def recommend(self, user_id, at=10, exclude_seen=True):
        pass

    def evaluate_MAP(self, URM_test):
        result = evaluate_MAP(URM_test, self)
        print("ItemCF -> MAP: {:.4f} with TopK = {} "
              "& Shrink = {}\t".format(result, self.topK, self.shrink))
        return result

    def evaluate_MAP_target(self, URM_test, target_user_list):
        result = evaluate_MAP_target_users(URM_test, self, target_user_list)
        print("ItemCF -> MAP: {:.4f} with TopK = {} "
              "& Shrink = {}\tTOTAL MISS={}\tGUESSED={}".format(result['MAP'], self.topK, self.shrink, result['TOT_MISS'], result['TOT_GUESS']))
        return result