from tqdm import tqdm
import numpy as np


def MAP(recommended_items, relevant_items):
    is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

    # Cumulative sum: precision at 1, at 2, at 3 ...
    p_at_k = is_relevant * np.cumsum(is_relevant, dtype=np.float32) / (1 + np.arange(len(is_relevant)))
    map_score = np.sum(p_at_k) / np.min([len(relevant_items), len(is_relevant)])
    return map_score


class Evaluator(object):
    def __init__(self):
        self.URM_train = None
        self.URM_test = None
        self.dict_test = None
        self.target_users = None
        self.at = 10

    def global_evaluate_single(self, recommender):
        MAP_final = 0
        recommender.fit(self.URM_train)
        for user_id in tqdm(self.target_users):
            recommended_items = recommender.recommend(user_id)
            MAP_final += self.evaluate(user_id, recommended_items)
        MAP_final /= len(self.target_users)
        return MAP_final

    def evaluate(self, user_id, recommended_items):
        return MAP(recommended_items, relevant_items)