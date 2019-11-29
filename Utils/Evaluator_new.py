from Utils.Algorithm.Cython.Evaluate import MAP
from tqdm import tqdm
import numpy as np

def evaluate_MAP(URM_test, recommender_object, at=10, verbose=False):
    cumulative_MAP = 0.0
    URM_test = sps.csr_matrix(URM_test)
    n_users = URM_test.shape[0]

    for user_id in tqdm(n_users):

        start_pos = URM_test.indptr[user_id]
        end_pos = URM_test.indptr[user_id + 1]

        if end_pos - start_pos > 0:
            relevant_items = URM_test.indices[start_pos:end_pos]

            recommended_items = recommender_object.recommend(user_id, at)

            is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

            cumulative_MAP += MAP(is_relevant, relevant_items)

    cumulative_MAP /= n_users

    return cumulative_MAP

def evaluate_MAP_target_users(URM_test, recommender_object, target_users, at=10):
    cumulative_MAP = 0.0
    num_eval = 0
    n_users = URM_test.shape[0]

    total_miss_groups = np.zeros(10, dtype=int)

    n_total_miss = 0
    total_guessed = 0

    for user_id in tqdm(target_users, desc=f'Evaluating MAP target, total miss {n_total_miss}'):

        start_pos = URM_test.indptr[user_id]
        end_pos = URM_test.indptr[user_id + 1]

        if end_pos - start_pos > 0:
            relevant_items = URM_test.indices[start_pos:end_pos]

            recommended_items = recommender_object.recommend(user_id, at)
            num_eval += 1

            is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

            val = MAP(is_relevant, relevant_items)

            if val == 0:
                n_total_miss += 1
            elif val == 0.1:
                total_guessed += 1

            cumulative_MAP += val

    cumulative_MAP /= n_users

    results = {
        'MAP' : cumulative_MAP,
        'TOT_MISS' : n_total_miss,
        'TOT_GUESS' : total_guessed
    }

    return results

def MAP_Python(is_relevant, relevant_items):
    # is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

    # Cumulative sum: precision at 1, at 2, at 3 ...
    p_at_k = is_relevant * np.cumsum(is_relevant, dtype=np.float32) / (1 + np.arange(is_relevant.shape[0]))

    # Sum of average precision
    map_score = np.sum(p_at_k) / np.min([relevant_items.shape[0], is_relevant.shape[0]])

    return map_score