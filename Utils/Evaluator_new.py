from Algorithms.Notebooks_utils.evaluation_function import evaluate_MAP
import numpy as np

class Evaluator(object):
    def __init__(self, recommender_array):
        self.recommenders = recommender_array
        self.MAP_results = np.array([])

    def evaluate_recommenders(self, URM_test):
        for recommender in self.recommenders:
            self.MAP_results = np.append(self.MAP_results, evaluate_MAP(URM_test, recommender))

    def get_best_recommender(self):
        index = np.argmax(self.MAP_results)
        return self.recommenders[index]