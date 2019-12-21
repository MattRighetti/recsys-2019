import numpy as np
from implicit.als import AlternatingLeastSquares
from tqdm import tqdm
from Recommenders.BaseRecommender import BaseRecommender
from Utils.Toolkit import get_data, get_URM_TFIDF, get_URM_BM_25

class AlternatingLeastSquare(BaseRecommender):
    """
    ALS implemented with implicit following guideline of
    https://medium.com/radon-dev/als-implicit-collaborative-filtering-5ed653ba39fe
    IDEA:
    Recomputing x_{u} and y_i can be done with Stochastic Gradient Descent, but this is a non-convex optimization problem.
    We can convert it into a set of quadratic problems, by keeping either x_u or y_i fixed while optimizing the other.
    In that case, we can iteratively solve x and y by alternating between them until the algorithm converges.
    This is Alternating Least Squares.
    """

    def __init__(self, n_factors=433, regularization=1.707545716729426e-05, iterations=29):
        super().__init__()
        self.n_factors = n_factors
        self.regularization = regularization
        self.iterations = iterations
        self.URM = None
        self.user_factors = None
        self.item_factors = None

    def fit(self, URM):
        self.URM = URM
        sparse_item_user = self.URM.T

        # Initialize the als model and fit it using the sparse item-user matrix
        model = AlternatingLeastSquares(factors=self.n_factors, regularization=self.regularization, iterations=self.iterations)

        alpha_val = 5

        # Calculate the confidence by multiplying it by our alpha value.
        data_conf = (sparse_item_user * alpha_val).astype('double')

        # Fit the model
        model.fit(data_conf)

        # Get the user and item vectors from our trained model
        self.user_factors = model.user_factors
        self.item_factors = model.item_factors


    def get_expected_ratings(self, user_id):
        scores = np.dot(self.user_factors[user_id], self.item_factors.T)
        return np.squeeze(scores)

    def recommend(self, user_id, at=10):
        expected_ratings = self.get_expected_ratings(user_id)

        recommended_items = np.flip(np.argsort(expected_ratings), 0)

        unseen_items_mask = np.in1d(recommended_items, self.URM[user_id].indices, assume_unique=True, invert=True)
        recommended_items = recommended_items[unseen_items_mask]

        return recommended_items[:at]

################################################ Test ##################################################
if __name__ == '__main__':
    data = get_data()

    ALS = AlternatingLeastSquare()
    ALS.fit(data['train'].tocsr())
    ALS.evaluate_MAP_target(data['test'].tocsr(), data['target_users'])
################################################ Test ##################################################