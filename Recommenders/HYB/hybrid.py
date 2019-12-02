from Algorithms.Notebooks_utils.evaluation_function import evaluate_MAP, evaluate_MAP_target_users
from Algorithms.GraphBased.P3alphaRecommender import P3alphaRecommender
from Recommenders.CF.item_cf import ItemBasedCollaborativeFiltering
from Recommenders.CF.user_cf import UserBasedCollaborativeFiltering
from Recommenders.CBF.item_CBF import ItemContentBasedRecommender
from Recommenders.SLIM.SLIM_BPR_Cython import SLIM_BPR_Cython
from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.NonPersonalized.top_pop import TopPop
from Utils.Toolkit import get_data, feature_boost_URM


class HybridRecommender(BaseRecommender):

    RECOMMENDER_NAME = "HYB"

    def __init__(self, weights=None, userCF_args=None, itemCBF_args=None, itemCF_args=None, SLIM_BPR_args=None,
                 P3Graph_args=None, with_top_pop=False):
        super().__init__()
        self.URM_train = None
        self.with_top_pop = with_top_pop

        ######################## DEFAULT VALUES ########################
        self.weight = {
            'user_cf' : 0.5,
            'item_cf' : 0.5,
            'SLIM_BPR' : 0.5,
            'item_cbf' : 0.3
        }

        self.itemCF_args = {
            'topK' : 31,
            'shrink' : 27
        }

        self.userCF_args = {
            'topK' : 881,
            'shrink' : 904
        }

        self.itemCBF_args = {
            'topK' : 100,
            'shrink' : 10
        }

        self.SLIM_BPR_args = {
            'topK': 25,
            'lambda_i': 0.03,
            'lambda_j': 0.9,
            'epochs': 3800,
            'learning_rate' : 1e-3,
            'sgd_mode' : 'adagrad'
        }

        self.P3Graph_args = {
            'topK': 52,
            'alpha': 0.015859078386749607,
            'normalize_similarity': True
        }

        self.ICF_args = {
            'topK' : 29,
            'shrink' : 5
        }

        ######################## Weights ########################
        if weights is not None:
            self.weight = weights
        if userCF_args is not None:
            self.userCF_args = userCF_args
        if itemCBF_args is not None:
            self.itemCBF_args = itemCBF_args
        if itemCF_args is not None:
            self.itemCF_args = itemCF_args
        if SLIM_BPR_args is not None:
            self.SLIM_BPR_args = SLIM_BPR_args
        if P3Graph_args is not None:
            self.P3Graph_args = P3Graph_args

        ######################## Collaborative Filtering ########################
        self.userCF = UserBasedCollaborativeFiltering(topK=self.userCF_args['topK'], shrink=self.userCF_args['shrink'])
        self.itemCF = ItemBasedCollaborativeFiltering(topK=self.itemCF_args['topK'], shrink=self.itemCF_args['shrink'])
        self.itemCBF = ItemContentBasedRecommender(topK=self.itemCBF_args['topK'],
                                                   shrink=self.itemCBF_args['shrink'])
        self.SLIM_BPR = SLIM_BPR_Cython(epochs=self.SLIM_BPR_args['epochs'],
                                topK=self.SLIM_BPR_args['topK'],
                                lambda_i=self.SLIM_BPR_args['lambda_i'],
                                lambda_j=self.SLIM_BPR_args['lambda_j'],
                                positive_threshold=1,
                                sgd_mode=self.SLIM_BPR_args['sgd_mode'],
                                learning_rate=self.SLIM_BPR_args['learning_rate'],
                                batch_size=1000)
        self.P3Graph = None
        self.topPop = TopPop()

        self.userCF_scores = None
        self.itemCF_scores = None
        self.SLIM_BPR_scores = None
        self.itemCBF_scores = None

    def fit(self, URM_train, ICM, boost=True):
        self.URM_train = URM_train.copy()

        ########### PREPROCESSING #########
        if boost:
            self.URM_train = feature_boost_URM(URM_train, 5)
            #self.URM_train = get_URM_TFIDF(self.URM_train.transpose())
            #self.URM_train = self.URM_train.transpose().tocsr()
            #self.URM_train = normalize_matrix(self.URM_train, axis=1)

        ########### Models ###########
        self.P3Graph = P3alphaRecommender(URM_train.copy(), verbose=False)
        ########### FITTING ##########
        self.userCF.fit(self.URM_train.copy())
        self.itemCF.fit(self.URM_train.copy())
        self.topPop.fit(self.URM_train.copy())
        self.SLIM_BPR.fit(self.URM_train.copy())
        self.itemCBF.fit(self.URM_train.copy(), ICM)
        self.P3Graph.fit(self.P3Graph_args['topK'], self.P3Graph_args['alpha'], normalize_similarity=self.P3Graph_args['normalize_similarity'])

    def recommend(self, user_id, at=10, exclude_seen=True):
        self.userCF_scores = self.userCF.get_expected_recommendations(user_id)
        self.itemCF_scores = self.itemCF.get_expected_ratings(user_id)
        self.SLIM_BPR_scores = self.SLIM_BPR.get_expected_ratings(user_id)
        self.itemCBF_scores = self.itemCBF.get_expected_ratings(user_id)
        # TODO get_expected_recommendations for TopPop

        start_pos = self.URM_train.indptr[user_id]
        end_pos = self.URM_train.indptr[user_id + 1]

        if self.with_top_pop and len(self.URM_train.indices[start_pos:end_pos]) < 1:
            scores = self.topPop.recommend(user_id)
            ranking = scores
        else:
            scores = (self.userCF_scores * self.weight['user_cf']) + \
                    (self.itemCF_scores * self.weight['item_cf']) + \
                    (self.SLIM_BPR_scores * self.weight['SLIM_BPR']) + \
                    (self.itemCBF_scores * self.weight['item_cbf'])

        if exclude_seen:
            scores = self.filter_seen(user_id, scores)

        ranking = scores.argsort()[::-1]

        return ranking[:at]

    def filter_seen(self, user_id, scores):
        start_pos = self.URM_train.indptr[user_id]
        end_pos = self.URM_train.indptr[user_id + 1]
        user_profile = self.URM_train.indices[start_pos:end_pos]

        scores[user_profile] = -50

        return scores

################################################ Test ##################################################
max_map = 0
data = get_data(dir_path='../../')

userCF_args = {
    'topK' : 204,
    'shrink' : 850
}

itemCF_args = {
    'topK' : 29,
    'shrink' : 5
}

SLIM_BPR_args = {
    'topK': 20,
    'lambda_i': 5.0,
    'lambda_j': 7.0,
    'epochs': 5000,
    'learning_rate' : 1e-4,
    'sgd_mode' : 'adam'
}

weights = {
    'user_cf' : 1,
    'item_cf' : 2,
    'SLIM_BPR' : 5,
    'item_cbf' : 0.3
}

hyb = HybridRecommender(weights=weights, userCF_args=userCF_args, itemCF_args=itemCF_args, with_top_pop=True)
hyb.fit(data['train'].tocsr(), data['ICM_subclass'].tocsr())
result = hyb.evaluate_MAP_target(data['test'], data['target_users'])
print(weights)
#
#URM_final = data['train'] + data['test']
#URM_final = URM_final.tocsr()
#
#print(type(URM_final))
#hyb.fit(URM_final, data['ICM'])
#write_output(hyb, target_user_list=data['target_users'])
################################################ Test ##################################################