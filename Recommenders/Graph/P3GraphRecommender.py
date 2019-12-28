import numpy as np
import scipy.sparse as sps
import time
from sklearn.preprocessing import normalize
from Utils.Toolkit import get_data
from Recommenders.BaseRecommender import BaseRecommender
from Algorithms.Base.Recommender_utils import check_matrix, similarityMatrixTopK

class P3AlphaRecommender(BaseRecommender):

    RECOMMENDER_NAME = "P3alphaRecommender"

    def __init__(self, topK=100, alpha=1.0, min_rating=0, implicit=True, normalize_similarity=False):
        super().__init__()

        self.topK = topK
        self.alpha = alpha
        self.min_rating = min_rating
        self.implicit = implicit
        self.normalize_similarity = normalize_similarity

        self.SM = None
        self.RM = None

    def __str__(self):
        return "P3alpha(alpha={}, min_rating={}, topk={}, implicit={}, normalize_similarity={})".format(self.alpha,
                                                                                                        self.min_rating,
                                                                                                        self.topK,
                                                                                                        self.implicit,
                                                                                                        self.normalize_similarity)

    def fit(self, URM_train, verbose=True):

        self.URM_train = URM_train
        self.verbose = verbose


        #
        # if X.dtype != np.float32:
        #     print("P3ALPHA fit: For memory usage reasons, we suggest to use np.float32 as dtype for the dataset")

        if self.min_rating > 0:
            self.URM_train.data[self.URM_train.data < self.min_rating] = 0
            self.URM_train.eliminate_zeros()
            if self.implicit:
                self.URM_train.data = np.ones(self.URM_train.data.size, dtype=np.float64)

        # Pui is the row-normalized urm
        Pui = normalize(self.URM_train, norm='l1', axis=1)

        # Piu is the column-normalized, "boolean" urm transposed
        X_bool = self.URM_train.transpose(copy=True)
        X_bool.data = np.ones(X_bool.data.size, np.float64)
        # ATTENTION: axis is still 1 because i transposed before the normalization
        Piu = normalize(X_bool, norm='l1', axis=1)
        del X_bool

        # Alfa power
        if self.alpha != 1.:
            Pui = Pui.power(self.alpha)
            Piu = Piu.power(self.alpha)

        # Final matrix is computed as Pui * Piu * Pui
        # Multiplication unpacked for memory usage reasons
        block_dim = 200
        d_t = Piu

        # Use array as it reduces memory requirements compared to lists
        dataBlock = 10000000

        rows = np.zeros(dataBlock, dtype=np.int32)
        cols = np.zeros(dataBlock, dtype=np.int32)
        values = np.zeros(dataBlock, dtype=np.float64)

        numCells = 0

        start_time = time.time()
        start_time_printBatch = start_time

        for current_block_start_row in range(0, Pui.shape[1], block_dim):

            if current_block_start_row + block_dim > Pui.shape[1]:
                block_dim = Pui.shape[1] - current_block_start_row

            similarity_block = d_t[current_block_start_row:current_block_start_row + block_dim, :] * Pui
            similarity_block = similarity_block.toarray()

            for row_in_block in range(block_dim):
                row_data = similarity_block[row_in_block, :]
                row_data[current_block_start_row + row_in_block] = 0

                best = row_data.argsort()[::-1][:self.topK]

                notZerosMask = row_data[best] != 0.0

                values_to_add = row_data[best][notZerosMask]
                cols_to_add = best[notZerosMask]

                for index in range(len(values_to_add)):

                    if numCells == len(rows):
                        rows = np.concatenate((rows, np.zeros(dataBlock, dtype=np.int32)))
                        cols = np.concatenate((cols, np.zeros(dataBlock, dtype=np.int32)))
                        values = np.concatenate((values, np.zeros(dataBlock, dtype=np.float64)))

                    rows[numCells] = current_block_start_row + row_in_block
                    cols[numCells] = cols_to_add[index]
                    values[numCells] = values_to_add[index]

                    numCells += 1

            if time.time() - start_time_printBatch > 60:
                self._print("Processed {} ( {:.2f}% ) in {:.2f} minutes. Rows per second: {:.0f}".format(
                    current_block_start_row,
                    100.0 * float(current_block_start_row) / Pui.shape[1],
                    (time.time() - start_time) / 60,
                    float(current_block_start_row) / (time.time() - start_time)))

                sys.stdout.flush()
                sys.stderr.flush()

                start_time_printBatch = time.time()

        self.SM = sps.csr_matrix((values[:numCells], (rows[:numCells], cols[:numCells])),
                                 shape=(Pui.shape[1], Pui.shape[1]))

        if self.normalize_similarity:
            self.SM = normalize(self.SM, norm='l1', axis=1)

        if self.topK:
            self.SM = similarityMatrixTopK(self.SM, k=self.topK)

        self.SM = check_matrix(self.SM, format='csr')

        self.RM = self.URM_train.dot(self.SM)

    def recommend(self, user_id, at=10, exclude_seen=True):
        expected_ratings = self.get_expected_ratings(user_id)
        recommended_items = np.flip(np.argsort(expected_ratings), 0)

        if exclude_seen:
            unseen_items_mask = np.in1d(recommended_items, self.URM_train[user_id].indices, assume_unique=True, invert=True)
            recommended_items = recommended_items[unseen_items_mask]

        return recommended_items[:at]

    def get_expected_ratings(self, user_id):
        expected_recommendations = self.RM[user_id].todense()
        return np.squeeze(np.asarray(expected_recommendations))


################################################ Test ##################################################
if __name__ == '__main__':

    data = get_data()
    train = data['train']
    test = data['test']
    target_users = data['target_users']

    P3 = P3alphaRecommender(topK=66, alpha=0.2731573847973295, min_rating=0, implicit=False, normalize_similarity=True)
    P3.fit(train)
    P3.evaluate_MAP_target(test, target_users)
################################################ Test ##################################################
