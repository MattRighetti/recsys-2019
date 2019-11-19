import numpy as np
import scipy.sparse as sps
from random import seed
from random import randint

class Evaluator(object):
    def __init__(self):
        self.URM_all = None
        self.URM_train = None
        self.URM_test = None
        self.target_users = None
        self.dict_test = None
        self.train_test_split = 0.8
        self.at = 10


    def apply_mask(self, train_mask):

        # URM_all to COO matrix
        URM_all_coo = self.URM_all.tocoo()
        # shape (num_of_user, num_of_items)
        shape = URM_all_coo.shape

        train_mask = train_mask.astype(bool)
        test_mask = np.logical_not(train_mask)

        self.URM_train = sps.coo_matrix(
            (URM_all_coo.data[train_mask], (URM_all_coo.row[train_mask], URM_all_coo.col[train_mask])), shape=shape)
        self.URM_test = sps.coo_matrix(
            (URM_all_coo.data[test_mask], (URM_all_coo.row[test_mask], URM_all_coo.col[test_mask])), shape=shape)

        self.URM_train = self.URM_train.tocsr()
        self.URM_test = self.URM_test.tocsr()

        return self.URM_train, self.URM_test


    def leave_one_out(self, URM_all):
        self.URM_all = URM_all

        indptr_array = self.URM_all.indptr

        train_mask = np.array([])

        for row in range(len(indptr_array) - 1):
            values_in_row = indptr_array[row + 1] - indptr_array[row]

            if values_in_row == 1:
                train_mask = np.append(train_mask, [True])
            elif values_in_row != 0:
                # Now values_in_row-1 must be True, 1 must be False
                # Remove last interaction
                sub_arr = np.array([True] * (values_in_row - 1) + [False])
                np.random.shuffle(sub_arr)
                train_mask = np.append(train_mask, sub_arr)


        return self.apply_mask(train_mask)

    def leave_k_out(self, URM_all, k):
        self.URM_all = URM_all

        indptr_array = self.URM_all.indptr

        train_mask = np.array([])

        for row in range(len(indptr_array) - 1):
            values_in_row = indptr_array[row + 1] - indptr_array[row]

            if values_in_row <= k:
                train_mask = np.append(train_mask, [True])
            elif values_in_row != 0:
                # Now values_in_row-1 must be True, 1 must be False
                # Remove last interaction
                sub_arr = np.array([True] * (values_in_row - k) + [False] * k)
                np.random.shuffle(sub_arr)
                train_mask = np.append(train_mask, sub_arr)

        return self.apply_mask(train_mask)