import numpy as np
import scipy.sparse as sps
from random import seed
from random import randint

class DataSplitter(object):
    def __init__(self, URM_all):
        self.URM_all = URM_all
        self.URM_train = None
        self.URM_test = None
        self.target_users = None
        self.dict_test = None
        self.train_test_split = 0.8
        self.at = 10
        self.indptr_array = self.URM_all.indptr

    def split_data(self, algorithm="leave_one_out"):
        if algorithm == "leave_one_out":
            self.leave_one_out()
        elif algorithm == "leave_k_out":
            self.leave_k_out(k=3)
        elif algorithm == "k_fold":
            self.k_fold(k=2)

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

        print(f'data in train: {len(self.URM_train.data)}, in test: {len(self.URM_test.data)}')

        return self.URM_train, self.URM_test


    def leave_one_out(self):

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

    def leave_k_out(self, k):

        train_mask = np.array([])

        num_us_with_k = 0
        num_us_not_k = 0

        for row in range(len(self.indptr_array) - 1):
            values_in_row = self.indptr_array[row + 1] - self.indptr_array[row]

            if values_in_row <= k:
                probability_false = values_in_row/k
                rand_arr = np.random.choice([True, False], values_in_row, p=[(1 - probability_false),probability_false])
                np.random.shuffle(rand_arr)
                train_mask = np.append(train_mask, rand_arr)
                num_us_not_k += 1
            elif values_in_row != 0:
                # Now values_in_row-1 must be True, 1 must be False
                # Remove last interaction
                sub_arr = np.array([True] * (values_in_row - k) + [False] * k)
                np.random.shuffle(sub_arr)
                train_mask = np.append(train_mask, sub_arr)
                num_us_with_k += 1

        print(f'\rUsers with more than {k} interactions: {num_us_with_k}, not {num_us_not_k}\r')

        return self.apply_mask(train_mask)

    def force_leave_k_out(self, k):
        """
        Leave K Out that removed k elements from row in dataset, if a row has less values
        than K then everything is removed from train mask and put in test matrix
        :param k:
        :return:
        """
        train_mask = np.array([])

        num_us_with_k = 0
        num_us_not_k = 0

        for row in range(len(self.indptr_array) - 1):
            values_in_row = self.indptr_array[row + 1] - self.indptr_array[row]

            if values_in_row <= k:
                # REMOVE ALL FROM TRAIN
                rand_arr = np.array([False] * values_in_row)
                train_mask = np.append(train_mask, rand_arr)
                num_us_not_k += 1
            elif values_in_row != 0:
                # Now values_in_row-1 must be True, 1 must be False
                # Remove last interaction
                sub_arr = np.array([True] * (values_in_row - k) + [False] * k)
                np.random.shuffle(sub_arr)
                train_mask = np.append(train_mask, sub_arr)
                num_us_with_k += 1

        print(f'\rUsers with more than {k} interactions: {num_us_with_k}, not {num_us_not_k}\r')

        return self.apply_mask(train_mask)

    def k_fold(self, k):

        train_mask = np.array([])

        for row in range(len(self.indptr_array) - 1):
            values_in_row = self.indptr_array[row + 1] - self.indptr_array[row]

            if values_in_row <= k:
                train_mask = np.append(train_mask, [True]*values_in_row)
            elif values_in_row != 0:
                # Now values_in_row-1 must be True, 1 must be False
                # Remove last interaction
                sub_arr = np.array([True] * (values_in_row - k) + [False] * k)
                np.random.shuffle(sub_arr)
                train_mask = np.append(train_mask, sub_arr)

        return self.apply_mask(train_mask)