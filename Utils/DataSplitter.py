import numpy as np
import scipy.sparse as sps

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

    def apply_mask(self, train_mask):

        # URM_all to COO matrix
        URM_all_coo = self.URM_all.tocoo()
        # shape (num_of_user, num_of_items)
        shape = URM_all_coo.shape
        print(shape)

        train_mask = train_mask.astype(bool)
        test_mask = np.logical_not(train_mask)

        self.URM_train = sps.coo_matrix(
            (URM_all_coo.data[train_mask], (URM_all_coo.row[train_mask], URM_all_coo.col[train_mask])), shape=shape)
        self.URM_test = sps.coo_matrix(
            (URM_all_coo.data[test_mask], (URM_all_coo.row[test_mask], URM_all_coo.col[test_mask])), shape=shape)

        self.URM_train = self.URM_train.tocsr()
        self.URM_test = self.URM_test.tocsr()

        #print(f'data in train: {len(self.URM_train.data)}, in test: {len(self.URM_test.data)}')

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
        #print(f'\rUsers with more than {k} interactions: {num_us_with_k}, not {num_us_not_k}\r')

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

        #print(f'\rUsers with more than {k} interactions: {num_us_with_k}, not {num_us_not_k}\r')

        return self.apply_mask(train_mask)

    def split(self, Matrix_CSR, k):
        Matrix_CSR = sps.csr_matrix(Matrix_CSR)
        include_mask = np.array([])
        for row in range(len(Matrix_CSR.indptr) - 1):
            values_in_row = Matrix_CSR.indptr[row + 1] - Matrix_CSR.indptr[row]
            if values_in_row != 0:
                # Now values_in_row-1 must be True, 1 must be False
                # Remove last interaction
                sub_arr = np.random.choice([True, False], values_in_row, p=[1/k, (k-1)/k ])
                include_mask = np.append(include_mask, sub_arr)

        # URM_all to COO matrix
        Matrix_CSR = Matrix_CSR.tocoo()
        # shape (num_of_user, num_of_items)
        shape = Matrix_CSR.shape

        train_mask = include_mask.astype(bool)
        test_mask = np.logical_not(train_mask)

        first_matrix = sps.coo_matrix(
            (Matrix_CSR.data[train_mask], (Matrix_CSR.row[train_mask], Matrix_CSR.col[train_mask])),
                shape=shape)
        last_matrix = sps.coo_matrix(
            (Matrix_CSR.data[test_mask], (Matrix_CSR.row[test_mask], Matrix_CSR.col[test_mask])),
                shape=shape)

        first_matrix = first_matrix.tocsr()
        last_matrix = last_matrix.tocsr()

        # print(f'data in train: {len(self.URM_train.data)}, in test: {len(self.URM_test.data)}')

        return first_matrix, last_matrix

    def k_fold(self, k=3):
        URM_to_be_splitted = sps.csr_matrix(self.URM_all)
        matrices = []
        last_matrix = None
        while k > 1:
            first_matrix, last_matrix = self.split(URM_to_be_splitted, k)
            matrices.append(first_matrix)
            URM_to_be_splitted = last_matrix
            k -= 1
        matrices.append(last_matrix)

        return matrices