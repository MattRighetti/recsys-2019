from Algorithms.Base.IR_feature_weighting import okapi_BM_25
from Utils.DataSplitter import DataSplitter
from sklearn.preprocessing import normalize
from sklearn import feature_extraction
import scipy.sparse as sps
from tqdm import tqdm
from enum import Enum
import pandas as pd
import numpy as np



class DataReader(object):
    """
    This class will read the URM_train and the Target_users files and will generate every URM that we'll need
    """
    def __init__(self, dir_path='/Users/mattiarighetti/Developer/PycharmProjects/recsys/'):
        dir_file = dir_path
        self.data_train_file_path = dir_file + "data/data_train.csv"
        self.user_target_file_path = dir_file + "data/alg_sample_submission.csv"
        self.item_subclass_file_path = dir_file + "data/data_ICM_sub_class.csv"
        self.item_assets_file_path = dir_file + "data/data_ICM_asset.csv"
        self.item_price_file_path = dir_file + "data/data_ICM_price.csv"
        self.user_age_file_path = dir_file + "data/data_UCM_age.csv"
        self.user_region_file_path = dir_file + "data/data_UCM_region.csv"

    def target_users(self):
        target_df = pd.read_csv(self.user_target_file_path)
        return list(target_df['user_id'])

    def URM_COO(self):
        df = pd.read_csv(self.data_train_file_path)
        userList = list(df['row'])
        itemList = list(df['col'])
        ratingList = list(df['data'])
        return sps.coo_matrix((ratingList, (userList, itemList)), dtype=np.float64)

    def ICM_subclass_COO(self):
        df = pd.read_csv(self.item_subclass_file_path)
        icm_items_list = list(df['row'])
        subclass_list = list(df['col'])
        item_in_subclass_list = list(df['data'])
        return sps.coo_matrix((item_in_subclass_list, (icm_items_list, subclass_list)), dtype=np.float64)

    def ICM_asset_COO(self):
        df = pd.read_csv(self.item_assets_file_path)

        # transform floats in int labels
        label_list = get_label_list(df['data'])
        df['col'] = 1
        features = np.ones(len(df['col']))
        items = list(df['row'])
        return sps.coo_matrix((features, (items, label_list)), dtype=np.float64)

    def ICM_price_COO(self):
        df = pd.read_csv(self.item_price_file_path)

        # transform floats in int labels
        label_list = get_label_list(df['data'])
        df['col'] = 1
        features = np.ones(len(df['col']))
        items = list(df['row'])
        return sps.coo_matrix((features, (items, label_list)), dtype=np.float64)

    def UCM_region_COO(self):
        df = pd.read_csv(self.user_region_file_path)
        ucm_user_list = list(df['row'])
        region_list = list(df['col'])
        user_region_list = list(df['data'])
        return sps.coo_matrix((user_region_list, (ucm_user_list, region_list)), shape=(30911, 8), dtype=np.float64)

    def UCM_age_COO(self):
        df = pd.read_csv(self.user_age_file_path)
        ucm_user_list = list(df['row'])
        age_list = list(df['col'])
        user_age_list = list(df['data'])
        return sps.coo_matrix((user_age_list, (ucm_user_list, age_list)), shape=(30911, 11), dtype=np.float64)

    def ICM_total(self):
        icm_asset = self.ICM_asset_COO()
        #icm_price = self.ICM_price_COO()
        icm_subclass = self.ICM_price_COO()
        #icm_total = sps.hstack((icm_asset, icm_subclass, icm_price))
        icm_total = sps.hstack((icm_asset, icm_subclass))
        return icm_total

    def UCM_total(self):
        ucm_age = self.UCM_age_COO()
        ucm_region = self.UCM_region_COO()
        ucm_total = sps.hstack((ucm_age, ucm_region))
        return ucm_total

class TestSplit(Enum):
    LEAVE_ONE_OUT = 1
    LEAVE_K_OUT = 2
    FORCE_LEAVE_K_OUT = 3
    K_FOLD = 4

class TestGen(object):
    """
    This class generates URM_train & URM_test matrices
    """
    def __init__(self, URM_all_csr, ICM_csr, kind=TestSplit.FORCE_LEAVE_K_OUT, k=10):
        if kind is TestSplit.FORCE_LEAVE_K_OUT:
            self.URM_train, self.URM_test = DataSplitter(URM_all_csr, ICM_csr).force_leave_k_out(k)
        elif kind is TestSplit.LEAVE_K_OUT:
            self.URM_train, self.URM_test = DataSplitter(URM_all_csr, ICM_csr).leave_k_out(k)
        elif kind is TestSplit.LEAVE_ONE_OUT:
            self.URM_train, self.URM_test = DataSplitter(URM_all_csr, ICM_csr).leave_one_out()
            self.ICM_train, self.ICM_test = DataSplitter(URM_all_csr, ICM_csr).leave_one_out_ICM()
        elif kind is TestSplit.K_FOLD:
            self.Matrices = DataSplitter(URM_all_csr, ICM_csr).k_fold(k=k)

    def get_k_fold_matrices(self):
        return self.Matrices

#########################################################################################################
#                                               UTILITIES                                               #
#########################################################################################################

def get_label_list(float_array):
    set_unique_prices = set(float_array)
    list_unique_prices = np.asarray(list(set_unique_prices), dtype=np.float64)

    label_list = []

    for value in float_array:
        label = np.where(list_unique_prices == value)[0]
        label_list.extend(label)

    return label_list

def get_URM_BM_25(URM):
    return okapi_BM_25(URM)

def normalize_matrix(URM, axis=0):
    """
    Matrix normalization
    :param URM:
    :param axis:
    :return: Normalized matrix
    """
    n_matrix = normalize(URM, 'l2', axis)
    return n_matrix.tocsr()

def TF_IDF(dataMatrix):
    """
    Items are assumed to be on rows
    :param dataMatrix:
    :return:
    """

    assert np.all(np.isfinite(dataMatrix.data)), \
        "TF_IDF: Data matrix contains {} non finite values.".format(np.sum(np.logical_not(np.isfinite(dataMatrix.data))))

    assert np.all(dataMatrix.data >= 0.0),\
        "TF_IDF: Data matrix contains {} negative values, computing the square root is not possible.".format(np.sum(dataMatrix.data < 0.0))

    # TFIDF each row of a sparse amtrix
    dataMatrix = sps.coo_matrix(dataMatrix)
    N = float(dataMatrix.shape[0])

    # calculate IDF
    idf = np.log(N / (1 + np.bincount(dataMatrix.col)))

    # apply TF-IDF adjustment
    dataMatrix.data = np.sqrt(dataMatrix.data) * idf[dataMatrix.col]

    return dataMatrix.tocsr()

def get_URM_TFIDF(URM):
    """
    Applies TF-IDF weighting to the passed matrix
    :param URM:
    :return:
    """
    URM_tfidf = feature_extraction.text.TfidfTransformer().fit_transform(URM)
    return URM_tfidf.tocsr()

def get_target_users_group(target_users, URM):
    target_users = target_users

    group_cold = []
    group_one = []
    group_two = []
    group_three = []

    print("Grouping users...")

    for user in target_users:
        s_pos = URM.indptr[user]
        e_pos = URM.indptr[user+1]

        if len(URM.indices[s_pos:e_pos]) == 0:
            group_cold.append(user)

        elif 0 < len(URM.indices[s_pos:e_pos]) <= 2:
            group_one.append(user)

        elif 2 < len(URM.indices[s_pos:e_pos]) <= 5:
            group_two.append(user)

        elif 5 < len(URM.indices[s_pos:e_pos]):
            group_three.append(user)

    print(f'Group Cold contains {len(group_cold)} users')
    print(f'Group One contains {len(group_one)} users')
    print(f'Group Two contains {len(group_two)} users')
    print(f'Group Three contains {len(group_three)} users')

    return np.array(group_cold), np.array(group_one), np.array(group_two), np.array(group_three)


def get_data(split_kind=None, test_train_index='1'):
    dataReader = DataReader()
    UCM_region = dataReader.UCM_region_COO()
    UCM_age = dataReader.UCM_age_COO()
    URM_all = dataReader.URM_COO()
    ICM_price = dataReader.ICM_price_COO()
    ICM_asset = dataReader.ICM_price_COO()
    ICM_subclass = dataReader.ICM_subclass_COO()
    ICM = dataReader.ICM_total()
    UCM = dataReader.UCM_total()
    target_users = dataReader.target_users()
    static_train = sps.load_npz(f'/Users/mattiarighetti/Developer/PycharmProjects/recsys/data/saved_test_train/train/train_{test_train_index}.npz')
    static_test = sps.load_npz(f'/Users/mattiarighetti/Developer/PycharmProjects/recsys/data/saved_test_train/test/test_{test_train_index}.npz')

    if split_kind is None:
        testGen = TestGen(URM_all.tocsr(), ICM_subclass.tocsr(), TestSplit.LEAVE_ONE_OUT)
    else:
        testGen = TestGen(URM_all.tocsr(), ICM_subclass.tocsr(), split_kind)

    data = {
        'URM_all': URM_all,
        'train': testGen.URM_train,
        'test': testGen.URM_test,
        'ICM_test' : testGen.ICM_test,
        'ICM_train' : testGen.ICM_train,
        'target_users': target_users,
        'UCM_region': UCM_region,
        'UCM_age': UCM_age,
        'ICM_price': ICM_price,
        'ICM_asset': ICM_asset,
        'ICM_subclass': ICM_subclass,
        'ICM' : ICM,
        'UCM' : UCM,
        's_train': static_train,
        's_test':static_test
    }

    return data

def get_static_data(index=1):
    static_train = sps.load_npz(f'/Users/mattiarighetti/Developer/PycharmProjects/recsys/data/saved_test_train/train/train_{index}.npz')
    static_test = sps.load_npz(f'/Users/mattiarighetti/Developer/PycharmProjects/recsys/data/saved_test_train/test/test_{index}.npz')
    urm = sps.load_npz(f'/Users/mattiarighetti/Developer/PycharmProjects/recsys/data/data_matrices/data_train.npz')
    icm_subclass = sps.load_npz(f'/Users/mattiarighetti/Developer/PycharmProjects/recsys/data/data_matrices/data_ICM_sub_class.npz')
    icm_asset = sps.load_npz(f'/Users/mattiarighetti/Developer/PycharmProjects/recsys/data/data_matrices/data_ICM_asset.npz')
    icm_price = sps.load_npz(f'/Users/mattiarighetti/Developer/PycharmProjects/recsys/data/data_matrices/data_ICM_price.npz')
    ucm_age = sps.load_npz(f'/Users/mattiarighetti/Developer/PycharmProjects/recsys/data/data_matrices/data_UCM_age.npz')
    ucm_region = sps.load_npz(f'/Users/mattiarighetti/Developer/PycharmProjects/recsys/data/data_matrices/data_UCM_region.npz')
    target_users = np.load('/Users/mattiarighetti/Developer/PycharmProjects/recsys/data/data_matrices/target_users.npy')
    ucm_joined = sps.hstack((ucm_age, ucm_region))

    data = {
        'URM_all' : urm,
        'UCM_region': ucm_region,
        'UCM_age': ucm_age,
        'UCM': ucm_joined,
        'ICM_subclass': icm_subclass,
        'ICM_asset': icm_asset,
        'ICM_price': icm_price,
        'test': static_test,
        'train': static_train,
        'target_users': target_users
    }
    
    return data
