from enum import Enum
import numpy as np
import pandas as pd
import scipy.sparse as sps
from sklearn import feature_extraction

from Algorithms.Base.IR_feature_weighting import okapi_BM_25
from Utils.DataSplitter import DataSplitter
from sklearn.preprocessing import normalize


class DataReader(object):
    """
    This class will read the URM_train and the Target_users files and will generate every URM that we'll need
    """
    def __init__(self, test=False):
        if test:
            dir_file = "../"
        else:
            dir_file = "./"
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
        icm_items_list = list(df['row'])
        asset_list = list(df['col'])
        item_asset_list = list(df['data'])
        return sps.coo_matrix((item_asset_list, (icm_items_list, asset_list)), dtype=np.float64)

    def ICM_price_COO(self):
        df = pd.read_csv(self.item_price_file_path)
        icm_items_list = list(df['row'])
        price_list = list(df['col'])
        item_price_list = list(df['data'])
        return sps.coo_matrix((item_price_list, (icm_items_list, price_list)), dtype=np.float64)

    def UCM_region_COO(self):
        df = pd.read_csv(self.user_region_file_path)
        ucm_user_list = list(df['row'])
        region_list = list(df['col'])
        user_region_list = list(df['data'])
        return sps.coo_matrix((user_region_list, (ucm_user_list, region_list)), dtype=np.float64)

    def UCM_age_COO(self):
        df = pd.read_csv(self.user_age_file_path)
        ucm_user_list = list(df['row'])
        age_list = list(df['col'])
        user_age_list = list(df['data'])
        return sps.coo_matrix((user_age_list, (ucm_user_list, age_list)), dtype=np.float64)

class TestSplit(Enum):
    LEAVE_ONE_OUT = 1
    LEAVE_K_OUT = 2
    FORCE_LEAVE_K_OUT = 3
    K_FOLD = 4

class TestGen(object):
    """
    This class generates URM_train & URM_test matrices
    """
    def __init__(self, URM_all_csr, kind=TestSplit.FORCE_LEAVE_K_OUT, k=10):
        if kind is TestSplit.FORCE_LEAVE_K_OUT:
            self.URM_train, self.URM_test = DataSplitter(URM_all_csr).force_leave_k_out(k)
        elif kind is TestSplit.LEAVE_K_OUT:
            self.URM_train, self.URM_test = DataSplitter(URM_all_csr).leave_k_out(k)
        elif kind is TestSplit.LEAVE_ONE_OUT:
            self.URM_train, self.URM_test = DataSplitter(URM_all_csr).leave_one_out()
        elif kind is TestSplit.K_FOLD:
            self.Matrices = DataSplitter(URM_all_csr).k_fold(k=k)

    def get_k_fold_matrices(self):
        return self.Matrices

class Tuner(object):
    @staticmethod
    def tune(recommender, split_kind=None, at=10):
        print(f'Running {recommender.NAME}...')

        if split_kind is None:
            data = get_data()
        else:
            data = get_data(split_kind)

        score = global_evaluate_single(recommender)
        print(f'Evaluation score: {score}')


#########################################################################################################
#                                               UTILITIES                                               #
#########################################################################################################

def get_URM_BM_25(URM):
    return okapi_BM_25(URM)

def normalize_matrix(URM, axis=0):
    n_matrix = normalize(URM, 'l2', axis)
    return n_matrix.tocsr()

def get_URM_TFIDF(URM):
    URM_tfidf = feature_extraction.text.TfidfTransformer().fit_transform(URM)
    return URM_tfidf.tocsr()

def get_data(split_kind=None, test=False):
    dataReader = DataReader(test=test)
    UCM_region = dataReader.UCM_region_COO()
    UCM_age = dataReader.UCM_age_COO()
    URM_all = dataReader.URM_COO()
    ICM_price = dataReader.ICM_price_COO()
    ICM_asset = dataReader.ICM_price_COO()
    ICM_subclass = dataReader.ICM_subclass_COO()
    target_users = dataReader.target_users()

    if split_kind is None:
        testGen = TestGen(URM_all.tocsr(), TestSplit.LEAVE_ONE_OUT)
    else:
        testGen = TestGen(URM_all.tocsr(), split_kind)

    data = {
        'URM_all': URM_all,
        'train': testGen.URM_train,
        'test': testGen.URM_test,
        'target_users': target_users,
        'UCM_region': UCM_region,
        'UCM_age': UCM_age,
        'ICM_price': ICM_price,
        'ICM_asset': ICM_asset,
        'ICM_subclass': ICM_subclass
    }

    return data