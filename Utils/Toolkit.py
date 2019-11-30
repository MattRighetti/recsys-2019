from Algorithms.Base.IR_feature_weighting import okapi_BM_25
from sklearn.preprocessing import normalize
from Utils.DataSplitter import DataSplitter
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
    def __init__(self, dir_path='./'):
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
        icm_price = self.ICM_price_COO()
        icm_subclass = self.ICM_price_COO()
        icm_total = sps.hstack((icm_asset, icm_subclass, icm_price))
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


from Recommenders.IFC.Cython.BoostSimilarityMatrix import Booster
class BoosterObject(object):

    def __init__(self, URM, ICM):
        self.URM = URM
        self.ICM = ICM
        self.UFM = generate_SM_user_feature_matrix(URM, ICM)
        self.booster = Booster()

        # self.ICM = get_URM_BM_25(self.ICM)
        self.ICM = get_URM_TFIDF(self.ICM)
        self.ICM = normalize_matrix(self.ICM)
        # self.SM_user_feature = get_URM_TFIDF(self.SM_user_feature)
        self.UFM = normalize_matrix(self.UFM)

    def boost_ratings_ICM(self, recommended_items, recommended_items_ratings, user_id):
        """
        Use after reordering the first recommended items
        :param recommended_items: Items recommended ( index of higher score first )
        :param recommended_items_ratings: Ratings of each Item
        :param user_id: User ID
        :return: New ICM weighted
        If detected COLD user returns the same dataset
        """
        boosted_ratings = self.booster.boost(recommended_items,
                                            recommended_items_ratings,
                                            user_id,
                                            1, 1.25,
                                            self.ICM,
                                            self.UFM)

        return boosted_ratings


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

def get_URM_TFIDF(URM):
    """
    Applies TF-IDF weighting to the passed matrix
    :param URM:
    :return:
    """
    URM_tfidf = feature_extraction.text.TfidfTransformer().fit_transform(URM)
    return URM_tfidf.tocsr()

def generate_SM_user_feature_matrix(URM_train, ICM):
    """
    Generates a UFM matrix, the dot product looked incorrect
    :return: UFM matrix ( USER x ITEMFEATURES )
    """
    SM_user_feature_matrix = np.zeros((URM_train.shape[0], ICM.shape[1]), dtype=int)

    for user_id in tqdm(range(URM_train.shape[0]), desc="Evaluating SM_user_feature_matrix"):
        u_start_pos = URM_train.indptr[user_id]
        u_end_pos = URM_train.indptr[user_id + 1]

        mask = URM_train.indices[u_start_pos:u_end_pos]

        if len(mask) > 0:
            features_matrix = ICM[mask,:].sum(axis=0)
            user_features = np.squeeze(np.asarray(features_matrix))
        else:
            user_features = np.zeros(ICM.shape[1], dtype=int)

        SM_user_feature_matrix[user_id] = user_features

    SM_user_feature_matrix = sps.csr_matrix(SM_user_feature_matrix)
    print(f'Generated UFM with shape {SM_user_feature_matrix.shape}')
    return SM_user_feature_matrix

def get_data(split_kind=None, dir_path=None):
    dataReader = DataReader(dir_path=dir_path)
    UCM_region = dataReader.UCM_region_COO()
    UCM_age = dataReader.UCM_age_COO()
    URM_all = dataReader.URM_COO()
    ICM_price = dataReader.ICM_price_COO()
    ICM_asset = dataReader.ICM_price_COO()
    ICM_subclass = dataReader.ICM_subclass_COO()
    ICM = dataReader.ICM_total()
    UCM = dataReader.UCM_total()
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
        'ICM_subclass': ICM_subclass,
        'ICM' : ICM,
        'UCM' : UCM
    }

    return data