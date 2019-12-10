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

def generate_SM_user_feature_matrix(URM_train, ICM):
    """
    Generates a UFM matrix, the dot product looked incorrect
    :return: UFM matrix ( USER x ITEMFEATURES )
    """
    data = []
    users = []
    features = []

    for user_id in tqdm(range(URM_train.shape[0]), desc="Evaluating SM_user_feature_matrix"):
        u_start_pos = URM_train.indptr[user_id]
        u_end_pos = URM_train.indptr[user_id + 1]

        mask = URM_train.indices[u_start_pos:u_end_pos]
        zero_mask = np.zeros(ICM.shape[1], dtype=int)
        user_row = zero_mask.copy() + user_id

        if len(mask) > 0:
            features_matrix = ICM[mask,:].sum(axis=0)
            user_features = np.squeeze(np.asarray(features_matrix))

            data.append(user_features)
            users.append(user_row)
            features.append(np.arange(ICM.shape[1]))
        else:
            data.append(zero_mask.copy())
            users.append(user_row)
            features.append(np.arange(ICM.shape[1]))

    data = np.asarray(data).ravel()
    users = np.asarray(users).ravel()
    features = np.asarray(features).ravel()

    mask_non_zero = data > 0

    data = data[mask_non_zero]
    users = users[mask_non_zero]
    features = features[mask_non_zero]

    SM_user_feature_matrix = sps.coo_matrix((data, (users, features)),
                                            dtype=int,
                                            shape=(URM_train.shape[0], ICM.shape[1]))

    SM_user_feature_matrix = SM_user_feature_matrix.tocsr()
    print(f'Generated UFM with shape {SM_user_feature_matrix.shape}')
    return SM_user_feature_matrix

def feature_boost_URM(URM_in, topN=5, min_interactions = 5, kind="subclass"):

    print(f'TopN={topN}, min_interactions={min_interactions}')

    boost_weight = 1

    data = get_data()
    URM = URM_in.copy()
    ICM = None
    boost_val = 0
    if kind == "subclass":
        ICM = data['ICM_subclass'].tocsr()
        boost_val = 5
    if kind == "asset":
        ICM = data['ICM_asset'].tocsr()
        boost_weight = 0.5
        boost_val = 5
    if kind == "price":
        ICM = data['ICM_price'].tocsr()
        boost_val = 2
    #target_users = data['target_users']

    RM_features, RM_features_scores = get_features_ratings(URM, ICM, at=topN)

    # Scorri tutti gli utenti nella URM
    for user_id in tqdm(range(URM.shape[0]), desc="Boosting URM..."):
        # Prendi solo utenti target
        #if user_id in target_users:
            # Prendi le topN features di un utente
            topNfeatures = RM_features[user_id].data
            # Prendi i rating delle topN features di un utente
            topNfeatures_ratings = RM_features_scores[user_id].data
            # Prendi il gradiente degli score delle topN features
            gradient_array = np.ediff1d(topNfeatures_ratings)

            rank = [1]
            for i in range(len(topNfeatures) - 1):
                if gradient_array[i] == 0:
                    rank.append(rank[i])
                else:
                    rank.append(rank[i] + 1)

            rank = np.array(rank)

            # Prendi tutte le item con cui un utente ha interagito
            user_interacted_items = URM[user_id].indices
            # Conta numero di items con cui l'utente ha interagito
            n_items = len(user_interacted_items)

            # Se il numero di items con cui ha interagito è minore del numero di interactions minime
            # Possiamo supporre che abbia interagito con items di suo interesse
            if n_items < min_interactions:
                start_pos = URM.indptr[user_id]
                end_pos = URM.indptr[user_id + 1]
                URM.data[start_pos:end_pos] = 5
            else:
                # Per ogni item
                for i in range(n_items):
                    # Prendi tutte le features dell'item
                    item_features = ICM[user_interacted_items[i]].indices
                    # Per la feature dell'item in considerazione
                    # TODO this is wrong, cumulative sum is different for items that has more than one feature
                    # Determina in che posizione si trova l'item se è tra le topN
                    if len(item_features) != 0:
                        feature_position, = np.where(topNfeatures == item_features[0])
                        # Se la pozione è nulla allora la feature non è da boostare
                        # altrimenti boosta
                        if len(feature_position) != 0:
                            # TODO logic here
                            additive_score = boost_val * (1 / (rank[feature_position[0]]) ) * boost_weight
                            #print(f'Added {additive_score} to item {i} to user {user_id}')
                            URM.data[URM.indptr[user_id] + i] += additive_score

    return URM

def get_features_ratings(URM_in, ICM_in, at=10):
    data = []
    users = []
    features = []
    scores = []

    URM = URM_in.copy()
    ICM = ICM_in.copy()

    SM_user_feature = generate_SM_user_feature_matrix(URM, ICM)

    # Scorro ogni utente
    for i in tqdm(range(SM_user_feature.shape[0]), desc="Getting ratings..."):
        # Creo un array vuoto
        recommended_features = np.zeros(SM_user_feature.shape[1])
        # Metto in ordine di importanza l'indice delle features più interacted
        recommended_features_indexes = np.flip(np.argsort(SM_user_feature[i].toarray().ravel()), 0)
        # Metto in ordine i ratings delle features in base all'array sopra
        ordered_features = SM_user_feature[i].toarray().ravel()[recommended_features_indexes]
        # Credo una maschera che ha come 1 i valori per cui le features sono state interacted with
        ordered_features_mul = np.where(ordered_features > 0, 1, ordered_features)
        # Creo l'array di features più importanti
        recommended_features = recommended_features + recommended_features_indexes
        # Metto a 0 le features che l'utente non ha mai avuto a che fare
        # Altrimenti l'argsort avrebbe ordinato a random anche gli zeri
        recommended_features *= ordered_features_mul

        # Costruisco COO Matrix man mano
        users_mask = np.zeros(SM_user_feature.shape[1], dtype=int) + i
        features_mask = np.arange(SM_user_feature.shape[1])

        users.append(users_mask)
        features.append(features_mask)

        data.append(recommended_features)
        scores.append(ordered_features)

    data = np.asarray(data).ravel()
    users = np.asarray(users).ravel()
    features = np.asarray(features).ravel()
    scores = np.asarray(scores).ravel()

    mask_non_zero = data > 0

    data = data[mask_non_zero]
    users = users[mask_non_zero]
    features = features[mask_non_zero]
    scores = scores[mask_non_zero]

    data_coo = sps.coo_matrix((data, (users.copy(), features.copy())), shape=SM_user_feature.shape)
    scores_coo = sps.coo_matrix((scores, (users.copy(), features.copy())), shape=SM_user_feature.shape)

    data_csr = data_coo.tocsc()
    scores_csr = scores_coo.tocsr()

    RM_user_feature = data_csr[:, :at]
    RM_user_feature_ratings = scores_csr[:, :at]

    return RM_user_feature, RM_user_feature_ratings


def rerank_based_on_ICM(UFM, recommended_items, recommended_items_ratings, user_id):
    data = get_data()

    ICM = data['ICM_subclass'].tocsr()
    UFM = UFM.copy()

    for item_index in range(len(recommended_items)):
        item_id = recommended_items[item_index]

        start_pos = ICM.indptr[item_id]
        end_pos = ICM.indptr[item_id + 1]

        # Features dell'item
        item_profile = ICM.indices[start_pos:end_pos]

        start_pos = UFM.indptr[user_id]
        end_pos = UFM.indptr[user_id + 1]

        user_feature_profile = UFM.indices[start_pos:end_pos]

        feature_position = np.where(user_feature_profile == item_profile[0])

        if len(feature_position) > 0:
            additive_score = 1
            recommended_items_ratings[item_index] += additive_score

    order_mask = np.flip(np.argsort(recommended_items_ratings), 0)

    return recommended_items[order_mask]


def get_data(split_kind=None):
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
        'UCM' : UCM
    }

    return data