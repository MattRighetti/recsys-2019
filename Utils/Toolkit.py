import pandas as pd
import scipy.sparse as sps
from Algorithms.Notebooks_utils.data_splitter import train_test_holdout
from Algorithms.Notebooks_utils.evaluation_function import evaluate_algorithm
from CF.item_cf import ItemBasedCollaborativeFiltering
from CF.user_cf import UserBasedCollaborativeFiltering

class DataReader(object):
    """
    This class will read the URM_train and the Target_users files and will generate every URM that we'll need
    """
    def __init__(self, filePath, targetPath):
        self.filePath = filePath
        self.targetPath = targetPath
        self.userList = []
        self.itemList = []
        self.ratingList = []
        self.targetUsersList = []

    def URM(self):
        df = pd.read_csv(self.filePath)
        target_df = pd.read_csv(self.targetPath)

        self.ratingList = list(df['data'])
        self.userList = list(df['row'])
        self.itemList = list(df['col'])
        self.targetUsersList = list(target_df['user_id'])

        return sps.coo_matrix((self.ratingList, (self.userList, self.itemList)))

    def URM_CSR(self):
        return self.URM().tocsr()

    def URM_CSC(self):
        return self.URM().tocsc()

class TestGen(object):
    """
    This class generates URM_train & URM_test matrices
    """
    def __init__(self, filePath, targetPath, train_perc=0.8):
        self.dataReader = DataReader(filePath, targetPath)
        self.URM_all_csr = self.dataReader.URM_CSR()
        self.URM_train, self.URM_test = train_test_holdout(self.URM_all_csr, train_perc=train_perc)


class Tester(object):
    """
    This class will test arrays of TopKs and Shrinks given a TestGen that will provide him the correct trainset
    and testset
    """
    def __init__(self, testGen, kind="user_cf"):
        self.testGen = testGen
        self.kind = kind

    def evaluateTopKs(self, arrayTopK=None, def_shrink=20):
        Map_values = []
        recommender = None

        if self.kind == "user_cf":
            for topK in arrayTopK:
                self.evaluateRecommender()

        elif self.kind == "item_cf":
            for topK in arrayTopK:
                recommender = ItemBasedCollaborativeFiltering(self.testGen.URM_train, topK=topK, shrink=def_shrink)
                recommender.fit()


        result_dict = evaluate_algorithm(self.testGen.URM_test, recommender)
        Map_values.append(result_dict["MAP"])

    def evaluateRecommender(self, recommender, topK, shrink):
            recommender = UserBasedCollaborativeFiltering(self.testGen.URM_train, topK=topK, shrink=def_shrink)
            recommender.fit()



class OutputFile(object):
    """
    This class will write to an output file a matrix of arrays (our data)
    """
    def __init__(self, outputFilePath):
        self.outputFile = open(outputFilePath, 'w')

    def writeLine(self, user_id, user_rec_array):
        self.outputFile.write(f'{user_id},'
                              f'{user_rec_array[0]} '
                              f'{user_rec_array[1]} '
                              f'{user_rec_array[2]} '
                              f'{user_rec_array[3]} '
                              f'{user_rec_array[4]} '
                              f'{user_rec_array[5]} '
                              f'{user_rec_array[6]} '
                              f'{user_rec_array[7]} '
                              f'{user_rec_array[8]} '
                              f'{user_rec_array[9]}\n')

    def closeFile(self):
        self.outputFile.close()