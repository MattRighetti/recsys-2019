from enum import Enum

import pandas as pd
import os
import datetime
import scipy.sparse as sps
from HYB.hybrid import HybridRecommender
from Utils.DataSplitter import DataSplitter
from Algorithms.Notebooks_utils.evaluation_function import evaluate_algorithm
from Algorithms.SLIM_BPR.SLIM_BPR import SLIM_BPR
from CF.item_cf import ItemBasedCollaborativeFiltering
from CF.user_cf import UserBasedCollaborativeFiltering
import matplotlib.pyplot as pyplot
from multiprocessing import Process, Array


class DataReader(object):
    """
    This class will read the URM_train and the Target_users files and will generate every URM that we'll need
    """
    def __init__(self):
        self.data_train_file_path = "./data/data_train.csv"
        self.user_target_file_path = "./data/alg_sample_submission.csv"
        self.userList = []
        self.itemList = []
        self.ratingList = []
        self.targetUsersList = []

    def URM_COO(self):
        df = pd.read_csv(self.data_train_file_path)
        target_df = pd.read_csv(self.user_target_file_path)

        self.ratingList = list(df['data'])
        self.userList = list(df['row'])
        self.itemList = list(df['col'])
        self.targetUsersList = list(target_df['user_id'])

        return sps.coo_matrix((self.ratingList, (self.userList, self.itemList)))

    def URM_CSR(self):
        return self.URM_COO().tocsr()

    def URM_CSC(self):
        return self.URM_COO().tocsc()

class TestSplit(Enum):
    LEAVE_ONE_OUT = 1
    LEAVE_K_OUT = 2
    FORCE_LEAVE_K_OUT = 3

class TestGen(object):
    """
    This class generates URM_train & URM_test matrices
    """

    def __init__(self, test=TestSplit.FORCE_LEAVE_K_OUT, k=10):
        self.dataReader = DataReader()
        self.URM_all_csr = self.dataReader.URM_CSR()

        if test is TestSplit.FORCE_LEAVE_K_OUT:
            self.URM_train, self.URM_test = DataSplitter(self.URM_all_csr).force_leave_k_out(k)
        elif test is TestSplit.LEAVE_K_OUT:
            self.URM_train, self.URM_test = DataSplitter(self.URM_all_csr).leave_k_out(k)
        elif test is TestSplit.LEAVE_ONE_OUT:
            self.URM_train, self.URM_test = DataSplitter(self.URM_all_csr).leave_one_out()

    def get_dataReader(self):
        return self.dataReader


class RecommenderGenerator(object):
    """
    Factory of Recommenders
    """
    def __init__(self, testGen):
        self.recommender = None
        self.testGen = testGen

    def get_recommender(self):
        if self.recommender is not None:
            return self.recommender

    def setTopK(self, topK):
        self.recommender.set_topK(topK)

    def initUserCF(self, topK=None, shrink=None):
        self.recommender = UserBasedCollaborativeFiltering(self.testGen.URM_train, topK=topK, shrink=shrink)
        return self.get_recommender()

    def initItemCF(self, topK=None, shrink=None):
        self.recommender = ItemBasedCollaborativeFiltering(self.testGen.URM_train, topK=topK, shrink=shrink)
        return self.get_recommender()

    def initHybrid(self):
        self.recommender = HybridRecommender(self.testGen.URM_train)
        return self.get_recommender()

    def initSLIM_BPR(self):
        self.recommender = SLIM_BPR(self.testGen.URM_train)

class Tester(object):
    """
    This class will test arrays of TopKs and Shrinks given a TestGen that will provide him the correct trainset
    and testset
    """
    def __init__(self, testGen, kind="user_cf"):
        self.testGen = testGen
        self.kind = kind
        self.arrayShrink = []
        self.arrayTopK = []
        self.MAP_TopK = []
        self.MAP_Shrink = []
        self.MAP_Shrink_TopK = []

    def evaluateTopKs(self, arrayTopK=None, def_shrink=20, boost=False):
        """
        CollaborativeRecommenders only
        :param arrayTopK: array of topK values to evaluate
        :param def_shrink: value of shrink value to be used in the evaluation
        :param boost: parameter that's going to activate multiprocessing if set to true
        :return: None
        """
        self.MAP_TopK = Array('d', 4)
        self.arrayTopK = arrayTopK
        recommender = None

        if self.kind == "user_cf":
            recommender = RecommenderGenerator(self.testGen).initUserCF()
        elif self.kind == "item_cf":
            recommender = RecommenderGenerator(self.testGen).initItemCF()

        recommender.set_shrink(def_shrink)

        if boost:
            processes = []
            counter = 0

            for topK in arrayTopK:
                p = Process(target=self.evaluateAndAppend,
                            args=[self.MAP_TopK, recommender, topK, "top_k", True, counter])
                p.start()
                processes.append(p)
                counter += 1

            for process in processes:
                process.join()

        else:
            self.MAP_TopK = []
            for topK in arrayTopK:
                self.evaluateAndAppend(self.MAP_TopK, recommender, topK, kind="top_k")

    def evaluateShrink(self, arrayShrink=None, def_topK=20, boost=False):
        """
        CollaborativeRecommenders only
        :param arrayShrink: array of shrink values to evaluate
        :param def_topK: default topK to be used in the evaluation
        :param boost: parameter that's going to activate multiprocessing if set to true
        :return: None
        """
        self.MAP_Shrink = Array('d', 4)
        self.arrayShrink = arrayShrink
        recommender = None

        if self.kind == "user_cf":
            recommender = RecommenderGenerator(self.testGen).initUserCF()
        elif self.kind == "item_cf":
            recommender = RecommenderGenerator(self.testGen).initItemCF()

        recommender.set_topK(def_topK)

        if boost:
            processes = []
            counter = 0

            for shrink in arrayShrink:
                p = Process(target=self.evaluateAndAppend,
                            args=[self.MAP_Shrink, recommender, shrink, "shrink", True, counter])

                p.start()
                processes.append(p)
                counter += 1

            for process in processes:
                process.join()

        else:
            self.MAP_Shrink = []
            for shrink in arrayShrink:
                self.evaluateAndAppend(self.MAP_Shrink, recommender, shrink, kind="shrink")

    def evaluateTopK_Shrink_Mixed(self, arrayTopKs, arrayShrinks, boost=False):
        """
        Permutation of both the arrays and evaluation
        """
        length = len(arrayTopKs)*len(arrayShrinks)
        self.MAP_Shrink_TopK = Array('d', length)
        self.arrayShrink = arrayShrinks
        self.arrayTopK = arrayTopKs
        recommender = None

        if self.kind == "user_cf":
            recommender = RecommenderGenerator(self.testGen).initUserCF()
        elif self.kind == "item_cf":
            recommender = RecommenderGenerator(self.testGen).initUserCF()

        if boost:
            processes = []
            counter = 0

            for shrink in arrayShrinks:
                for topK in arrayTopKs:
                    p = Process(target=self.evaluateAndAppend,
                                args=[self.MAP_Shrink_TopK, recommender, [topK, shrink], "both", True, counter])
                    counter += 1
                    processes.append(p)

            for process in processes:
                process.start()

            for process in processes:
                process.join()


        else:
            self.MAP_Shrink_TopK = []
            for topK in arrayTopKs:
                for shrink in arrayShrinks:
                    self.evaluateAndAppend(self.MAP_Shrink_TopK, recommender, shrink, topK)


    def evaluateSLIM_BPR(self, epoch=1, lambda_i=0.025, lambda_j=0.025, learning_rate=0.05):
        MAP_array = []
        recommender = SLIM_BPR(URM_train=self.testGen.URM_train, lambda_i=lambda_i, lambda_j=lambda_j, learning_rate=learning_rate)
        recommender.fit(epochs=epoch)

        result_dict = evaluate_algorithm(self.testGen.URM_test, recommender)
        MAP_array.append(result_dict["MAP"])

        map_result = result_dict['MAP']
        print("{} -> MAP: {:.4f}\t".format(self.kind, map_result))

    def evaluate_HYB(self, userCF_w, itemCF_w, test_value=""):
        recommender = HybridRecommender(self.testGen.URM_train)
        recommender.fit(userCF_w=userCF_w, itemCF_w=itemCF_w)

        result_dict = evaluate_algorithm(self.testGen.URM_test, recommender)

        map_result = result_dict['MAP']
        print("HYB -> MAP: {:.4f}, UserCF_weight: {}, ItemCFweight: {}, TestGen: {}".format(map_result, userCF_w, itemCF_w, test_value))

    def evaluateAndAppend(self, MAP_array, recommender, value, kind="shrink", boost=False, index=None):

        if kind == "shrink":
            recommender.set_shrink(value)
        elif kind == "top_k":
            recommender.set_topK(value)
        elif kind == "both":
            recommender.set_topK(value[0])
            recommender.set_shrink(value[1])

        recommender.fit()

        result_dict = evaluate_algorithm(self.testGen.URM_test, recommender)

        if boost:
            MAP_array[index] = result_dict["MAP"]
        else:
            MAP_array.append(result_dict["MAP"])

        map_result = result_dict['MAP']
        print("{} -> MAP: {:.4f} with TopK = {} "
              "& Shrink = {}\t".format(self.kind, map_result, recommender.get_topK(), recommender.get_shrink()))

    def evaluate(self, topK, shrink):
        recommender = None

        if self.kind == "user_cf":
            recommender = UserBasedCollaborativeFiltering(self.testGen.URM_train, topK, shrink)
        elif self.kind == "item_cf":
            recommender = ItemBasedCollaborativeFiltering(self.testGen.URM_train, topK, shrink)

        recommender.fit()

        result_dict = evaluate_algorithm(self.testGen.URM_test, recommender)
        map_result = result_dict['MAP']
        print("{} -> MAP: {:.4f} with TopK = {} "
              "& Shrink = {}\t".format(self.kind, map_result, recommender.get_topK(), recommender.get_shrink()))

    def multiProcEvaluateShrink(self, arrayShrink=None, def_topK=20):
        processes = []

        for shrink in arrayShrink:
            p = Process(target=self.evaluate, args=[def_topK, shrink])
            p.start()
            processes.append(p)

        for process in processes:
            process.join()

    def multiProcEvaluateTopK(self, arrayTopK=None, def_shrink=20):
        processes = []

        for topK in arrayTopK:
            p = Process(target=self.evaluate, args=[topK, def_shrink])
            p.start()
            processes.append(p)

        for process in processes:
            process.join()

    def plotShrink(self):
        pyplot.plot(self.arrayShrink, self.MAP_Shrink)
        pyplot.ylabel('MAP')
        pyplot.xlabel('Shrink')
        pyplot.show()

    def plotTopK(self):
        pyplot.plot(self.arrayTopK, self.MAP_TopK)
        pyplot.ylabel('MAP')
        pyplot.xlabel('TopK')
        pyplot.show()


class OutputFile(object):
    """
    This class will write to an output file a matrix of arrays (our data)
    """
    def __init__(self):
        self.outputFile = self.create_unique_file()

    def write_output(self, fittedRecommender, testGen):
        """
        Create a new file and writes to it
        :param fittedRecommender: recommender already fitted
        :param testGen:
        :return:
        """
        file = open(self.outputFile, "w+")
        file.write("user_id,item_list\n")

        dataReader = testGen.get_dataReader()

        for user_id in dataReader.targetUsersList:
            recommendations = fittedRecommender.recommend(user_id, at=10)
            array_string = " ".join(str(x) for x in recommendations)
            file.write(f'{user_id},{array_string}\n')

        file.close()

    def create_unique_file(self):
        """
        Creates filename
        :return: New random filename file_out_day_hour_minute_second.csv
        """
        current_date = datetime.datetime.now()
        folder_path = "./output"

        file_name = f'file_out_{current_date.day}_{current_date.hour}_{current_date.minute}_{current_date.second}.csv'

        print(file_name)

        return os.path.join(folder_path, file_name)