import pandas as pd
import scipy.sparse as sps

class DataReader:

    def __init__(self, filePath):
        self.filePath = filePath

    def readData(self):
        data_frame = pd.read_csv(self.filePath)


    def splitData(self, columnName):
        print("ok")