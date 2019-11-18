from Utils.Toolkit import TestGen, Tester, OutputFile, RecommenderGenerator
from CF.item_cf import ItemBasedCollaborativeFiltering
from Algorithms.SLIM_BPR.SLIM_BPR import SLIM_BPR

filePath = "./data/data_train.csv"
targetFile = "./data/alg_sample_submission.csv"

testGen = TestGen(filePath, targetFile, train_perc=0.8)
tester = Tester(testGen)
tester.evaluateSLIM_BPR(epoch=40)
