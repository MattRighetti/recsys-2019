from Utils.Toolkit import TestGen, Tester, OutputFile
from Utils.Toolkit import TestSplit
from CBF.item_CBF import ItemContentBasedRecommender
from multiprocessing import Process

testGen = TestGen(test=TestSplit.LEAVE_K_OUT, k=10)
recommender = ItemContentBasedRecommender(testGen.URM_train, 10,20)
recommender.fit()
recommender.evaluate(testGen.URM_test)