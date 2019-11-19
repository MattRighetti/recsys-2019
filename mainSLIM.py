from Utils.Toolkit import TestGen, Tester, OutputFile, RecommenderGenerator
from CF.item_cf import ItemBasedCollaborativeFiltering
from Algorithms.SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from Algorithms.Base.Evaluation.Evaluator import EvaluatorHoldout

filePath = "./data/data_train.csv"
targetFile = "./data/alg_sample_submission.csv"

testGen = TestGen(filePath, targetFile)

i = 100
recommender = SLIM_BPR_Cython(testGen.URM_train)

for topK in range(10):
    recommender.fit(i,topK=topK, sgd_mode="sgd")
    ev = EvaluatorHoldout(testGen.URM_test, [10])
    result = ev.evaluateRecommender(recommender)
    print(result['MAP'])
