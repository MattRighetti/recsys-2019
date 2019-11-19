from Utils.Toolkit import TestGen, Tester, OutputFile, RecommenderGenerator
from Algorithms.SLIM_ElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from Algorithms.Notebooks_utils.evaluation_function import evaluate_algorithm
filePath = "./data/data_train.csv"
targetFile = "./data/alg_sample_submission.csv"

testGen = TestGen(filePath, targetFile)
tester = Tester(testGen)

recommender = SLIMElasticNetRecommender(testGen.URM_train)
l1_ratio = 0.3
topk=10
recommender.fit(topK=topk, l1_ratio=l1_ratio, alpha=0.3, positive_only=True, verbose=True)
print("end fit")
evaluate_algorithm(testGen.URM_test, recommender)


