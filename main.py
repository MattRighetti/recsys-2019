from Utils.Toolkit import TestGen, Tester, OutputFile, RecommenderGenerator
from CF.item_cf import ItemBasedCollaborativeFiltering

filePath = "./data/data_train.csv"
targetFile = "./data/alg_sample_submission.csv"

testGen = TestGen(filePath, targetFile, train_perc=0.8)
tester = Tester(testGen, kind="user_cf")
tester.evaluateShrink([0, 10, 20, 30], def_topK=20, boost=True)
tester.plotShrink()
tester.evaluateTopKs([5, 7, 9, 12], def_shrink=600, boost=True)
tester.plotTopK()
tester = Tester(testGen, kind="item_cf")
tester.evaluateTopKs([3, 5, 7, 10], def_shrink=400, boost=True)
tester.evaluateShrink([100, 200, 300, 10], def_topK=20, boost=True)
tester.plotTopK()
tester.plotShrink()

recommender = ItemBasedCollaborativeFiltering(testGen.URM_train, topK=20, shrink=10)
recommender.fit()

outputFile = OutputFile("./output/file_out.csv")
outputFile.write_output(recommender, testGen)