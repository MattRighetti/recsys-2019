from Utils.Toolkit import TestGen, Tester

filePath = "./data/data_train.csv"
targetFile = "./data/alg_sample_submission.csv"

testGen = TestGen(filePath, targetFile, train_perc=0.8)
tester = Tester(testGen, kind="user_cf")
tester.evaluateShrink([0, 10, 20, 30], def_topK=20, boost=True)
tester.plotShrink()
tester.evaluateTopKs([5, 7, 9, 12], def_shrink=600, boost=True)
tester.plotTopK()