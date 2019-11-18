from Utils.Toolkit import TestGen, Tester

filePath = "./data/data_train.csv"
targetFile = "./data/alg_sample_submission.csv"

testGen = TestGen(filePath, targetFile, 0.8)
tester = Tester(testGen, kind="user_cf")
tester.evaluateTopKs([5, 10, 12], def_shrink=500)
tester.evaluateShrink([100, 300, 400, 600], def_topK=7)
tester.plotTopK()
tester.plotShrink()