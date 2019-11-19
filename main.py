from Utils.Toolkit import TestGen, Tester

filePath = "./data/data_train.csv"
targetFile = "./data/alg_sample_submission.csv"

testGen = TestGen(filePath, targetFile)
tester = Tester(testGen, kind="user_cf")

tester.evaluateTopK_Shrink_Mixed([10, 20], [700, 720, 730, 760], boost=True)
