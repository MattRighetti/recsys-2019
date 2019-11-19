from Algorithms.Notebooks_utils.evaluation_function import evaluate_algorithm
from HYB.hybrid import HybridRecommender
from Utils.Toolkit import TestGen, Tester, OutputFile
from Utils.Toolkit import TestSplit
from multiprocessing import Process

testGen1 = TestGen(test=TestSplit.LEAVE_ONE_OUT)
testGen2 = TestGen(test=TestSplit.FORCE_LEAVE_K_OUT, k=5)
testGen3 = TestGen(test=TestSplit.LEAVE_K_OUT, k=10)
testGen4 = TestGen(test=TestSplit.FORCE_LEAVE_K_OUT, k=10)
tester1 = Tester(testGen1)
tester2= Tester(testGen2)
tester3 = Tester(testGen3)
tester4 = Tester(testGen4)

p1 = Process(target=tester1.evaluate_HYB, args=[1, 2, "1"])
p2 = Process(target=tester2.evaluate_HYB, args=[1, 2, "2"])
p3 = Process(target=tester3.evaluate_HYB, args=[1, 2, "3"])
p4 = Process(target=tester4.evaluate_HYB, args=[1, 2, "4"])

p1.start()
p2.start()
p3.start()
p4.start()
p1.join()
p2.join()
p3.join()
p4.join()
