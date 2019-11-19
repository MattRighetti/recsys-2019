from Algorithms.Notebooks_utils.evaluation_function import evaluate_algorithm
from HYB.hybrid import HybridRecommender
from Utils.Toolkit import TestGen, Tester, OutputFile
from Utils.Toolkit import TestSplit
from CF.user_cf import UserBasedCollaborativeFiltering
from CF.item_cf import ItemBasedCollaborativeFiltering
from CBF.item_CBF import ItemContentBasedRecommender
from multiprocessing import Process

#force_leave_k_out_5 = TestGen(test=TestSplit.FORCE_LEAVE_K_OUT, k=5)
leave_k_out = TestGen(test=TestSplit.LEAVE_K_OUT, k=10)
#force_leave_k_out_10 = TestGen(test=TestSplit.FORCE_LEAVE_K_OUT, k=10)

#tester_1 = Tester(leave_k_out)
#tester_1_item = Tester(leave_k_out, kind="item_cf")
#tester_2 = Tester(force_leave_k_out_5)
#tester_3 = Tester(force_leave_k_out_10)
"""
processes = []

for i in range(1,4):
    for j in range(1,5):
        p1 = Process(target=tester_1.evaluate_HYB, args=[j, i, "1"])
        processes.append(p1)

i = 0
while i < len(processes):
    processes[i].start()
    processes[i+1].start()
    processes[i+2].start()
    processes[i+3].start()
    processes[i].join()
    processes[i+1].join()
    processes[i+2].join()
    processes[i+3].join()
    i+=4


tester_1.evaluateTopKs([5, 10, 12, 7], def_shrink=600, boost=True)
tester_1.evaluateShrink([500, 600, 700, 800], def_topK=5, boost=True)
tester_1_item.evaluateShrink([100, 300, 400, 600], def_topK=10, boost=True)
tester_1_item.evaluateTopKs([10, 20, 30, 40], def_shrink=100, boost=True)
"""

#user_cf = UserBasedCollaborativeFiltering(leave_k_out.URM_train, topK=7, shrink=700)
#user_cf.fit()
#user_cf.evaluate(leave_k_out.URM_test)

item_CBF = ItemContentBasedRecommender(leave_k_out.ICM, leave_k_out.URM_train, topK=20, shrink=100)
item_CBF.fit()
item_CBF.evaluate(leave_k_out.URM_test)