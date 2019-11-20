from Algorithms.Notebooks_utils.evaluation_function import evaluate_MAP
from HYB.hybrid import HybridRecommender
from Utils.Toolkit import TestGen, Tester, OutputFile, DataReader
from Utils.Toolkit import TestSplit
from Utils.Evaluator_new import Evaluator
from Algorithms.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from CF.user_cf import UserBasedCollaborativeFiltering
from CF.item_cf import ItemBasedCollaborativeFiltering
from CBF.item_CBF import ItemContentBasedRecommender
from multiprocessing import Process

#force_leave_k_out_5 = TestGen(test=TestSplit.FORCE_LEAVE_K_OUT, k=5)
leave_k_out = TestGen(test=TestSplit.LEAVE_K_OUT, k=10)
#force_leave_k_out_10 = TestGen(test=TestSplit.FORCE_LEAVE_K_OUT, k=10)

itemCF1 = ItemBasedCollaborativeFiltering(leave_k_out.URM_train, topK=22, shrink=210)
itemCF1.fit()
itemCF2 = ItemBasedCollaborativeFiltering(leave_k_out.URM_train, topK=10, shrink=10)
itemCF2.fit()

print(evaluate_MAP(leave_k_out.URM_test, itemCF2))
print(evaluate_MAP(leave_k_out.URM_test, itemCF1))

ev = Evaluator([itemCF1, itemCF2])
ev.evaluate_recommenders(leave_k_out.URM_test)
print(ev.get_best_recommender().get_topK())