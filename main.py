from Algorithms.Notebooks_utils.evaluation_function import evaluate_MAP, evaluate_MAP_target_users
from HYB.hybrid import HybridRecommender
from Utils.Toolkit import TestGen, Tester, OutputFile, DataReader
from Utils.Toolkit import TestSplit
from Utils.Evaluator_new import Evaluator
from Algorithms.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from CF.user_cf import UserBasedCollaborativeFiltering
from CF.item_cf import ItemBasedCollaborativeFiltering
from CBF.item_CBF import ItemContentBasedRecommender
from multiprocessing import Process

k_fold = TestGen(TestSplit.K_FOLD, k=4)
matrices = k_fold.get_k_fold_matrices()
print(len(matrices))
target_users = k_fold.get_targetList()



#itemCF1 = ItemBasedCollaborativeFiltering(leave_k_out.URM_train, topK=topK, shrink=15)
#itemCF1.fit(similarity="tanimoto")
#print(evaluate_MAP_target_users(leave_k_out.URM_test, itemCF1, target_users))