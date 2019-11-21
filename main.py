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

k_fold = TestGen(TestSplit.K_FOLD, k=3)
matrices = k_fold.get_k_fold_matrices()
target_users = k_fold.get_targetList()


for i in range(len(matrices)):
    test_matrix = matrices[i]
    train_matrix = None

    for j in range(len(matrices)):
            if j != i:
                if train_matrix is not None:
                    train_matrix += matrices[j]
                else:
                    train_matrix = matrices[j]

    itemCF1 = ItemBasedCollaborativeFiltering(train_matrix, topK=7, shrink=15)
    itemCF1.fit(similarity="tversky")
    print(evaluate_MAP_target_users(test_matrix, itemCF1, target_users))