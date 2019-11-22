from Algorithms.Notebooks_utils.evaluation_function import evaluate_MAP_target_users
from Utils.Toolkit import TestGen, DataReader
from Utils.Toolkit import TestSplit
from CF.item_cf import ItemBasedCollaborativeFiltering
from Utils.OutputWriter import write_output

import numpy as np

dr = DataReader()
URM_all_CSR = dr.URM_CSR()
target_users = dr.targetUsersList
lou = TestGen(URM_all_CSR, TestSplit.LEAVE_ONE_OUT)
URM_train = lou.URM_train
URM_test = lou.URM_test
topK_collection = [10, 15, 22, 35, 50, 70, 90, 120, 150]
shrink_collection = [5, 10, 20, 30, 40, 50, 80, 120, 200]
max_map = 57
max_topK = 0
max_shrink = 0

URM_final = URM_train + URM_test

itemCF1 = ItemBasedCollaborativeFiltering(URM_final, 50, 90)
itemCF1.fit()
write_output(itemCF1, target_users)
