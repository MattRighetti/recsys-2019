from Utils.Toolkit import TestGen, DataReader
from Utils.Toolkit import TestSplit
from CF.item_cf import ItemBasedCollaborativeFiltering

import numpy as np

dr = DataReader()
URM_all_CSR = dr.URM_CSR()
target_users = dr.targetUsersList
lou = TestGen(URM_all_CSR, TestSplit.LEAVE_ONE_OUT)
URM_train = lou.URM_train
print(URM_train.nnz)
URM_test = lou.URM_test
print(URM_test.nnz)

itemCF1 = ItemBasedCollaborativeFiltering(URM_train=URM_train, topK=2, shrink=2)
itemCF1.fit()
print(itemCF1.evaluate_MAP_target(URM_test, target_users))

itemCF1 = ItemBasedCollaborativeFiltering(URM_train, 2,2)
itemCF1.fit()
print(itemCF1.evaluate_MAP_target(URM_test, target_users))


