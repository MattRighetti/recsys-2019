from Algorithms.Notebooks_utils.evaluation_function import evaluate_MAP_target_users
from Utils.Toolkit import TestGen, DataReader
from Utils.Toolkit import TestSplit
from CF.item_cf import ItemBasedCollaborativeFiltering
from CF.user_cf import UserBasedCollaborativeFiltering
from Utils.OutputWriter import write_output

import numpy as np

URM_all_CSR = DataReader().URM_CSR()
target_users = DataReader().targetUsersList

lou_matrices = TestGen(URM_all_CSR, TestSplit.LEAVE_ONE_OUT)

URM_train = lou_matrices.URM_train
URM_test = lou_matrices.URM_test

topK_collection = np.arange(5, 11, 5)
shrink_collection = np.arange(20, 26, 1)
max_map = 0
max_topK = 0
max_shrink = 0

for topK in topK_collection:
    for shrink in shrink_collection:

        itemCF1 = ItemBasedCollaborativeFiltering(topK=topK, shrink=shrink)
        itemCF1.fit(URM_train)
        map_elem = evaluate_MAP_target_users(URM_test, itemCF1, target_users)
        print(map_elem)

        if map_elem > max_map:
            max_map = map_elem
            max_topK = topK
            max_shrink = shrink

            print("Found new parameters: map " + str(max_map) + " with topk " + str(max_topK) + " shrink " + str(max_shrink))


print("topk " + str(max_topK))
print("shrink " + str(max_shrink))
print("Map " + str(max_map))

URM_final = URM_train + URM_test

itemCF1 = ItemBasedCollaborativeFiltering(max_topK, max_shrink)
itemCF1.fit(URM_final)
write_output(itemCF1, target_users)
