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
max_map = 0
max_topK = 0
max_shrink = 0

for topK in topK_collection:
    for shrink in shrink_collection:
        map_collection = []

        itemCF1 = ItemBasedCollaborativeFiltering(URM_train=URM_train, topK=topK, shrink=shrink)
        itemCF1.fit()
        map_elem = evaluate_MAP_target_users(URM_test, itemCF1, target_users)
        map_collection.append(map_elem)
        print(map_elem)

        map_avg = np.average(map_collection)
        print("Average MAP " + str(map_avg))

        if map_avg > max_map:
            max_map = map_avg
            max_topK = topK
            max_shrink = shrink

            print("Found new parameters: map " + str(max_map) + " with topk " + str(max_topK) + " shrink " + str(max_shrink))


print("topk " + str(max_topK))
print("shrink " + str(max_shrink))
print("Map " + str(max_map))

URM_final = URM_train + URM_test

itemCF1 = ItemBasedCollaborativeFiltering(URM_final, max_topK, max_shrink)
itemCF1.fit()
write_output(itemCF1, target_users)
