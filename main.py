from Algorithms.Notebooks_utils.evaluation_function import evaluate_MAP_target_users
from Utils.Toolkit import TestGen, DataReader
from Utils.Toolkit import TestSplit
from CF.item_cf import ItemBasedCollaborativeFiltering

import numpy as np

k_fold = TestGen(TestSplit.K_FOLD, k=3)
matrices = k_fold.get_k_fold_matrices()
target_users = k_fold.get_targetList()
topK_collection = [0,1,2,3,4,5,10,12,14,16,20,25,30,40,50,70,90,120,150,200,500]
shrink_collection = [0,1,2,3,4,5,10,12,14,16,20,25,30,40,50,70,90,120,150,200,500]
max_map = 0
max_topK = 0
max_shrink = 0

for topK in topK_collection:
    for shrink in shrink_collection:
        map_collection = []
        for i in range(len(matrices)):
            test_matrix = matrices[i]
            train_matrix = None

            for j in range(len(matrices)):
                if j != i:
                    if train_matrix is not None:
                        train_matrix += matrices[j]
                    else:
                        train_matrix = matrices[j]



            itemCF1 = ItemBasedCollaborativeFiltering(URM_train=train_matrix, topK=topK, shrink=shrink)
            itemCF1.fit(similarity='cosine')
            map_elem = evaluate_MAP_target_users(test_matrix, itemCF1, target_users)
            map_collection.append(map_elem)
            print(map_elem)

        map_avg = np.average(map_collection)
        print("Average MAP " + str(map_avg))

        if map_avg > max_map:
            max_map = map_avg
            max_topK = topK
            max_shrink = max_shrink

            print("Found new parameters: map " + str(max_map) + " with topk " + str(max_topK) + " shrink " + str(max_shrink))
