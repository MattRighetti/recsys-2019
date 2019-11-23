from multiprocessing import Array, Process

import threading
from Utils.OutputWriter import write_output
from Utils.Toolkit import TestGen, DataReader
from Utils.Toolkit import TestSplit
from CF.item_cf import ItemBasedCollaborativeFiltering
import numpy as np

dr = DataReader()
URM_all_CSR = dr.URM_CSR()
target_users = dr.targetUsersList
lou = TestGen(URM_all_CSR, TestSplit.LEAVE_ONE_OUT)
URM_train = lou.URM_train
URM_test = lou.URM_test
topK_collection = [10, 15, 22, 35, 50, 70, 90, 120, 150]
shrink_collection = [5, 10, 20, 25, 30, 40, 50, 80, 120, 200]
max_topK = 0
max_shrink = 0
max_map = 0
test_matrix1 = URM_test.copy()
test_matrix2 = URM_test.copy()
train_matrix1 = URM_train.copy()
train_matrix2 = URM_train.copy()

for topK in topK_collection:
    for shrink in range(0, len(shrink_collection), 2):
        processes = []

        itemCF1 = ItemBasedCollaborativeFiltering(URM_train=train_matrix1, topK=topK, shrink=shrink_collection[shrink])
        itemCF2 = ItemBasedCollaborativeFiltering(URM_train=train_matrix1, topK=topK, shrink=shrink_collection[shrink + 1])


#        p0 = Process(target=itemCF1.fit)
#        p1 = Process(target=itemCF2.fit)

#        processes.append(p0)
#        processes.append(p1)

 #       for p in processes:
  #          p.start()

   #     for p in processes:
    #        p.join()

     #   print("first done")

      #  processes.clear()
        map1 = 0
        map2 = 0

        t1 = threading.Thread(target=itemCF1.wrapper, args=(train_matrix1, test_matrix1, target_users, map1))

        t2 = threading.Thread(target=itemCF2.wrapper,
                     args=(train_matrix2, test_matrix2, target_users, map2))

        t1.start()
        t2.start()
        t1.join()
        t2.join()
        map_avg = (map1 + map2) / 2


        map_avg = map_avg/2
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