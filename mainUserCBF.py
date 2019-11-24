from Algorithms.Notebooks_utils.evaluation_function import evaluate_MAP_target_users
from Utils.Toolkit import TestGen, DataReader
from Utils.Toolkit import TestSplit
from CBF.user_CBF import UserContentBasedRecommender

URM_all_CSR = DataReader().URM_CSR()
target_users = DataReader().targetUsersList

lou_matrices = TestGen(URM_all_CSR, TestSplit.LEAVE_ONE_OUT)

URM_train = lou_matrices.URM_train
URM_test = lou_matrices.URM_test

topK_collection_age = [0]
topK_collection_region = [40, 400]
shrink_collection_age = [0]
shrink_collection_region = [10, 20, 100, 200, 300]
weight_coll = [1]
max_map = 0
max_topK_region = 0
max_shrink_region = 0
max_topK_age = 0
max_shrink_age = 0
max_weight = 0

for topK in topK_collection_region:
    for shrink in shrink_collection_region:

        itemCBF = UserContentBasedRecommender(1, topK, 1, shrink, 1)
        itemCBF.fit(URM_train)
        map_elem = evaluate_MAP_target_users(URM_test, itemCBF, target_users)
        print(map_elem)

        if map_elem > max_map:
            max_map = map_elem
            max_topK_age = 0
            max_shrink_age = 0
            max_shrink_region = shrink
            max_topK_region = topK

            print("Found new parameters: map " + str(max_map) + ", Weight: " + str(max_weight) + " with topk_asset " + str(max_topK_region) + " shrink_asset " + str(max_shrink_region) + " with topk_sub " + str(max_topK_age) + " shrink_asset " + str(max_shrink_age))

print("topk_region " + str(max_topK_region))
print("shrink_region " + str(max_shrink_region))
print("topk_age " + str(max_topK_age))
print("shrink_age " + str(max_shrink_age))
print("weight " + str(max_weight))
print("Map " + str(max_map))

"""
URM_final = URM_train + URM_test

itemCF1 = ItemBasedCollaborativeFiltering(max_topK, max_shrink)
itemCF1.fit(URM_final)
write_output(itemCF1, target_users)
"""
