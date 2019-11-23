from Algorithms.Notebooks_utils.evaluation_function import evaluate_MAP_target_users
from Utils.Toolkit import TestGen, DataReader
from Utils.Toolkit import TestSplit
from CBF.item_CBF import ItemContentBasedRecommender

URM_all_CSR = DataReader().URM_CSR()
target_users = DataReader().targetUsersList

lou_matrices = TestGen(URM_all_CSR, TestSplit.LEAVE_ONE_OUT)

URM_train = lou_matrices.URM_train
URM_test = lou_matrices.URM_test

topK_collection_asset = [1]
topK_collection_subclass = [40, 400]
shrink_collection_asset = [1]
shrink_collection_sub = [10, 20, 100, 200, 300]
weight_coll = [1]
max_map = 0
max_topK_sub = 0
max_shrink_sub = 0
max_topK_asset = 0
max_shrink_asset = 0
max_weight = 0

for topK in topK_collection_asset:
    for shrink in shrink_collection_sub:
        for topK_sub in topK_collection_subclass:
            for weight in weight_coll:

                itemCBF = ItemContentBasedRecommender(topK, topK_sub, shrink, shrink, weight)
                itemCBF.fit(URM_train)
                map_elem = evaluate_MAP_target_users(URM_test, itemCBF, target_users)
                print(map_elem)

                if map_elem > max_map:
                    max_map = map_elem
                    max_topK_asset = topK
                    max_shrink_asset = shrink
                    max_shrink_sub = shrink
                    max_topK_sub = topK_sub
                    max_weight = weight

                    print("Found new parameters: map " + str(max_map) + ", Weight: " + str(max_weight) + " with topk_asset " + str(max_topK_asset) + " shrink_asset " + str(max_shrink_asset) + " with topk_sub " + str(max_topK_sub) + " shrink_asset " + str(max_shrink_sub))

print("topk_sub " + str(max_topK_sub))
print("shrink_sub " + str(max_shrink_sub))
print("topk_asset " + str(max_topK_asset))
print("shrink_asset " + str(max_shrink_asset))
print("weight " + str(max_weight))
print("Map " + str(max_map))

"""
URM_final = URM_train + URM_test

itemCF1 = ItemBasedCollaborativeFiltering(max_topK, max_shrink)
itemCF1.fit(URM_final)
write_output(itemCF1, target_users)
"""
