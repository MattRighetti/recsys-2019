import pandas as pd
import numpy as np
import scipy.sparse as sps
from Algorithms.Notebooks_utils.data_splitter import train_test_holdout
from CF.user_cf import UserBasedCollaborativeFiltering
from CF.item_cf import ItemBasedCollaborativeFiltering
from Algorithms.Notebooks_utils.evaluation_function import evaluate_algorithm

filePath = "./data/data_train.csv"
df = pd.read_csv(filePath)

ratingList = list(df['data'])
userList = list(df['row'])
itemList = list(df['col'])

URM_all = sps.coo_matrix((ratingList, (userList, itemList)))
URM_all = URM_all.tocsr()


warm_items_mask = np.ediff1d(URM_all.tocsc().indptr) > 0
warm_items = np.arange(URM_all.shape[1])[warm_items_mask]

URM_all = URM_all[:, warm_items]

warm_users_mask = np.ediff1d(URM_all.tocsr().indptr) > 0
warm_users = np.arange(URM_all.shape[0])[warm_users_mask]

URM_all = URM_all[warm_users, :]

URM_train, URM_test = train_test_holdout(URM_all, train_perc = 0.7)

x_tick = [10, 50, 100, 200, 500]
MAP_per_k = []

for topK in x_tick:
    recommender = UserBasedCollaborativeFiltering(URM_train, topK=topK, shrink=20)
    recommender.fit()

    result_dict = evaluate_algorithm(URM_test, recommender)
    MAP_per_k.append(result_dict["MAP"])

x_tick = [0, 10, 50, 100, 200, 500]
MAP_per_shrinkage = []

for shrinkage in x_tick:
    recommender = UserBasedCollaborativeFiltering(URM_train, topK=10, shrink=shrinkage)
    recommender.fit()

    result_dict = evaluate_algorithm(URM_test, recommender)
    MAP_per_k.append(result_dict["MAP"])

################################################ WRITE TO FILE ################################################

counter = 0

file = open("./output/file_out.csv", "w")

file.write("user_id,item_list\n")

for user_array in array:
    file.write(f'{counter},')
    for i in range(0,10):
        if i != 9:
            file.write(f'{user_array[i]} ')
        else:
            file.write(f'{user_array[i]}')
    file.write("\n")
    counter += 1