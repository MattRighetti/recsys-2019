from Utils.Toolkit import TestGen, DataReader
from Utils.Toolkit import TestSplit
from CF.item_cf import ItemBasedCollaborativeFiltering

dataReader = DataReader()
k_fold = TestGen(dataReader.URM_CSR(), TestSplit.K_FOLD, k=3)
matrices = k_fold.get_k_fold_matrices()
target_users = dataReader.targetUsersList


for i in range(len(matrices)):
    test_matrix = matrices[i]
    train_matrix = None

    for j in range(len(matrices)):
            if j != i:
                if train_matrix is not None:
                    train_matrix += matrices[j]
                else:
                    train_matrix = matrices[j]

    itemCF1 = ItemBasedCollaborativeFiltering(train_matrix, topK=12, shrink=325)
    itemCF1.fit(similarity="tanimoto")
    itemCF1.evaluate_MAP_target(test_matrix, target_users)

