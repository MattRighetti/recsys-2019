from NonPersonalized.top_pop import TopPop
from Utils.Toolkit import DataReader, TestGen, TestSplit

top_pop = TopPop()
URM_all_CSR = DataReader().URM_CSR()
target_users = DataReader().targetUsersList

lou_matrices = TestGen(URM_all_CSR, TestSplit.LEAVE_ONE_OUT)

URM_train = lou_matrices.URM_train
URM_test = lou_matrices.URM_test
top_pop.fit(URM_train)
for i in range(10):
    print(top_pop.recommend(i))