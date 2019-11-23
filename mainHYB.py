from Algorithms.Notebooks_utils.evaluation_function import evaluate_MAP_target_users
from HYB.user_wise_hybrid import UserWiseHybridRecommender
from Utils.OutputWriter import write_output
from Utils.Toolkit import TestGen, DataReader
from Utils.Toolkit import TestSplit

dr = DataReader()
URM_all_CSR = dr.URM_CSR()
target_users = dr.targetUsersList
lou = TestGen(URM_all_CSR, TestSplit.LEAVE_ONE_OUT)
URM_train = lou.URM_train
URM_test = lou.URM_test

hyb = UserWiseHybridRecommender()
hyb.fit(URM_train)
hyb.evaluate_MAP_target(URM_test, target_users)

URM_final = URM_train+URM_test

hyb.fit(URM_final)
write_output(hyb, target_users)