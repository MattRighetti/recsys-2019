from Utils.Toolkit import TestGen, DataReader
from Utils.Toolkit import TestSplit


dr = DataReader()
URM_all_CSR = dr.URM_CSR()
target_users = dr.targetUsersList
k_fold = TestGen(URM_all_CSR, TestSplit.LEAVE_ONE_OUT, k=3)