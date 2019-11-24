from lightfm import LightFM
from lightfm.evaluation import precision_at_k, auc_score

from Utils.Toolkit import DataReader, TestGen, TestSplit

dr = DataReader()
URM_all_CSR = dr.URM_CSR()
target_users = dr.targetUsersList
lou = TestGen(URM_all_CSR, TestSplit.LEAVE_ONE_OUT)
URM_train = lou.URM_train
URM_test = lou.URM_test

model = LightFM(loss='warp')
model.fit(URM_train.tocsr(), epochs=50)

train_precision = precision_at_k(model, URM_train, k=10, num_threads=8).mean()
test_precision = precision_at_k(model, URM_test, k=10, num_threads=8).mean()

train_auc = auc_score(model, URM_train).mean()
test_auc = auc_score(model, URM_test).mean()

print('Precision: train %.2f, test %.2f.' % (train_precision, test_precision))
print('AUC: train %.2f, test %.2f.' % (train_auc, test_auc))

