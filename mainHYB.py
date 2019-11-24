from HYB.user_wise_hybrid import UserWiseHybridRecommender
from Utils.OutputWriter import write_output
from Utils.Toolkit import TestGen, DataReader, TestSplit, get_data

data = get_data()
URM_all_CSR = data['URM_all'].tocsr()
target_users = data['target_users']
train = data['train'].tocsr()
test = data['test'].tocsr()

max_map = 0
max_recomm = None

hyb = UserWiseHybridRecommender()
for i in range(30, 35):
    for j in range(25, 29):
        hyb.set_shrink(j)
        hyb.set_topK(i)
        hyb.fit(train)
        result = hyb.evaluate_MAP_target(test, target_users)

        if result > max_map:
            max_map = result
            print(f'best TopK: {i}, Shrink: {j}')
            URM_final = URM_all_CSR
            hyb.fit(URM_final)
            max_recomm = hyb

write_output(max_recomm, target_users)