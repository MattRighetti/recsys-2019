from Utils.Toolkit import TestGen, Tester, OutputFile, RecommenderGenerator, TestSplit
from Algorithms.SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from Algorithms.Base.Evaluation.Evaluator import EvaluatorHoldout
from Algorithms.Base.NonPersonalizedRecommender import TopPop
from Algorithms.Notebooks_utils.data_splitter import train_test_holdout

leave_k_out = TestGen(test=TestSplit.LEAVE_K_OUT, k=3)

URM_train, URM_validation = train_test_holdout(leave_k_out.URM_train)
evaluator_validation_early_stopping = EvaluatorHoldout(URM_validation, cutoff_list=[10], exclude_seen = False)

recommender = None
MAPS = []
topKS = []
max_topk = 0
max_MAP = 0

for topK in range(0,300, 10):
    recommender = SLIM_BPR_Cython(leave_k_out.URM_train)
    recommender.fit(topK=topK,
                    epochs = 1000,
                    validation_every_n = 100,
                    stop_on_validation = True,
                    evaluator_object = evaluator_validation_early_stopping,
                    lower_validations_allowed = 10,
                    validation_metric = "MAP"
    )
    ev = EvaluatorHoldout(leave_k_out.URM_test, [10])
    result = ev.evaluateRecommender(recommender)
    print(result)
    if result[0].get(10).get('MAP') > max_MAP:
        max_MAP = result[0].get(10).get('MAP')
        max_topk = topK
        MAPS.append(max_MAP)
        topKS.append(max_topk)
        print("MAP " + str(result))

print("MAPS")
print("topk " + str(max_topk))
recommender_2 = TopPop(URM_train=leave_k_out.URM_train)
recommender_2.fit()
recommender = SLIM_BPR_Cython(leave_k_out.URM_train)
recommender.fit(1000)
ev = EvaluatorHoldout(leave_k_out.URM_test, [10])
result = ev.evaluateRecommender(recommender)
print(result)
output = OutputFile()
output.write_output(recommender, leave_k_out, recommender_2)




