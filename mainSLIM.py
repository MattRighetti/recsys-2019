from Algorithms.Notebooks_utils.evaluation_function import evaluate_MAP_target_users
from Utils.OutputWriter import write_output
from Utils.Toolkit import TestGen, TestSplit, DataReader
from Algorithms.SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from Algorithms.Base.Evaluation.Evaluator import EvaluatorHoldout
from Algorithms.Base.NonPersonalizedRecommender import TopPop
from Algorithms.Notebooks_utils.data_splitter import train_test_holdout

URM_all_CSR = DataReader().URM_CSR()
target_users = DataReader().targetUsersList

lou_matrices = TestGen(URM_all_CSR, TestSplit.LEAVE_ONE_OUT)

URM_train = lou_matrices.URM_train
URM_test = lou_matrices.URM_test

URM_train, URM_validation = train_test_holdout(URM_train, 0.85)

evaluator_validation_early_stopping = EvaluatorHoldout(URM_validation, cutoff_list=[10], exclude_seen=True)

recommender = None
MAPS = []
topKS = []
max_topk = 0
max_lambda_v = 0
max_MAP = 0

for lambda_v in range(10, 100, 2):
    for topK in range(0,300, 10):
        recommender = SLIM_BPR_Cython(URM_train, verbose=False)
        recommender.fit(topK=topK,
                        lambda_i=(lambda_v/100),
                        lambda_j=(lambda_v/100),
                        epochs = 100,
                        validation_every_n = 20,
                        stop_on_validation = True,
                        evaluator_object = evaluator_validation_early_stopping,
                        lower_validations_allowed = 10,
                        validation_metric = "MAP"
        )
        result = evaluate_MAP_target_users(URM_test, recommender, target_users)
        print(result)
        if result > max_MAP:
            max_MAP = result
            max_topk = topK
            max_lambda_v = lambda_v
            MAPS.append(max_MAP)
            topKS.append(max_topk)
            print("Found new model with MAP " + str(max_MAP) + " topk " + str(max_topk) + " lambda_v " + str(max_lambda_v))
            
URM_final = URM_train + URM_test
recommender = SLIM_BPR_Cython(URM_final, verbose=False)
recommender.fit(topK=max_topk,
                lambda_i=(max_lambda_v/100),
                lambda_j=(max_lambda_v/100),
                epochs = 1000,
                validation_every_n = 50,
                stop_on_validation = True,
                evaluator_object = evaluator_validation_early_stopping,
                lower_validations_allowed = 10,
                validation_metric = "MAP"
                )
write_output(recommender, target_users)




