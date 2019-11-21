from Algorithms.Notebooks_utils.data_splitter import train_test_holdout
from Utils.Toolkit import TestGen, Tester, OutputFile, RecommenderGenerator, TestSplit
from Algorithms.MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_BPR_Cython
from Algorithms.Notebooks_utils.evaluation_function import evaluate_algorithm
from Algorithms.Base.Evaluation.Evaluator import EvaluatorHoldout

leave_k_out = TestGen(test=TestSplit.LEAVE_K_OUT, k=10)

URM_train, URM_validation = train_test_holdout(leave_k_out.URM_train)
evaluator_validation_early_stopping = EvaluatorHoldout(URM_validation, cutoff_list=[10], exclude_seen = False)

recommender = MatrixFactorization_BPR_Cython(leave_k_out.URM_train)
for factors in range(100,400,50):
    recommender.fit(epochs=1000,
                    num_factors=factors,
                    validation_every_n=100,
                    stop_on_validation=True,
                    evaluator_object=evaluator_validation_early_stopping,
                    lower_validations_allowed=10,
                    validation_metric="MAP")
    result = evaluate_algorithm(leave_k_out.URM_test,recommender)
    print(result)

