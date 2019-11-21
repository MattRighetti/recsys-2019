from Utils.Toolkit import TestGen, Tester, OutputFile, RecommenderGenerator, TestSplit
from CF.item_cf import ItemBasedCollaborativeFiltering
from Algorithms.SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from Algorithms.ParameterTuning.SearchBayesianSkopt import SearchBayesianSkopt
from Algorithms.Base.Evaluation.Evaluator import EvaluatorHoldout
from Algorithms.ParameterTuning.run_parameter_search import runParameterSearch_Collaborative
from Algorithms.Notebooks_utils.data_splitter import train_test_holdout
from Algorithms.Notebooks_utils.evaluation_function import evaluate_algorithm

leave_k_out = TestGen(test=TestSplit.LEAVE_K_OUT, k=10)
leave_k_out.URM