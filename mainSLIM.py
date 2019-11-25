from Algorithms.Notebooks_utils.evaluation_function import evaluate_MAP_target_users
from Utils.OutputWriter import write_output
from Utils.Toolkit import TestGen, TestSplit, DataReader, get_data
from Algorithms.SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from Algorithms.Base.Evaluation.Evaluator import EvaluatorHoldout

data = get_data()
URM_all_CSR = data['URM_all']
target_users = data['target_users']

URM_train = data['train']
URM_test = data['test']

evaluator_validation_early_stopping = EvaluatorHoldout(URM_test, cutoff_list=[10], exclude_seen=True)

recommender = None
MAPS = []
topKS = []
max_topk = 0
max_lambda_v = 0
max_MAP = 0

best_values_5 = {'topK': 30, 'lambda_i': 0.06, 'lambda_j':0.001}
best_values_2 = {'topK': 30, 'lambda_i': 0.6, 'lambda_j': 0.02, 'epochs': 1000}    # MAP 0.0330
best_values_3 = {'topK': 30, 'lambda_i': 0.35, 'lambda_j': 0.03, 'epochs': 1000}  # MAO 0.0329
best_values_4 = {'topK': 30, 'lambda_i': 0.6, 'lambda_j': 0.03, 'epochs': 1000}   # MAP 0.0329
best_values_1 = {'topK': 30, 'lambda_i': 0.6, 'lambda_j': 1, 'epochs': 400}   # MAP 0.0336
best_values = {'topK': 30, 'lambda_i': 0.6, 'lambda_j': 0.02, 'epochs': 1000}

for topK in [30]:

    args = {
        'topK' : 30,
        'lambda_i' : 1,
        'lambda_j' : 0.025,
        'epochs' : 200
    }

    recommender = SLIM_BPR_Cython(URM_train, verbose=False)
    recommender.fit(topK=args['topK'],
                    lambda_i=args['lambda_i'],
                    lambda_j=args['lambda_j'],
                    epochs = args['epochs'],
                    validation_every_n = 200,
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
        MAPS.append(max_MAP)
        topKS.append(max_topk)
        print(f'best_values = {args}')
            
#URM_final = URM_train + URM_test
#recommender = SLIM_BPR_Cython(URM_final, verbose=False)
#recommender.fit(topK=max_topk,
#                 lambda_i=(max_lambda_v/100),
#                 lambda_j=(max_lambda_v/100),
#                 epochs = 1000,
#                 validation_every_n = 50,
#                 stop_on_validation = True,
#                 evaluator_object = evaluator_validation_early_stopping,
#                 lower_validations_allowed = 10,
#                 validation_metric = "MAP"
# )
# write_output(recommender, target_users)




