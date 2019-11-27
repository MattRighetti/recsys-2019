def compute_similarity_two_matrices(matrix_1, matrix_2=None, normalize=False, topK=None, shrink=None, block_size = 100):

    n_rows_1, n_columns_1 = matrix_1.shape
    n_rows_2, n_columns_2 = matrix_2.shape

    matrix_1 = matrix_1.copy()
    matrix_2 = matrix_2.copy()

    topK = min(topK, n_columns_1)

    values = []
    rows = []
    cols = []

    if matrix_2 is None:
        print("Computing similarity on matrix_1 (Similarity between elems in the row)")

    if matrix_2 is not None:
        print("Computing similarity between matrix_1 & matrix_2 (Similarity between elems in the row")

