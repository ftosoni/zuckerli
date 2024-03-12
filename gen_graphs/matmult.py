import pandas as pd

def multiply_matrices(matrix1, matrix2):
    df1 = pd.DataFrame(matrix1)
    df2 = pd.DataFrame(matrix2)
    print(df1.shape)
    print(df2.shape)
    result = df1.dot(df2)
    return result.values.tolist()

# Esempio di utilizzo
pre_matrix1 = [
    [   0,     1, 2, 4,      5, 7,     10, 11, 12],
    [1, 2, 3, 4, 8, 9, 10, 11, 12, 13], #res.: 3 8 9 13 -> 3 8 9 3 -> 23; copied: 1 2 4 10 11 12
    [1, 2, 3, 4, 5, 6, 7, 8, 13], #res.: 5 6 7 -> 18 
    [1, 2, 3, 4, 5, 6, 7, 8],
    [1, 2, 3, 6, 7, 8],
    [1],
    [1],
    [1],
    [1],
    [1],
    [1],
    [1],
    [1],
    [1],
]

matrix1 = [[0 for _ in range(14)] for __ in range(14)]
for r,adjl in enumerate(pre_matrix1) :
    for c in adjl :
        matrix1[r][c] = 1 

matrix2 = [[x%10] for x in range(14)]

result_matrix = multiply_matrices(matrix1, matrix2)
print(result_matrix)