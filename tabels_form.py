import pandas as pd

Student_distribution = pd.read_csv('Student_distribution.csv',header=0, index_col=0)

Chi_square = pd.read_csv('Chi_square.csv', header=0, index_col=0)

Laplas_function = pd.read_csv('Laplas_func.csv', index_col=0)
Laplas_function = dict(zip(list(Laplas_function.index), Laplas_function.values[:, 0]))
for i in Laplas_function:
    Laplas_function[i] = round(Laplas_function[i], 5)


