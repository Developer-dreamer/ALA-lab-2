import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from sklearn.decomposition import PCA


def find_eigen(matrix):
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Matrix must be square")

    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    mul_by_A = matrix@eigenvectors
    mul_by_values = eigenvalues*eigenvectors

    print(f"A multiplied:\n {mul_by_A}\n")
    print(f"Eigenvalues multiplied:\n {mul_by_values}\n")

    return eigenvalues, eigenvectors


A = np.array([
    [3, 2, 2],
    [1, 3, 1],
    [1, 4, 5]
])

find_eigen(A)
