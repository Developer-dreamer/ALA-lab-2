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

print("--------------------\n")

image_raw = imread("image.jpg")
print(f"{image_raw.shape}\n")
plt.imshow(image_raw)
plt.show()

print("--------------------\n")

image_sum = image_raw.sum(axis=2)
image_bw = image_sum/image_sum.max()
print(f"{image_sum}\n")
print(f"{image_bw.max()}\n")
plt.imshow(image_bw, cmap="gray")
print(f"{image_bw.shape}\n")
plt.show()

print("--------------------\n")

pca = PCA()
pca.fit(image_bw)

cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
num_components = np.argmax(cumulative_variance >= 0.95) + 1
print(f"{num_components}\n")

plt.figure(figsize=(10, 7))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance)
plt.title('Cumulative Explained Variance')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.axhline(y=0.95, color='r', linestyle='--')
plt.axvline(x=num_components, color='g', linestyle='--')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.show()
