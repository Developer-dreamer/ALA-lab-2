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

print("--------------------\n")

pca = PCA(n_components=num_components)
image_transformed = pca.fit_transform(image_bw)
image_inverted = pca.inverse_transform(image_transformed)

print(f"{image_transformed.shape}\n")
print(f"{image_inverted.shape}\n")

plt.imshow(image_inverted, cmap="gray")
plt.show()

# --------------------

for i, n in enumerate([5, 15, 25, 75, 100, 170]):
    pca = PCA(n_components=n)
    image_pca = pca.fit_transform(image_bw)
    image_rec = pca.inverse_transform(image_pca)
    plt.imshow(image_rec, cmap='gray')
    plt.title(f'{n} components reconstruction')
    plt.axis('off')
    plt.show()


def encrypt_message(message, key_matrix):
    message_vector = np.array([ord(char) for char in message], dtype=float)
    eigenvalues, eigenvectors = np.linalg.eig(key_matrix)
    diagonalized_key_matrix = np.dot(np.dot(eigenvectors, np.diag(eigenvalues)), np.linalg.inv(eigenvectors))
    encrypted_vector = np.dot(diagonalized_key_matrix, message_vector)
    return encrypted_vector


def decrypt_message(encrypted_message, key_matrix):
    eigenvalues, eigenvectors = np.linalg.eig(key_matrix)
    diagonalized_key_matrix = np.dot(np.dot(eigenvectors, np.diag(eigenvalues)), np.linalg.inv(eigenvectors))
    decrypted = np.dot(np.linalg.inv(diagonalized_key_matrix), encrypted_message)
    if np.all(np.isclose(decrypted.imag, 0, atol=1e-10)):
        decrypted_real = decrypted.real
    else:
        raise ValueError("Decrypted message contains significant imaginary components.")

    return ''.join([chr(int(np.round(i))) for i in decrypted_real])


print("--------------------\n")

message = "Hello, world!"
key_matrix = np.random.randint(0, 256, (len(message), len(message)))
enc_message = encrypt_message(message, key_matrix)
dec_message = decrypt_message(enc_message, key_matrix)

print(f"Original message: {message}")
print(f"Encrypted message: {enc_message}")
print(f"Decrypted message: {dec_message}")
