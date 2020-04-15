import gzip

import numpy as np
import time

from sklearn.linear_model import SGDClassifier

IMAGE_SIZE = 28

validation_image = gzip.open("C:\\Users\\rafal\\python\\t10k-images-idx3-ubyte.gz", "r")
validation_labels = gzip.open("C:\\Users\\rafal\\python\\t10k-labels-idx1-ubyte.gz", "r")

train_image = gzip.open("C:\\Users\\rafal\\python\\train-images-idx3-ubyte.gz", "r")
train_labels = gzip.open("C:\\Users\\rafal\\python\\train-labels-idx1-ubyte.gz", "r")

validation_image.read(16)
validation_labels.read(8)
train_image.read(16)
train_labels.read(8)

def is_prime(x):
    if x in [2,3,5,7]:
        return 1
    elif x in [4,6,8]:
        return 0
    return -1

is_prime = np.vectorize(is_prime)

def generate_matrices(buffer, length):
    M = np.frombuffer(buffer, dtype=np.uint8)
    M = M.reshape(int(M.shape[0]/(length**2)), length**2)

    return M

def generate_map(labels):
    map = np.argwhere(labels == -1)
    return map.T[0]


X = generate_matrices(train_image.read(), IMAGE_SIZE)
y = is_prime(generate_matrices(train_labels.read(), 1))
map = generate_map(y)

X = np.delete(X, map, 0)
y = np.delete(y, map, 0)



v_X = generate_matrices(validation_image.read(), IMAGE_SIZE)
v_y = is_prime(generate_matrices(validation_labels.read(), 1))
map = generate_map(v_y)

v_X = np.delete(v_X, map, 0)
v_y = np.delete(v_y, map, 0)

clf = SGDClassifier(fit_intercept=True)
clf.fit(X,y)


print (clf.intercept_)
print(clf.coef_)
print ('Accuracy from sk-learn: {0}'.format(clf.score(v_X,v_y)))
# class Model:
#     image_size = 28
#     num_example = 60000
#     step_size = 0.1
#
#     def __init__(self):
#         image_file = gzip.open("train/train-images-idx3-ubyte.gz", "r")
#         label_file = gzip.open("train/train-labels-idx1-ubyte.gz", "r")
#
#         image_file.read(16)
#         label_file.read(8)
#
#         image_buffer = image_file.read()
#         self.X = np.frombuffer(image_buffer, dtype=np.uint8)
#         print(self.X.shape)
#         self.X = self.X.reshape(60000, self.image_size * self.image_size)
#
#         label_buffer = label_file.read(self.num_example)
#         self.Y = np.frombuffer(label_buffer, dtype=np.uint8)
#         self.Y = self.Y.reshape(self.num_example, 1)
#         self.Y = self.parity(self.Y)
#
#         print(self.X)
#
#         self.theta = np.zeros((self.image_size * self.image_size, 1))
#
#     def parity(self, x):
#         return x % 2
#
#     def sigma(self, z):
#         return 1 / (1 + np.exp(-z))
#
#     def fit(self) -> None:
#         start = time.time()
#         iteration_start = start
#         for i in range(self.num_example):
#
#             for j in range(self.image_size ** 2):
#                 self.theta[j][0] = self.theta[j][0] + self.step_size * (
#                             self.Y[i] - self.sigma(np.matmul(self.theta.T, self.X[i]))) * self.X[i][j]
#             if i % 1000 == 0:
#                 print(f"{i/1000} iteracja zakonczona po {time.time() - iteration_start}")
#                 iteration_start = time.time()
#
#
#
#         print(f"trening zakonczony po {time.time() - start}")
#         print(self.theta)
#
#
#     def predict(self, X: np.ndarray) -> np.ndarray:
#         pass
#
#
#     def mypredict(self, i):
#         print(f"predicted = {self.sigma(np.matmul(self.theta.T, self.X[i]))}, correct = {self.Y[i]}")
#         return self.sigma(np.matmul(self.theta.T, self.X[i]))
#
#     # @staticmethod
#     def evaluate(self) -> float:
#         image_file = gzip.open("validate/t10k-images-idx3-ubyte.gz", "r")
#         label_file = gzip.open("validate/t10k-labels-idx1-ubyte.gz", "r")
#
#         image_file.read(16)
#         label_file.read(8)
#
#         image_buffer = image_file.read(self.num_example * self.image_size * self.image_size)
#
#         X = np.frombuffer(image_buffer, dtype=np.uint8)
#         X = X.reshape(self.num_example, self.image_size * self.image_size)
#
#         label_buffer = label_file.read(self.num_example)
#         Y = np.frombuffer(label_buffer, dtype=np.uint8)
#         Y =Y.reshape(self.num_example, 1)
#         Y = self.parity(Y)
#
#         correct = 0.0
#         total = 0.0
#         for i in range(self.num_example):
#             if self.sigma(np.matmul(self.theta.T, X[i])) == Y[i] :
#                 correct += 1
#             total += 1
#         return total/correct
#
#
#
#
#     # def show_image(matrix):
#     # 	image = matrix[0].reshape(image_size, image_size)
#     # 	print(image.shape)
#     # 	plt.imshow(image)
#     # 	plt.show()
#     #
#
#     #
#
#     #
#     # print(theta)
#     # print()
#     #
#     # print(theta)
