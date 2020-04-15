import numpy as np
import gzip
import sklearn.linear_model
import time

IMAGE_SIZE = 28

validation_image = gzip.open("C:\\Users\\rafal.000\\python\\allegro\\logistic_regression\\validate\\t10k-images-idx3-ubyte.gz", "r")
validation_labels = gzip.open("C:\\Users\\rafal.000\\python\\allegro\\logistic_regression\\validate\\t10k-labels-idx1-ubyte.gz", "r")

train_image = gzip.open("C:\\Users\\rafal.000\\python\\allegro\\logistic_regression\\train\\train-images-idx3-ubyte.gz", "r")
train_labels = gzip.open("C:\\Users\\rafal.000\\python\\allegro\\logistic_regression\\train\\train-labels-idx1-ubyte.gz", "r")

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


clf = sklearn.linear_model.SGDClassifier(loss = 'log', max_iter=1000, tol=None, n_iter_no_change = 1000, learning_rate = 'constant', eta0 = 0.8, validation_fraction = 0.01)
start = time.time()
print("zaczynam trening")
clf.fit(X, y)
print(clf.score(v_X, v_y))
print(f"training took {time.time() - start} seconds")