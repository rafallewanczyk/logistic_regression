#!/usr/bin/env python
# coding: utf-8

# Przygotowanie danych 

# In[13]:


import gzip
import numpy as np
import time
import warnings
warnings.filterwarnings("error")


# In[14]:


IMAGE_SIZE = 28

validation_image = gzip.open(".\\validate\\t10k-images-idx3-ubyte.gz", "r")
validation_labels = gzip.open(".\\validate\\t10k-labels-idx1-ubyte.gz", "r")

train_image = gzip.open(".\\train\\train-images-idx3-ubyte.gz", "r")
train_labels = gzip.open(".\\train\\train-labels-idx1-ubyte.gz", "r")

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
    M = M.reshape(int(M.shape[0]/(length**2)), -1)

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

print(X.shape)
print(y.shape)


# Model

# In[21]:


theta = np.ndarray((IMAGE_SIZE **2, 1))
step_size = 0.8
epochs = 30
momentum = 0.9
reg = 0.01

np.random.seed(1337)

def sigma(z):
    return 1 / (1 + np.exp(-z))

def fit(X, y):
    theta = np.zeros((IMAGE_SIZE **2, 1))
    V = np.zeros((IMAGE_SIZE **2, 1))

    start = time.time()
    iteration_start = start
    permutation = np.arange(X.shape[0])
    for epoch in range(1, epochs+1):
        np.random.shuffle(permutation)
        for i in permutation:
            try:
                gradient = (y[i] - sigma(np.matmul(theta.T, X[i])))*X[i].reshape(IMAGE_SIZE**2, 1) #+ 2*reg*theta
                gradient = gradient + 2*reg*theta
                V = momentum * V + (1-momentum) * gradient
                theta = theta + step_size*V
            except RuntimeWarning :
                continue

       
        print(f"{epoch}/{epochs} iteracja zakonczona po {time.time() - iteration_start}")
        iteration_start = time.time()
    print(f"trening zakonczony po {time.time() - start}")
    return theta

def predict(theta, X):
    return sigma(theta.T @ X)

def evaluate(theta, X, y):
    correct = 0.0
    total = 0.0 
    
    for i in range(X.shape[0]):
        if np.rint(predict(theta, X[i])) == y[i] : 
            correct += 1 
        total += 1 
    return correct/total


# In[ ]:


theta = fit(X, y)


# In[20]:


evaluate(theta, v_X, v_y)


# In[ ]:





# In[ ]:





# In[ ]:




