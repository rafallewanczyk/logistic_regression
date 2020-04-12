import numpy as np
import matplotlib.pyplot as plt
import gzip

def show_image(matrix):
	image = matrix[0].reshape(image_size, image_size)
	print(image.shape)
	plt.imshow(image)
	plt.show()

def sigma(z):
	return 1/(1+np.exp(-z))

def parity(x):
	return x%2


image_file = gzip.open("train/train-images-idx3-ubyte.gz", "r")
label_file = gzip.open("train/train-labels-idx1-ubyte.gz", "r")

image_file.read(16)
label_file.read(8)

image_size = 4#28
num_example = 4#60000
step_size = 0.1

image_buffer = image_file.read(num_example * image_size * image_size)
X = np.frombuffer(image_buffer, dtype=np.uint8)
X = X.reshape(num_example, image_size * image_size)

label_buffer = label_file.read(num_example)
Y = np.frombuffer(label_buffer, dtype=np.uint8)
Y = Y.reshape(num_example, 1)
Y = parity(Y)

theta = np.zeros((image_size * image_size, 1))
print(theta)
print()

for i in range(num_example):
	for j in range(image_size**2):
		theta[j][0] = theta[j][0] + step_size *(Y[i] - sigma(np.matmul(theta.T,X[i]))) * X[i][j]

print(theta)