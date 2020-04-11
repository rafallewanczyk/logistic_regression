import numpy as np
import matplotlib.pyplot as plt
import gzip

image_file = gzip.open("train/train-images-idx3-ubyte.gz", "r")
label_file = gzip.open("train/train-labels-idx1-ubyte.gz", "r")

image_file.read(16)
label_file.read(8)

image_size = 28
image_buffer = image_file.read(image_size * image_size)
image_data = np.frombuffer(image_buffer, dtype=np.uint8).astype(np.float32)
image_data = image_data.reshape(image_size, image_size)

image = np.asarray(image_data).squeeze()
print(image.shape)
plt.imshow(image)
plt.show()

label_buffer = label_file.read(1)
label_data = np.frombuffer(label_buffer, dtype=np.uint8)
print(label_data)

