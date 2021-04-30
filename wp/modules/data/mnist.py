import gzip
import os
import numpy as np

def load(path):
    """
        Load manualle downloaded mnist dataset from "http://yann.lecun.com/exdb/mnist/" into memory.

        Args:
            path (str) The path to the main directory where separate files lie
        
        Returns:
            tuple(images, labels) as np.ndarrays
    """

    # Get the ubit zip files
    gzip_files = os.listdir(path)

    image_size = 28
    num_images_train = 5
    num_images_test = 5

    # Load dataset into lists
    images = []
    labels = []
    for zipped_file in gzip_files:
        mnist_file = os.path.join(path, zipped_file)
        gzip_file = gzip.open(mnist_file, 'r')
        
        magic_num = int.from_bytes(gzip_file.read(4), "big")
        num_images = int.from_bytes(gzip_file.read(4), "big")
        
        if "labels" in zipped_file:
            buf = gzip_file.read()
            data = np.frombuffer(buf, dtype=np.uint8)
            labels.append(data)
        else:
            row_count = int.from_bytes(gzip_file.read(4), "big")
            column_count = int.from_bytes(gzip_file.read(4), "big")
            buf = gzip_file.read()
            data = np.frombuffer(buf, dtype=np.uint8).reshape((num_images, row_count, column_count))
            images.append(data)

    return np.vstack(images), np.hstack(labels)