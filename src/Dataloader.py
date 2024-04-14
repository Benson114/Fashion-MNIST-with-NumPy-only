import gzip
import numpy as np


class FashionMNISTDataloader:
    def __init__(self, path_dir="fashion-mnist", n_valid=1000, batch_size=32):
        X = self.load_mnist_images(f"{path_dir}/train-images-idx3-ubyte.gz")
        y = self.load_mnist_labels(f"{path_dir}/train-labels-idx1-ubyte.gz")
        self.x_train, self.y_train, self.x_valid, self.y_valid = self.train_valid_split(X, y, n_valid)
        self.x_test = self.load_mnist_images(f"{path_dir}/t10k-images-idx3-ubyte.gz")
        self.y_test = self.load_mnist_labels(f"{path_dir}/t10k-labels-idx1-ubyte.gz")
        self.batch_size = batch_size

    @staticmethod
    def load_mnist_images(filename):
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
            data = data.reshape(-1, 784)
            data = data.astype(np.float32) / 255.0
        return data

    @staticmethod
    def load_mnist_labels(filename):
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
            data = np.eye(10)[data]
        return data

    @staticmethod
    def train_valid_split(x_train, y_train, n_valid):
        n_samples = x_train.shape[0]
        indices = np.random.permutation(n_samples)
        valid_indices = indices[:n_valid]
        train_indices = indices[n_valid:]
        return x_train[train_indices], y_train[train_indices], x_train[valid_indices], y_train[valid_indices]

    def generate_train_batch(self):
        n_samples = self.x_train.shape[0]
        indices = np.random.permutation(self.x_train.shape[0])
        for i in range(0, n_samples, self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            yield self.x_train[batch_indices], self.y_train[batch_indices]

    def generate_valid_batch(self):
        n_samples = self.x_valid.shape[0]
        indices = np.arange(n_samples)
        for i in range(0, n_samples, self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            yield self.x_valid[batch_indices], self.y_valid[batch_indices]

    def generate_test_batch(self):
        n_samples = self.x_test.shape[0]
        indices = np.arange(n_samples)
        for i in range(0, n_samples, self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            yield self.x_test[batch_indices], self.y_test[batch_indices]
