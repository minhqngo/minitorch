import os
import gzip
import shutil
import urllib.request
import numpy as np


class MNISTDataset:
    @staticmethod
    def load_mnist_img(path):
        try:
            with open(path, "rb") as fi:
                _ = int.from_bytes(fi.read(4), "big")  # magic number
                n_images = int.from_bytes(fi.read(4), "big")
                h = int.from_bytes(fi.read(4), "big")
                w = int.from_bytes(fi.read(4), "big")
                buffer = fi.read()
                images = np.frombuffer(buffer, dtype=np.uint8).reshape(n_images, h, w)
        except Exception as e:
            print(f"Could not read MNIST image file at {path}")
            print(e)
            exit(1)
        return images
    
    @staticmethod
    def load_mnist_lbl(path):
        try:
            with open(path, "rb") as fi:
                _ = int.from_bytes(fi.read(4), "big")
                n_labels = int.from_bytes(fi.read(4), "big")
                buffer = fi.read()
                labels = np.frombuffer(buffer, dtype=np.uint8)
        except Exception as e:
            print(f"Could not read MNIST label file at {path}")
            print(e)
            exit(1) 
        return labels
    
    @staticmethod
    def _download_and_extract(root):
        """
        Downloads and extracts the MNIST dataset files if they don't exist.
        """
        mnist_path = os.path.join(root, "MNIST")
        os.makedirs(mnist_path, exist_ok=True)
        
        urls = [
            "https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-images-idx3-ubyte.gz",
            "https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-labels-idx1-ubyte.gz",
            "https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-images-idx3-ubyte.gz",
            "https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-labels-idx1-ubyte.gz",
        ]

        for url in urls:
            filename = url.split("/")[-1]
            gz_path = os.path.join(mnist_path, filename)
            uncompressed_path = os.path.join(mnist_path, filename[:-3])

            if not os.path.exists(uncompressed_path):
                print(f"Downloading {url}")
                urllib.request.urlretrieve(url, gz_path)

                print(f"Extracting {gz_path}")
                with gzip.open(gz_path, 'rb') as f_in:
                    with open(uncompressed_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                os.remove(gz_path)
    
    '''
    dataset_dir
    ├── MNIST
        ├── train-images.idx3-ubyte (train images file)
        ├── train-labels.idx1-ubyte
        ├── t10k-images.idx3-ubyte (val images file)
        ├── t10k-labels.idx1-ubyte
    '''
    
    def __init__(self, root, download=True, train=True):
        if download and not os.path.exists(os.path.join(root, "MNIST")):
            self._download_and_extract(root)
        
        if train:
            img_dir = os.path.join(root, "MNIST", "train-images-idx3-ubyte")
            lbl_dir = os.path.join(root, "MNIST", "train-labels-idx1-ubyte")
        else:
            img_dir = os.path.join(root, "MNIST", "t10k-images-idx3-ubyte")
            lbl_dir = os.path.join(root, "MNIST", "t10k-labels-idx1-ubyte")
        
        images = self.load_mnist_img(img_dir)
        labels = self.load_mnist_lbl(lbl_dir)
        
        self.data = [(image, label) for image, label in zip(images, labels)]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]
