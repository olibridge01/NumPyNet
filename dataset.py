import gzip
import pickle
import numpy as np
from urllib import request

class MNIST:
    """
    Dataset class for MNIST handwritten digits.
    """

    base_url = "https://ossci-datasets.s3.amazonaws.com/mnist/"
    filenames = [
        ["training_images", "train-images-idx3-ubyte.gz"],
        ["test_images", "t10k-images-idx3-ubyte.gz"],
        ["training_labels", "train-labels-idx1-ubyte.gz"],
        ["test_labels", "t10k-labels-idx1-ubyte.gz"]
    ]

    def __init__(self, root: str = "data/", train: bool = True, download: bool = False):
        self.root = root
        self.train = train

        if download:
            self.download()
            self.save()

        self.images, self.labels = self.load()
        self.index = 0

    def download(self):
        for name in self.filenames:
            print("Downloading " + name[1] + "...")
            request.urlretrieve(self.base_url + name[1], self.root + name[1])
        print("Download complete.")

    def save(self):
        mnist = {}
        for name in self.filenames[:2]:
            with gzip.open(self.root + name[1], 'rb') as f:
                mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28 * 28) / 255
        for name in self.filenames[-2:]:
            with gzip.open(self.root + name[1], 'rb') as f:
                mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
        with open(self.root + "mnist.pkl", 'wb') as f:
            pickle.dump(mnist, f)
        print("Save complete.")

    def load(self):

        # Check that pickle file exists
        try:
            with open(self.root + "mnist.pkl", 'rb') as f:
                pass
        except FileNotFoundError:
            print("File not found. Downloading and saving...")
            self.download()
            self.save()

        with open(self.root + "mnist.pkl", 'rb') as f:
            mnist = pickle.load(f)
        if self.train:
            return mnist["training_images"], mnist["training_labels"]
        else:
            return mnist["test_images"], mnist["test_labels"]

    def __iter__(self):
        return self
    
    def __next__(self):
        if self.index == len(self.images):
            raise StopIteration
        else:
            self.index += 1
            return self.images[self.index - 1], self.labels[self.index - 1]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        return self.images[index], self.labels[index]
    
    def __repr__(self):
        return "MNIST Dataset"
    

class DataLoader:
    """
    DataLoader class for iterating over a dataset.
    """
    def __init__(self, dataset: MNIST, batch_size: int = 32, shuffle: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.reset()

    def __iter__(self):
        return self
    
    def reset(self):
        self.index = 0
        if self.shuffle:
            self.sample_ids = np.random.permutation(len(self.dataset))
        else:
            self.sample_ids = np.arange(len(self.dataset))


    def __next__(self):
        if self.index == len(self.dataset):
            self.reset()
            raise StopIteration
        else:
            batch_ids = self.sample_ids[self.index:self.index + self.batch_size]
            self.index += self.batch_size
            images, labels = self.dataset[batch_ids]
            return images, labels
        
    def __len__(self):
        return len(self.dataset) // self.batch_size + 1
    
    def __repr__(self):
        return f"DataLoader, batch size {self.batch_size}"