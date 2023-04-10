from  torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.datasets import CIFAR10
from tqdm import tqdm
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

def SVD():
    train_dataset = CIFAR10(
        root="./data/CIFAR10/",
        train=True,
        download=True
    )
    # 단계 1&2: 이미지 불러오기
    X = []
    for i in tqdm(range(len(train_dataset))):
        X.append(np.array(train_dataset[i][0]))

    X = np.array(X)

    batch_size, height, width, nchannels = X.shape
    X = X.reshape(batch_size, height * width * nchannels)

    X_norm = X / 255.

    X_norm = X_norm - X_norm.mean(axis=0)

    cov = np.cov(X_norm, rowvar=False)

    U,S,V = np.linalg.svd(cov)

    return U, S, V

class WhitenedCIFAR10(Dataset):
    def __init__(self, U, S, epsilon=0.1, train=True, transform=None):
        self.X, self.Y = self.preprocess(U=U, S=S, epsilon=epsilon, train=train)
        self.transform = transform

    def __getitem__(self, idx):
        #assert self.transform is not None, "transform is None"
        #image = Image.fromarray(self.X[idx])
        image = self.X[idx]
        label = self.Y[idx]

        if self.transform:
            image = self.transform(image=image)['image']

        return image, label
    
    def preprocess(self, U, S, epsilon=0.1, train=True):
        
        print("Start Preprocessing...")
        dataset = CIFAR10(
            root="./data/CIFAR10/",
            train=train,
            download=True
        )
        X = []
        Y = []
        for i in tqdm(range(len(dataset))):
            X.append(np.array(dataset[i][0]))
            Y.append(dataset[i][1])
        
        X = np.array(X)
        Y = np.array(Y)

        batch_size, height, width, nchannels = X.shape
        X = X.reshape(batch_size, height * width * nchannels)

        X = X / 255.
        X = X - X.mean(axis=0)

        X = U.dot(np.diag(1.0/np.sqrt(S + epsilon))).dot(U.T).dot(X.T).T
        X = (X - X.min()) / (X.max() - X.min())

        #X = X.astype(np.uint8)
        
        X = X.reshape(batch_size, height, width, nchannels)

        print("X.shape:", X.shape)
        print("len(X):", len(X))
        print("Finish Preprocessing...")

        return X, Y

    def __len__(self):
        return len(self.X)

class PerPixelSubCIFAR10(Dataset):
    def __init__(self, train=True, transform=None, **kwargs):
        self.X, self.Y = self.preprocess(train=train)
        self.transform = transform

    def __getitem__(self, idx):
        #assert self.transform is not None, "transform is None"
        image = self.X[idx]
        label = self.Y[idx]

        if self.transform:
            image = self.transform(image=image)['image']

        return image, label
    
    def preprocess(self, train=True):
        print("Start Preprocessing...")
        train_dataset = CIFAR10(
            root="./data/CIFAR10/",
            train=True,
            download=True
        )
        train_X = []
        train_Y = []
        for i in tqdm(range(len(train_dataset))):
            train_X.append(np.array(train_dataset[i][0]))
            train_Y.append(train_dataset[i][1])
        
        train_X = np.array(train_X)
        train_Y = np.array(train_Y)

        batch_size, height, width, nchannels = train_X.shape
        train_X = train_X.reshape(batch_size, height * width * nchannels)

        if train:
            X = train_X
            Y = train_Y
        else:
            dataset = CIFAR10(
                root="./data/CIFAR10/",
                train=False,
                download=True
            )
            X = []
            Y = []
            for i in tqdm(range(len(dataset))):
                X.append(np.array(dataset[i][0]))
                Y.append(dataset[i][1])
            
            X = np.array(X)
            Y = np.array(Y)

            batch_size, height, width, nchannels = X.shape
            X = X.reshape(batch_size, height * width * nchannels)


        X = X - train_X.mean(axis=0)

        X = X.astype(np.uint8)
        
        X = X.reshape(batch_size, height, width, nchannels)

        print("X.shape:", X.shape)
        print("len(X):", len(X))
        print("Finish Preprocessing...")

        return X, Y

    def __len__(self):
        return len(self.X)

class PerPixelSubCIFAR10Split(Dataset):
    def __init__(self, X, Y, transform=None, **kwargs):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, idx):
        #assert self.transform is not None, "transform is None"
        image = self.X[idx]
        label = self.Y[idx]

        if self.transform:
            image = self.transform(image=image)['image']

        return image, label

    def __len__(self):
        return len(self.X)

def preprocess():
    print("Start Preprocessing...")
    train_dataset = CIFAR10(
        root="./data/CIFAR10/",
        train=True,
        download=True
    )
    X = []
    Y = []
    for i in tqdm(range(len(train_dataset))):
        X.append(np.array(train_dataset[i][0]))
        Y.append(train_dataset[i][1])
    
    train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.1, random_state=2023, stratify=Y)
    
    train_X = np.array(train_X)
    train_Y = np.array(train_Y)
    test_X = np.array(test_X)
    test_Y = np.array(test_Y)

    train_batch_size, train_height, train_width, train_nchannels = train_X.shape
    test_batch_size, test_height, test_width, test_nchannels = test_X.shape
    train_X = train_X.reshape(train_batch_size, train_height * train_width * train_nchannels)
    test_X = test_X.reshape(test_batch_size, test_height * test_width * test_nchannels)

    train_mean = train_X.mean(axis=0)
    train_X = train_X - train_mean
    test_X = test_X - train_mean

    train_X = train_X.astype(np.uint8)
    test_X = test_X.astype(np.uint8)
    
    train_X = train_X.reshape(train_batch_size, train_height, train_width, train_nchannels)
    test_X = test_X.reshape(test_batch_size, test_height, test_width, test_nchannels)

    print("train_X.shape:", train_X.shape)
    print("len(train_X):", len(train_X))
    print("test_X.shape:", test_X.shape)
    print("len(test_X):", len(test_X))
    print("Finish Preprocessing...")

    return train_X, test_X, train_Y, test_Y


