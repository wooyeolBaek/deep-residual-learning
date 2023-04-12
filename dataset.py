import numpy as np
from tqdm import tqdm
from  torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
from sklearn.model_selection import train_test_split

class CustomCIFAR10(Dataset):
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

def preprocess(per_pixel_mean_sub=True, whitening=False, normalization=False, epsilon=0.1, include_valid=True):
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
    
    train_X, valid_X, train_Y, valid_Y = train_test_split(X, Y, test_size=0.1, random_state=2023, stratify=Y)
    
    train_X = np.array(train_X)
    train_Y = np.array(train_Y)
    valid_X = np.array(valid_X)
    valid_Y = np.array(valid_Y)

    train_batch_size, train_height, train_width, train_nchannels = train_X.shape
    valid_batch_size, valid_height, valid_width, valid_nchannels = valid_X.shape
    train_X = train_X.reshape(train_batch_size, train_height * train_width * train_nchannels)
    valid_X = valid_X.reshape(valid_batch_size, valid_height * valid_width * valid_nchannels)

    if normalization:
        train_X = train_X / 255.
        if include_valid:
            valid_X = valid_X / 255.

    if per_pixel_mean_sub:
        train_mean = train_X.mean(axis=0)
        train_X = train_X - train_mean
        if include_valid:
            valid_X = valid_X - train_mean

    if not normalization:
        train_X = train_X.astype(np.uint8)
        valid_X = valid_X.astype(np.uint8)

    if whitening:
        cov = np.cov(train_X, rowvar=False)
        U, S, V = np.linalg.svd(cov)

        train_X = U.dot(np.diag(1.0/np.sqrt(S + epsilon))).dot(U.T).dot(train_X.T).T
        train_X = (train_X - train_X.min()) / (train_X.max() - train_X.min())
        valid_X = U.dot(np.diag(1.0/np.sqrt(S + epsilon))).dot(U.T).dot(valid_X.T).T
        valid_X = (valid_X - valid_X.min()) / (valid_X.max() - valid_X.min())
    
    train_X = train_X.reshape(train_batch_size, train_height, train_width, train_nchannels)
    valid_X = valid_X.reshape(valid_batch_size, valid_height, valid_width, valid_nchannels)

    print("train_X.shape:", train_X.shape)
    print("valid_X.shape:", valid_X.shape)
    print("Finish Preprocessing...")

    return train_X, valid_X, train_Y, valid_Y


