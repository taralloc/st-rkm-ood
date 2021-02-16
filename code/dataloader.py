import os
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

def get_dataloader(args):
    print(f'Loading data for {args.dataset_name}...')

    args_dict = vars(args)
    if "post_transformations" not in args_dict:
        args_dict["post_transformations"] = []
    if "pre_transformations" not in args_dict:
        args_dict["pre_transformations"] = []
    if "train" not in args_dict:
        args_dict["train"] = False


    if args.dataset_name == 'mnist':
        return get_mnist_dataloader(args=args)

    elif args.dataset_name == 'fashion-mnist':
        return get_fashion_mnist_dataloaders(args=args)

    elif args.dataset_name == 'svhn':
        return get_svhn_dataloader(args=args)

    elif args.dataset_name == 'dsprites':
        return get_dsprites_dataloader(args=args)

    elif args.dataset_name == 'cifar10':
        return get_cifar10_dataloader(args=args)

    elif args.dataset_name == 'ecg5000':
        return get_ecg5000_dataloader(args=args)

    elif args.dataset_name == 'ecg5000out':
        return get_ecg5000out_dataloader(args=args)

    elif args.dataset_name == 'isun':
        return get_isun_dataloader(args=args)

def get_dataloader_training(args):
    print(f'Loading data for {args.dataset_name}...')

    train_perc = 0.85
    if args.dataset_name == 'fashion-mnist':
        all_transforms = transforms.Compose([transforms.ToTensor()])
        dataset = datasets.FashionMNIST("fashion-mnist", train=True, download=True, transform=all_transforms)
        train_size = int(train_perc * len(dataset))
        val_size = len(dataset) - train_size
        splits = torch.utils.data.random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(splits[0], batch_size=args.mb_size, shuffle=args.shuffle,
                                  pin_memory=True, num_workers=args.workers)
        val_loader = DataLoader(splits[1], batch_size=args.mb_size, shuffle=args.shuffle,
                                  pin_memory=True, num_workers=args.workers)
        _, c, x, y = next(iter(train_loader))[0].size()
        return train_loader, c * x * y, c, val_loader
    elif args.dataset_name == 'cifar10':
        all_transforms = transforms.Compose([
            transforms.ToTensor()])
        dataset = datasets.CIFAR10('cifar10', train=True, download=True, transform=all_transforms)
        train_size = int(train_perc * len(dataset))
        val_size = len(dataset) - train_size
        splits = torch.utils.data.random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(splits[0], batch_size=args.mb_size, shuffle=args.shuffle,
                                  pin_memory=True, num_workers=args.workers)
        val_loader = DataLoader(splits[1], batch_size=args.mb_size, shuffle=args.shuffle,
                                  pin_memory=True, num_workers=args.workers)
        _, c, x, y = next(iter(train_loader))[0].size()
        return train_loader, c * x * y, c, val_loader



def get_mnist_dataloader(args, path_to_data='mnist'):
    """MNIST dataloader with (28, 28) images."""

    print("Loading MNIST.")
    all_transforms = transforms.Compose(args.pre_transformations + [
                                         transforms.ToTensor()] + args.post_transformations)
    train_data = datasets.MNIST(path_to_data, train=args.train, download=True, transform=all_transforms)
    train_loader = DataLoader(train_data, batch_size=args.mb_size, shuffle=args.shuffle,
                              pin_memory=True, num_workers=args.workers)
    _, c, x, y = next(iter(train_loader))[0].size()
    return train_loader, c*x*y, c


def get_fashion_mnist_dataloaders(args, path_to_data='fashion-mnist'):
    """FashionMNIST dataloader with (28, 28) images."""

    print("Loading Fashion-MNIST.")
    all_transforms = transforms.Compose(args.pre_transformations + [
                                         transforms.ToTensor()] + args.post_transformations)
    train_data = datasets.FashionMNIST(path_to_data, train=args.train, download=True, transform=all_transforms)
    train_loader = DataLoader(train_data, batch_size=args.mb_size, shuffle=args.shuffle,
                              pin_memory=True, num_workers=args.workers)
    _, c, x, y = next(iter(train_loader))[0].size()
    return train_loader, c*x*y, c


def get_svhn_dataloader(args, path_to_data='svhn'):
    """SVHN dataloader with (28, 28) images."""

    print("Loading SVHN.")
    all_transforms = transforms.Compose(args.pre_transformations + [
                                         transforms.ToTensor()] + args.post_transformations)
    train_data = datasets.SVHN(path_to_data, split='test', download=False, transform=all_transforms)
    train_loader = DataLoader(train_data, batch_size=args.mb_size, shuffle=args.shuffle,
                              pin_memory=True, num_workers=args.workers)
    _, c, x, y = next(iter(train_loader))[0].size()
    return train_loader, c*x*y, c

# From https://github.com/tejaslodaya/timeseries-clustering-vae
def open_ecg5000_data(direc, ratio_train=0.9, dataset="ECG5000"):
    """Input:
    direc: location of the UCR archive
    ratio_train: ratio to split training and testset
    dataset: name of the dataset in the UCR archive"""
    datadir = direc + '/' + dataset
    data_train = np.loadtxt(datadir + '_TRAIN', delimiter=',')
    data_test = np.loadtxt(datadir + '_TEST', delimiter=',')[:-1]
    data = np.concatenate((data_train, data_test), axis=0)
    data = np.expand_dims(data, -1)

    N, D, _ = data.shape

    ind_cut = int(ratio_train * N)
    ind = np.arange(N)
    return data[ind[:ind_cut], 1:, :], data[ind[ind_cut:], 1:, :], data[ind[:ind_cut], 0, :], data[ind[ind_cut:], 0, :]

# From https://stackoverflow.com/a/55593757/3830367
class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors, transforms=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transforms = transforms

    def __getitem__(self, index):
        x = self.tensors[0][index]

        x = self.transforms(x)

        y = self.tensors[1][index]

        return x, y

    def __len__(self):
        return self.tensors[0].size(0)

def get_ecg5000_dataloader(args, path_to_data='ecg5000'):
    """ECG5000 chf07 inlier dataloader."""

    print("Loading ECG5000 inlier.")
    X_train, X_val, y_train, y_val = open_ecg5000_data(path_to_data)
    sequence_length = X_train.shape[1] # 140
    number_of_features = X_train.shape[2] # 1
    # if args.train:
    X, y = X_train, y_train
    # else:
    #     X, y = X_val, y_val
    inlier_idx = np.where(y == 1)[0]
    X, y = X[inlier_idx], np.zeros_like(y[inlier_idx])
    all_transforms = transforms.Compose(args.pre_transformations + args.post_transformations)
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(X).type(torch.FloatTensor))
    loader = DataLoader(dataset, batch_size=args.mb_size, shuffle=args.shuffle,
                              pin_memory=True, num_workers=args.workers, drop_last=True)
    # _, c, x, y = next(iter(loader))[0].size()
    return loader, sequence_length, -1

def get_ecg5000out_dataloader(args, path_to_data='ecg5000'):
    """ECG5000 chf07 outlier dataloader."""

    print("Loading ECG5000 outlier.")
    X_train, X_val, y_train, y_val = open_ecg5000_data(path_to_data)
    sequence_length = X_train.shape[1] # 140
    number_of_features = X_train.shape[2] # 1
    if args.train:
        X, y = X_train, y_train
    else:
        X, y = X_val, y_val
    outlier_idx = np.where(y != 1)[0]
    X, y = X[outlier_idx], np.zeros_like(y[outlier_idx])
    all_transforms = transforms.Compose(args.pre_transformations + args.post_transformations)
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(X).type(torch.FloatTensor))
    loader = DataLoader(dataset, batch_size=args.mb_size, shuffle=args.shuffle,
                              pin_memory=True, num_workers=args.workers, drop_last=True)
    # _, c, x, y = next(iter(loader))[0].size()
    return loader, sequence_length, -1



def get_dsprites_dataloader(args, path_to_data='dsprites'):
    """DSprites dataloader (64, 64) images"""

    name = '{}/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'.format(path_to_data)
    if not os.path.exists(name):
        print('Data at the given path doesn\'t exist. Downloading now...')
        os.system("  mkdir dsprites;"
                  "  wget -O dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz https://github.com/deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz")

    transform = transforms.Compose([transforms.ToPILImage()] + args.pre_transformations + [
                                    transforms.ToTensor()] + args.post_transformations)

    dsprites_data = DSpritesDataset(name, transform=transform)
    dsprites_loader = DataLoader(dsprites_data, batch_size=args.mb_size,
                                 shuffle=args.shuffle, pin_memory=True, num_workers=args.workers)
    _, c, x, y = next(iter(dsprites_loader))[0].size()
    return dsprites_loader, c*x*y, c


def get_cifar10_dataloader(args, path_to_data='cifar10'):
    """CIFAR10 dataloader with (32,32,3) images."""

    all_transforms = transforms.Compose(args.pre_transformations + [
                                         transforms.ToTensor()] + args.post_transformations)
    train_data = datasets.CIFAR10(path_to_data, train=args.train, download=True, transform=all_transforms)
    train_loader = DataLoader(train_data, batch_size=args.mb_size, shuffle=args.shuffle,
                              pin_memory=True, num_workers=args.workers)
    test_data = datasets.CIFAR10(path_to_data, train=False, download=True, transform=all_transforms)
    test_loader = DataLoader(test_data, batch_size=args.mb_size, shuffle=args.shuffle,
                             pin_memory=True, num_workers=args.workers)
    _, c, x, y = next(iter(train_loader))[0].size()
    return train_loader, c*x*y, c



class DSpritesDataset(Dataset):
    """DSprites dataloader class"""

    lat_names = ('shape', 'scale', 'orientation', 'posX', 'posY')
    lat_sizes = np.array([3, 6, 40, 32, 32])

    def __init__(self, path_to_data, subsample=1, transform=None):
        """
        Parameters
        ----------
        subsample : int
            Only load every |subsample| number of images.
        """
        dat = np.load(path_to_data)
        self.imgs = dat['imgs'][::subsample]
        self.lv = dat['latents_values'][::subsample]
        # self.lc = dat['latents_classes'][::subsample]
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        sample = self.imgs[idx] * 255
        sample = sample.reshape(sample.shape + (1,))

        if self.transform:
            sample = self.transform(sample)
        return sample, self.lv[idx]

def get_isun_dataloader(args, path_to_data='isun'):
    """iSUN dataloader."""

    print("Loading iSUN.")
    all_transforms = transforms.Compose(args.pre_transformations + [
                                         transforms.ToTensor()] + args.post_transformations)
    split = 'val' if not args.train else 'train-standard'
    train_data = torchvision.datasets.ImageFolder(root=path_to_data,
                                                  transform=all_transforms)
    train_loader = DataLoader(train_data, batch_size=args.mb_size, shuffle=args.shuffle,
                              pin_memory=True, num_workers=args.workers)
    _, c, x, y = next(iter(train_loader))[0].size()
    return train_loader, c*x*y, c
