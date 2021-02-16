import copy
from utils import pca_create_dirs
from stiefel_rkm_model import *
import logging
import argparse
import time
from datetime import datetime

def pca(xtrain, opt):
    newopt = copy.deepcopy(opt)
    newopt.mb_size = len(xtrain.dataset)
    xtrain, _, _= get_dataloader(args=newopt)
    X = None
    for i, data in enumerate(xtrain, 0):
        X = data[0]
    X = X.view(X.shape[0], -1)
    mean = torch.mean(X, dim=0)
    X = X - mean
    C = torch.mm(X.t(), X) / (X.shape[0] - 1)
    eigvecs, eigvals, _ = torch.svd(C, some=False)
    return X, eigvecs, eigvals, mean

if __name__ == '__main__':
    # Model Settings =================================================================================================
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', type=str, default='cifar10',
                        help='Dataset name: mnist/fashion-mnist/svhn/dsprites/cifar10')
    parser.add_argument('--maxfrac', type=float, default=0.02, help='Maximum fraction of variance for removed components')
    parser.add_argument('--mb_size', type=int, default=256, help='Mini-batch size')
    parser.add_argument('--shuffle', type=bool, default=True, help='shuffle dataset: True/False')
    parser.add_argument('--proc', type=str, default='cpu', help='device type: cuda or cpu')
    parser.add_argument('--workers', type=int, default=0, help='Number of workers for dataloader')
    opt = parser.parse_args()
    # ==================================================================================================================

    ct = time.strftime("%Y%m%d-%H%M")
    dirs = pca_create_dirs(name=opt.dataset_name, ct=ct)
    dirs.create()

    # noinspection PyArgumentList
    logging.basicConfig(level=logging.INFO,
                        format='%(message)s',
                        handlers=[logging.FileHandler('log/{}/{}_Trained_pca_{}.log'.format(opt.dataset_name, opt.dataset_name, ct)),
                                  logging.StreamHandler()])

    """ Load Training Data """
    vars(opt)['train'] = True
    xtrain, ipVec_dim, nChannels, xval = get_dataloader_training(args=opt)

    logging.info(opt)
    logging.info('\nN: {}, mb_size: {}'.format(len(xtrain.dataset), opt.mb_size))

    # Train =========================================================================================================
    start = datetime.now()

    temp = opt.mb_size
    opt.mb_size = len(xtrain.dataset)
    xtrain, _, _, _ = get_dataloader_training(args=opt)
    opt.mb_size = temp
    X = None
    for i, data in enumerate(xtrain, 0):
        X = data[0]
    X = X.view(X.shape[0], -1)
    mean = torch.mean(X, dim=0)
    X = X - mean
    C = torch.mm(X.t(), X) / (X.shape[0] - 1)
    eigvecs, eigvals, _ = torch.svd(C, some=False)

    total_variance = torch.sum(eigvals)
    cutoff_index = eigvecs.shape[0]
    for i in range(eigvecs.shape[0]):
        explained_variance = eigvals[i] / total_variance
        if explained_variance < opt.maxfrac:
            cutoff_index = i
            break
    logging.info(f"{cutoff_index} principal components kept.")

    logging.info("\nTraining complete in: " + str(datetime.now() - start))

    # Save Model and Tensors ======================================================================================
    torch.save({'eigvec': eigvecs[:cutoff_index],
                'mean': mean,
                'opt': opt}, 'out/{}/{}'.format(opt.dataset_name, dirs.dirout))
    logging.info('\nSaved File: {}'.format(dirs.dirout))
