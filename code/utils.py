from pathlib import Path
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy import stats
import sklearn
from sklearn.metrics import adjusted_rand_score
import os
import skimage.transform
import numpy as np
import scipy.misc
from torchvision import transforms
from mmd import MMD_loss

rcParams['animation.convert_path'] = r'/usr/bin/convert'
rcParams['animation.ffmpeg_path'] = r'/usr/bin/ffmpeg'

# mean and standard deviation of grayscale datasets for method 6
dataset_mean = {"cifar10": 0.4809, "fashion-mnist": 0.2860, "mnist": 0.1307}
dataset_std = {"cifar10": 0.0342, "fashion-mnist": 0.0441, "mnist": 0.0385}
datasetrgb_mean = {"cifar10": [x / 255 for x in [125.3, 123.0, 113.9]], "fashion-mnist": [0.2860 for _ in range(3)], "mnist": [0.1307 for _ in range(3)]}
datasetrgb_std = {"cifar10": [x / 255 for x in [63.0, 62.1, 66.7]], "fashion-mnist": [0.0441 for _ in range(3)], "mnist": [0.0385 for _ in range(3)]}


class create_dirs:
    """ Creates directories for logging, Checkpoints and saving trained models """

    def __init__(self, name, ct):
        self.name = name
        self.ct = ct
        self.dircp = '{}_checkpoint_rkm_{}.tar'.format(self.name, self.ct)
        self.dirout = '{}_Trained_rkm_{}.tar'.format(self.name, self.ct)

    def create(self):
        if not os.path.exists('cp/{}'.format(self.name)):
            os.makedirs('cp/{}'.format(self.name))

        if not os.path.exists('log/{}'.format(self.name)):
            os.makedirs('log/{}'.format(self.name))

        if not os.path.exists('out/{}'.format(self.name)):
            os.makedirs('out/{}'.format(self.name))

    def save_checkpoint(self, state, is_best):
        if is_best:
            torch.save(state, 'cp/{}/{}'.format(self.name, self.dircp))

    def remove_checkpoint(self):
        path_checkpoint = Path('cp/{}/{}'.format(self.name, self.dircp))
        if path_checkpoint.exists():
            path_checkpoint.unlink()

class vae_create_dirs:
    """ Creates directories for logging, Checkpoints and saving trained models """

    def __init__(self, name, ct):
        self.name = name
        self.ct = ct
        self.dircp = '{}_checkpoint_vae_{}.tar'.format(self.name, self.ct)
        self.dirout = '{}_Trained_vae_{}.tar'.format(self.name, self.ct)

    def create(self):
        if not os.path.exists('cp/{}'.format(self.name)):
            os.makedirs('cp/{}'.format(self.name))

        if not os.path.exists('log/{}'.format(self.name)):
            os.makedirs('log/{}'.format(self.name))

        if not os.path.exists('out/{}'.format(self.name)):
            os.makedirs('out/{}'.format(self.name))

    def save_checkpoint(self, state, is_best):
        if is_best:
            torch.save(state, 'cp/{}/{}'.format(self.name, self.dircp))

    def remove_checkpoint(self):
        path_checkpoint = Path('cp/{}/{}'.format(self.name, self.dircp))
        if path_checkpoint.exists():
            path_checkpoint.unlink()

class gan_create_dirs:
    """ Creates directories for logging, Checkpoints and saving trained models """

    def __init__(self, name, ct):
        self.name = name
        self.ct = ct
        self.dircp = '{}_checkpoint_gan_{}.tar'.format(self.name, self.ct)
        self.dirout = '{}_Trained_gan_{}.tar'.format(self.name, self.ct)

    def create(self):
        if not os.path.exists('cp/{}'.format(self.name)):
            os.makedirs('cp/{}'.format(self.name))

        if not os.path.exists('log/{}'.format(self.name)):
            os.makedirs('log/{}'.format(self.name))

        if not os.path.exists('out/{}'.format(self.name)):
            os.makedirs('out/{}'.format(self.name))

    def save_checkpoint(self, state, is_best):
        if is_best:
            torch.save(state, 'cp/{}/{}'.format(self.name, self.dircp))

    def remove_checkpoint(self):
        path_checkpoint = Path('cp/{}/{}'.format(self.name, self.dircp))
        if path_checkpoint.exists():
            path_checkpoint.unlink()

class cnn_create_dirs:
    """ Creates directories for logging, Checkpoints and saving trained models """

    def __init__(self, name, ct):
        self.name = name
        self.ct = ct
        self.dircp = '{}_checkpoint_cnn_{}.tar'.format(self.name, self.ct)
        self.dirout = '{}_Trained_cnn_{}.tar'.format(self.name, self.ct)

    def create(self):
        if not os.path.exists('cp/{}'.format(self.name)):
            os.makedirs('cp/{}'.format(self.name))

        if not os.path.exists('log/{}'.format(self.name)):
            os.makedirs('log/{}'.format(self.name))

        if not os.path.exists('out/{}'.format(self.name)):
            os.makedirs('out/{}'.format(self.name))

    def save_checkpoint(self, state, is_best):
        if is_best:
            torch.save(state, 'cp/{}/{}'.format(self.name, self.dircp))


class pca_create_dirs:
    """ Creates directories for logging, Checkpoints and saving trained models """

    def __init__(self, name, ct):
        self.name = name
        self.ct = ct
        self.dircp = 'checkpoint.pth_{}.tar'.format(self.ct)
        self.dirout = '{}_Trained_pca_{}.tar'.format(self.name, self.ct)

    def create(self):
        if not os.path.exists('cp/{}'.format(self.name)):
            os.makedirs('cp/{}'.format(self.name))

        if not os.path.exists('log/{}'.format(self.name)):
            os.makedirs('log/{}'.format(self.name))

        if not os.path.exists('out/{}'.format(self.name)):
            os.makedirs('out/{}'.format(self.name))

    def save_checkpoint(self, state, is_best):
        if is_best:
            torch.save(state, 'cp/{}/{}'.format(self.name, self.dircp))


def convert_to_imshow_format(image):
    # convert from CHW to HWC
    if image.shape[0] == 1:
        return image[0, :, :]
    else:
        if np.any(np.where(image < 0)):
            # first convert back to [0,1] range from [-1,1] range
            image = image / 2 + 0.5
        return image.transpose(1, 2, 0)



class Lin_View(nn.Module):
    """ Unflatten linear layer to be used in Convolution layer"""

    def __init__(self, c, a, b):
        super(Lin_View, self).__init__()
        self.c, self.a, self.b = c, a, b

    def forward(self, x):
        try:
            return x.view(x.size(0), self.c, self.a, self.b)
        except:
            return x.view(1, self.c, self.a, self.b)


class Resize:
    def __init__(self, size):
        assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        if isinstance(size, int):
            self._size = (size, size)
        else:
            self._size = size

    def __call__(self, img: np.ndarray):
        resize_image = skimage.transform.resize(img, self._size)
        # the resize will return a float32 array
        return skimage.util.img_as_float32(resize_image)


def scatter_w_hist(h,y=None):
    """ 2D scatter plot of latent variables"""
    fig = plt.figure()
    grid = plt.GridSpec(4, 4, hspace=0.0, wspace=0.0)
    main_ax = fig.add_subplot(grid[:-1, 1:])
    y_hist = fig.add_subplot(grid[:-1, 0], xticklabels=[], sharey=main_ax)
    x_hist = fig.add_subplot(grid[-1, 1:], yticklabels=[], sharex=main_ax)

    sc = main_ax.scatter(h[:, 0].detach().numpy(), h[:, 1].detach().numpy(), s=0.01, c=y)
    main_ax.legend(*sc.legend_elements())
    _, binsx, _ = x_hist.hist(h[:, 0].detach().numpy(), 40, histtype='stepfilled', density=True,
                              orientation='vertical')
    _, binsy, _ = y_hist.hist(h[:, 1].detach().numpy(), 40, histtype='stepfilled', density=True,
                              orientation='horizontal')
    x_hist.invert_yaxis()
    y_hist.invert_xaxis()
    plt.setp(main_ax.get_xticklabels(), visible=False)
    plt.setp(main_ax.get_yticklabels(), visible=False)
    plt.show()
    return fig

# From https://github.com/wetliu/energy_ood
def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """
    From https://github.com/wetliu/energy_ood
    """
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out

# From https://github.com/wetliu/energy_ood
def fpr_and_fdr_at_recall(y_true, y_score, recall_level=0.95, pos_label=None):
    """
    From https://github.com/wetliu/energy_ood
    """
    classes = np.unique(y_true)
    if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                     np.array_equal(classes, [-1, 1]) or
                     np.array_equal(classes, [0]) or
                     np.array_equal(classes, [-1]) or
                     np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps      # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)      # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    return fps[cutoff] / (np.sum(np.logical_not(y_true)))   # , fps[cutoff]/(fps[cutoff] + tps[cutoff])

# From https://github.com/wetliu/energy_ood
def get_measures(_pos, _neg, recall_level=0.95):
    """
    From https://github.com/wetliu/energy_ood
    The higher the value, the more positive, the more likely to be in distribution.
    """
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1

    auroc = sklearn.metrics.roc_auc_score(labels, examples)
    aupr = sklearn.metrics.average_precision_score(labels, examples)
    fpr = fpr_and_fdr_at_recall(labels, examples, recall_level)

    return auroc, aupr, fpr

# From https://github.com/wetliu/energy_ood
def compute_mean_std_dataloader(loader):
    mean = 0.0
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
    mean = mean / len(loader.dataset)

    var = 0.0
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        var += ((images - mean.unsqueeze(1)) ** 2).sum([0, 2])
    std = torch.sqrt(var / (len(loader.dataset) * 224 * 224))
    return mean, std

class ChannelTransform(torch.nn.Module):
    """Depending on a flag, it always converts images to grayscale or
       it always converts images to RGB.

    Args:
        rgb: if True, always RGB, if False, always Grayscale.
    """

    def __init__(self, rgb=False):
        super().__init__()
        self.rgb = rgb
        self.grayscale_transform = transforms.Grayscale()
        self.rgb_transform = transforms.Lambda(lambda x: x.convert('RGB'))

    def forward(self, sample):
        if not self.rgb:
            return self.grayscale_transform(sample)
        else:
            return self.rgb_transform(sample)

def merge_two_dicts(x, y):
    return {**x, **y}


def get_overlap(pos_scores: np.ndarray, neg_scores: np.ndarray, preprocess=True):
    if preprocess:
        scores = sklearn.preprocessing.scale(np.concatenate([pos_scores, neg_scores]))
        scores += -scores.min()
        pos_scores = scores[:len(pos_scores)]
        neg_scores = scores[len(pos_scores):]

    # Start R
    from rpy2.robjects import r, FloatVector
    r["library"]("overlapping")

    wd = r["overlap"](list(map(FloatVector, [pos_scores, neg_scores])))[1][0]
    return wd

def get_mmd(pos_scores: np.ndarray, neg_scores: np.ndarray, preprocess=True):
    if preprocess:
        scores = sklearn.preprocessing.scale(np.concatenate([pos_scores, neg_scores]))
        scores += -scores.min()
        pos_scores = scores[:len(pos_scores)]
        neg_scores = scores[len(pos_scores):]
    mmd_evaluator = MMD_loss()
    wd = float(mmd_evaluator(torch.from_numpy(pos_scores).to("cuda"), torch.from_numpy(neg_scores).to("cuda")))
    return wd

def get_wd(pos_scores: np.ndarray, neg_scores: np.ndarray, preprocess=True):
    if preprocess:
        scores = sklearn.preprocessing.scale(np.concatenate([pos_scores, neg_scores]))
        scores += -scores.min()
        pos_scores = scores[:len(pos_scores)]
        neg_scores = scores[len(pos_scores):]

    wd = scipy.stats.wasserstein_distance(pos_scores, neg_scores)
    return wd
