import torch
import torch.nn as nn


class GaussianKernelTorch(nn.Module):
    def __init__(self, sigma2=50.0):
        super(GaussianKernelTorch, self).__init__()
        self.sigma2 = sigma2

    def forward(self, X, Y = None, D = None):
        """
        Computes the kernel matrix for some observation matrices X and Y.
        :param X: d x N matrix
        :param Y: d x M matrix. If not specified, it is assumed to be X.
        :return: N x M kernel matrix
        """
        if Y is None:
            Y = X
        N = X.shape[1] if len(X.shape) > 1 else 1
        M = Y.shape[1] if len(Y.shape) > 1 else 1

        if D is None:
            D = self.my_cdist(X, Y)

        return torch.exp(- torch.pow(D, 2) / (2 * self.sigma2))

    @staticmethod
    def my_cdist(x1, x2):
        """
        Computes a matrix of the norm of the difference.
        """
        x1 = torch.t(x1)
        x2 = torch.t(x2)
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(x2_norm.transpose(-2, -1), x1, x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
        res = res.clamp_min_(1e-30).sqrt_()
        return res



class MMD_loss(nn.Module):
    def __init__(self, sigma2 = None):
        super(MMD_loss, self).__init__()
        self.sigma2 = sigma2

    def forward(self, x, y):
        x = x.view(-1, 1)
        y = y.view(-1, 1)
        x = x.t()
        y = y.t()
        N, M = x.shape[1], y.shape[1]
        xy = torch.cat((x, y), dim=1)
        sum_Dx = torch.sum(GaussianKernelTorch.my_cdist(x, x))
        sum_Dy = torch.sum(GaussianKernelTorch.my_cdist(y, y))
        sum_Dxy = torch.sum(GaussianKernelTorch.my_cdist(xy, xy))
        # Compute sigma2 if not defined
        if self.sigma2 is None:
            self.sigma2 = 0.5 * (sum_Dx + sum_Dy + 2*sum_Dxy) / (N+M) ** 2
        kernel = GaussianKernelTorch(self.sigma2)
        mean_Kx = torch.mean(kernel(x))
        mean_Ky = torch.mean(kernel(y))
        mean_Kxy = torch.mean(kernel(x, y))
        return float(mean_Kx - 2 * mean_Kxy + mean_Ky)
