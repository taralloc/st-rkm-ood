import shutil

import torch
from tqdm import tqdm
import torch.nn as nn
import stiefel_optimizer
from dataloader import *
from utils import Lin_View

class Net1(nn.Module):
    """ Encoder - network architecture """
    def __init__(self, nChannels, args, cnn_kwargs):
        super(Net1, self).__init__()  # inheritance used here.
        self.args = args
        self.main = nn.Sequential(
            nn.Conv2d(nChannels, self.args.capacity, **cnn_kwargs[0]),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Conv2d(self.args.capacity, self.args.capacity * 2, **cnn_kwargs[0]),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Conv2d(self.args.capacity * 2, self.args.capacity * 4, **cnn_kwargs[1]),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Flatten(),
            nn.Linear(self.args.capacity * 4 * cnn_kwargs[2] ** 2, self.args.x_fdim1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(self.args.x_fdim1, self.args.x_fdim2),
        )

    def forward(self, x):
        return self.main(x)



class Net3(nn.Module):
    """ Decoder - network architecture """
    def __init__(self, nChannels, args, cnn_kwargs):
        super(Net3, self).__init__()
        self.args = args
        self.main = nn.Sequential(
            nn.Linear(self.args.x_fdim2, self.args.x_fdim1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(self.args.x_fdim1, self.args.capacity * 4 * cnn_kwargs[2] ** 2),
            nn.LeakyReLU(negative_slope=0.2),
            Lin_View(self.args.capacity * 4, cnn_kwargs[2], cnn_kwargs[2]),  # Unflatten

            nn.ConvTranspose2d(self.args.capacity * 4, self.args.capacity * 2, **cnn_kwargs[1]),
            nn.LeakyReLU(negative_slope=0.2),

            nn.ConvTranspose2d(self.args.capacity * 2, self.args.capacity, **cnn_kwargs[0]),
            nn.LeakyReLU(negative_slope=0.2),

            nn.ConvTranspose2d(self.args.capacity, nChannels, **cnn_kwargs[0]),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)

class Net2(nn.Module):
    """ Encoder - network architecture """
    def __init__(self, nChannels, args, cnn_kwargs):
        super(Net2, self).__init__()  # inheritance used here.
        self.args = args
        self.main = nn.Sequential(
            nn.Conv2d(nChannels, self.args.capacity, **cnn_kwargs[0]),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Conv2d(self.args.capacity, self.args.capacity * 2, **cnn_kwargs[0]),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Conv2d(self.args.capacity * 2, self.args.capacity * 4, **cnn_kwargs[1]),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Flatten(),
            nn.Linear(self.args.capacity * 4 * cnn_kwargs[2] ** 2, self.args.x_fdim2),
            nn.LeakyReLU(negative_slope=0.2),
        )

    def forward(self, x):
        return self.main(x)

class Net4(nn.Module):
    """ Decoder - network architecture """
    def __init__(self, nChannels, args, cnn_kwargs):
        super(Net4, self).__init__()
        self.args = args
        self.main = nn.Sequential(
            nn.Linear(self.args.x_fdim2, self.args.capacity * 4 * cnn_kwargs[2] ** 2),
            nn.LeakyReLU(negative_slope=0.2),

            Lin_View(self.args.capacity * 4, cnn_kwargs[2], cnn_kwargs[2]),  # Unflatten

            nn.ConvTranspose2d(self.args.capacity * 4, self.args.capacity * 2, **cnn_kwargs[1]),
            nn.LeakyReLU(negative_slope=0.2),

            nn.ConvTranspose2d(self.args.capacity * 2, self.args.capacity, **cnn_kwargs[0]),
            nn.LeakyReLU(negative_slope=0.2),

            nn.ConvTranspose2d(self.args.capacity, nChannels, **cnn_kwargs[0]),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)





class RKM_Stiefel(nn.Module):
    """ Defines the Stiefel RKM model and its loss functions """
    def __init__(self, ipVec_dim, args, nChannels=1, recon_loss=nn.MSELoss(reduction='sum'), ngpus=1, device='cpu', cutoff=700,
                 Encoder=Net1, Decoder=Net3, EncoderDecoder=None):
        super(RKM_Stiefel, self).__init__()
        self.ipVec_dim = ipVec_dim
        self.ngpus = ngpus
        self.args = args
        self.nChannels = nChannels

        # Initialize manifold parameter
        self.manifold_param = nn.Parameter(nn.init.orthogonal_(torch.Tensor(self.args.h_dim, self.args.x_fdim2))) #U matrix, m x l

        # Settings for Conv layers
        self.cnn_kwargs = dict(kernel_size=4, stride=2, padding=1)
        if self.ipVec_dim == 140: #ecg5000
            self.cnn_kwargs = [1, ipVec_dim]
        elif self.ipVec_dim <= 28*28*3:
            self.cnn_kwargs = self.cnn_kwargs, dict(kernel_size=3, stride=1), 5
        else:
            self.cnn_kwargs = self.cnn_kwargs, self.cnn_kwargs, 4

        if type(recon_loss) == str:
            self.recon_loss = nn.BCELoss(reduction='sum')
            if recon_loss == "mse":
                self.recon_loss = nn.MSELoss(reduction='sum')
            elif recon_loss == "bce":
                self.recon_loss = nn.BCELoss(reduction='sum')
        else:
            self.recon_loss = recon_loss

        if Encoder is not None:
            self.encoder = Encoder(self.nChannels, self.args, self.cnn_kwargs)
        else:
            self.encoder = None
        if Decoder is not None:
            self.decoder = Decoder(self.nChannels, self.args, self.cnn_kwargs)
        else:
            self.decoder = None
        if EncoderDecoder is not None:
            self.encoderdecoder = EncoderDecoder().to(device)
        else:
            self.encoderdecoder = None
        self.device = device
        self.cutoff = cutoff

    def forward(self, x, t=0):
        x_tilde = None
        if self.encoderdecoder is not None:
            x_tilde, op1 = self.encoderdecoder(x)
            op1 = op1.view(op1.shape[0], -1)
        else:
            op1 = self.encoder(x)  # features
        op1 = op1 - torch.mean(op1, dim=0)  # feature centering
        C = torch.mm(op1.t(), op1)  # Covariance matrix

        """ Various types of losses as described in paper """
        if self.args.loss == 'splitloss':
            x_tilde1 = self.decoder(torch.mm(torch.mm(op1, self.manifold_param.t())
                                            + self.args.noise_level * torch.randn((x.shape[0], self.args.h_dim)).to(self.args.proc),
                                            self.manifold_param))
            x_tilde2 = self.decoder(torch.mm(torch.mm(op1, self.manifold_param.t()), self.manifold_param))
            f2 = self.args.c_accu * 0.5 * (
                    self.recon_loss(x_tilde2.view(-1, self.ipVec_dim), x.view(-1, self.ipVec_dim))
                    + self.recon_loss(x_tilde2.view(-1, self.ipVec_dim),
                                      x_tilde1.view(-1, self.ipVec_dim))) / x.size(0)  # Recons_loss

        elif self.args.loss == 'noisyU':
            x_tilde = self.decoder(torch.mm(torch.mm(op1, self.manifold_param.t())
                                            + self.args.noise_level * torch.randn((x.shape[0], self.args.h_dim)).to(self.args.proc),
                                            self.manifold_param))
            f2 = self.args.c_accu * 0.5 * (
                self.recon_loss(x_tilde.view(-1, self.ipVec_dim), x.view(-1, self.ipVec_dim))) / x.size(0)  # Recons_loss

        elif self.args.loss == 'deterministic':
            if x_tilde is None:
                x_tilde = self.decoder(torch.mm(op1, torch.mm(self.manifold_param.t(), self.manifold_param))) #output of decoder
            f2 = self.args.c_accu * 0.5 * (self.recon_loss(x_tilde.view(-1, self.ipVec_dim), x.view(-1, self.ipVec_dim)))/x.size(0)  # Recons_loss

        f1 = torch.trace(C - torch.mm(torch.mm(self.manifold_param.t(), self.manifold_param), C))/x.size(0)  # KPCA

        # Weighted version
        if t < self.cutoff: #first some% epochs
            f1_k = f1
        else:
            K = torch.mm(op1, op1.t())
            D = torch.diag(torch.mm(K, torch.ones((x.size(0), 1), device=self.device)).flatten())
            f1_k = torch.trace(torch.mm(D, K)) - torch.trace(torch.mm(torch.mm(self.manifold_param.t(), self.manifold_param), torch.mm(torch.mm(op1.t(), D), op1)))
        return f1_k + f2, f1_k, f2

        return f1 + f2, f1, f2

    def compute_pointwise_energy(self, x, mean=0.0, method=0, normalize_corr=False):
        assert method in [0, 1, 2, 3, 4, 8, 9, 10, 11, 12, 13]
        N = x.shape[0]

        op1 = self.encoder(x) # [N, l]
        op1 = op1 - mean

        h = torch.matmul(op1, self.manifold_param.t()) # [N, m]

        f1_0 = - 2.0 * torch.diagonal(torch.matmul(torch.matmul(op1, self.manifold_param.t()), h.t()))

        f1_1 = torch.diagonal(torch.mm(h, h.t()))

        f1_2 = torch.diagonal(torch.mm(op1, op1.t()))

        """ Various types of losses as described in paper """
        temp = self.recon_loss.reduction
        self.recon_loss.reduction = 'none'
        if self.args.loss == 'splitloss':
            x_tilde1 = self.decoder(torch.mm(torch.mm(op1, self.manifold_param.t())
                                            + self.args.noise_level * torch.randn((x.shape[0], self.args.h_dim)).to(self.args.proc),
                                            self.manifold_param))
            x_tilde2 = self.decoder(torch.mm(torch.mm(op1, self.manifold_param.t()), self.manifold_param))
            f2 = self.args.c_accu * 0.5 * (
                    self.recon_loss(x_tilde2.view(-1, self.ipVec_dim), x.view(-1, self.ipVec_dim))
                    + self.recon_loss(x_tilde2.view(-1, self.ipVec_dim),
                                      x_tilde1.view(-1, self.ipVec_dim))) / x.size(0)  # Recons_loss

        elif self.args.loss == 'noisyU':
            x_tilde = self.decoder(torch.mm(torch.mm(op1, self.manifold_param.t())
                                            + self.args.noise_level * torch.randn((x.shape[0], self.args.h_dim)).to(self.args.proc),
                                            self.manifold_param))
            f2 = self.args.c_accu * 0.5 * torch.sum(self.recon_loss(x_tilde.view(-1, self.ipVec_dim), x.view(-1, self.ipVec_dim)), axis=1) / x.size(0)  # Recons_loss

        elif self.args.loss == 'deterministic':
            x_tilde = self.decoder(torch.mm(op1, torch.mm(self.manifold_param.t(), self.manifold_param))) #output of decoder
            f2 = self.args.c_accu * 0.5 * torch.sum(self.recon_loss(x_tilde.view(-1, self.ipVec_dim), x.view(-1, self.ipVec_dim)), axis=1)/x.size(0)  # Recons_loss
        self.recon_loss.reduction = temp

        if normalize_corr:
            f1_0 = f1_0 / (f1_1 * f1_2)

        if method == 0:
            kpcaerr = f1_0 + f1_1 + f1_2
            aeloss = f2
            return kpcaerr + aeloss
        elif method == 1:
            kpcaerr = f1_0 + f1_1 + f1_2
            if torch.norm(torch.eye(self.manifold_param.shape[1], device=self.manifold_param.device) - torch.mm(self.manifold_param.t(), self.manifold_param)) < 1e-3: #if UU' is I, perfect reconstruction
                kpcaerr = torch.zeros(kpcaerr.shape, device=kpcaerr.device)
            return kpcaerr
        elif method == 2:
            assert (f2 >= 0).all()
            return f2
        elif method == 3:
            return f1_0
        elif method == 4:
            return -f1_0
        elif method == 8:
            return f1_1
        elif method == 9:
            return f1_2
        elif method == 10:
            return f1_1 + f1_2
        elif method == 11:
            return f1_0 + f1_1
        elif method == 12:
            return f1_0 + f1_2
        elif method == 13:
            return f1_0 / (f1_1 * f1_2) if not normalize_corr else f1_0

# Accumulate trainable parameters in 2 groups. 1. Manifold_params 2. Network param
def param_state(model):
    param_g, param_e1 = [], []
    for name, param in model.named_parameters():
        if param.requires_grad and name != 'manifold_param':
            param_e1.append(param)
        elif name == 'manifold_param':
            param_g.append(param)
    return param_g, param_e1

def stiefel_opti(stief_param, lrg=1e-4):
    dict_g = {'params': stief_param, 'lr': lrg, 'momentum': 0.9, 'weight_decay': 0.0005, 'stiefel': True}
    return stiefel_optimizer.AdamG([dict_g])  # CayleyAdam

def compute_ot(model, args, ct, device=torch.device('cpu'), train_mean=None):
    if not os.path.exists('oti/'):
        os.makedirs('oti/')

    args.shuffle = False
    x, _, _ = get_dataloader(args)

    # Compute feature-vectors
    for i, sample_batch in enumerate(tqdm(x)):
        torch.save({'oti': model.encoder(sample_batch[0].to(device))},
                   'oti/oti{}_checkpoint.pth_{}.tar'.format(i, ct))

    # Load feature-vectors
    ot = torch.Tensor([]).to(device)
    for i in range(0, len(x)):
        ot = torch.cat((ot, torch.load('oti/oti{}_checkpoint.pth_{}.tar'.format(i, ct))['oti']), dim=0)
    #os.removedirs("oti/")
    shutil.rmtree("oti/")

    mean = train_mean if train_mean is not None else torch.mean(ot, dim=0)
    mean = mean.to(device)
    ot = (ot - mean).to(device)  # Centering
    return ot, mean


def final_compute(model, args, ct, device=torch.device('cuda'), train_mean: float = None):
    """ Utility to re-compute U. Since some datasets could exceed the GPU memory limits, some intermediate
    variables are saved  on HDD, and retrieved later"""
    if not os.path.exists('oti/'):
        os.makedirs('oti/')

    args.shuffle = False
    x, _, _ = get_dataloader(args)

    # Compute feature-vectors
    for i, sample_batch in enumerate(tqdm(x)):
        torch.save({'oti': model.encoder(sample_batch[0].to(device))},
                   'oti/oti{}_checkpoint.pth_{}.tar'.format(i, ct))

    # Load feature-vectors
    ot = torch.Tensor([]).to(device)
    for i in range(0, len(x)):
        ot = torch.cat((ot, torch.load('oti/oti{}_checkpoint.pth_{}.tar'.format(i, ct))['oti']), dim=0)
    #os.removedirs("oti/")
    shutil.rmtree("oti/")

    mean = train_mean if train_mean is not None else torch.mean(ot, dim=0)
    ot = (ot - mean).to(device)  # Centering
    u, _, _ = torch.svd(torch.mm(ot.t(), ot))
    u = u[:, :args.h_dim]
    with torch.no_grad():
        model.manifold_param.masked_scatter_(model.manifold_param != u.t(), u.t())
    return torch.mm(ot, u.to(device)), u, ot, mean
