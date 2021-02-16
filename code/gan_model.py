import torch
import torch.nn as nn
from stiefel_rkm_model import Net1, Net3


class Discriminator(nn.Module):
    def __init__(self, ipVec_dim, args, nChannels=1, device=torch.device('cuda'), Encoder=Net1):
        super(Discriminator, self).__init__()
        self.device = device
        self.ipVec_dim = ipVec_dim
        self.args = args
        self.nChannels = nChannels

        self.cnn_kwargs = dict(kernel_size=4, stride=2, padding=1)
        if self.ipVec_dim == 140:
            self.cnn_kwargs = [1, ipVec_dim]
        elif self.ipVec_dim <= 28 * 28 * 3:
            self.cnn_kwargs = self.cnn_kwargs, dict(kernel_size=3, stride=1), 5
        else:
            self.cnn_kwargs = self.cnn_kwargs, self.cnn_kwargs, 4

        self.encoder = Encoder(self.nChannels, self.args, self.cnn_kwargs)

        self.fc1 = nn.Linear(self.args.x_fdim2, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        return self.sig(self.fc1(self.encoder(x)))

class Generator(nn.Module):
    def __init__(self, ipVec_dim, args, nChannels=1, device=torch.device('cuda'), Decoder=Net3):
        super(Generator, self).__init__()
        self.device = device
        self.ipVec_dim = ipVec_dim
        self.args = args
        self.nChannels = nChannels

        self.cnn_kwargs = dict(kernel_size=4, stride=2, padding=1)
        if self.ipVec_dim == 140:
            self.cnn_kwargs = [1, ipVec_dim]
        elif self.ipVec_dim <= 28 * 28 * 3:
            self.cnn_kwargs = self.cnn_kwargs, dict(kernel_size=3, stride=1), 5
        else:
            self.cnn_kwargs = self.cnn_kwargs, self.cnn_kwargs, 4

        self.decoder = Decoder(self.nChannels, self.args, self.cnn_kwargs)

        self.fc3 = nn.Linear(self.args.h_dim, self.args.x_fdim2)  # upsample from z

    def forward(self, z):
        # loss function ----------
        recon = self.decoder(self.fc3(z.view(-1,self.args.h_dim)))
        return recon
