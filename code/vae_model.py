import torch
import torch.nn as nn
from dataloader import *
from stiefel_rkm_model import Net1, Net3
from tqdm import tqdm


class VAE(nn.Module):
    def __init__(self, ipVec_dim, args, nChannels=1, recon_loss=nn.BCELoss(reduction='sum'), ngpus=1,
                 device=torch.device('cuda'), Encoder=Net1, Decoder=Net3):
        super(VAE, self).__init__()
        self.device = device
        self.ipVec_dim = ipVec_dim
        self.ngpus = ngpus
        self.args = args
        self.nChannels = nChannels

        # Settings for Conv layers
        self.cnn_kwargs = dict(kernel_size=4, stride=2, padding=1)
        if self.ipVec_dim == 140: #ecg5000
            self.cnn_kwargs = [1, ipVec_dim]
        elif self.ipVec_dim <= 28*28*3:
            self.cnn_kwargs = self.cnn_kwargs, dict(kernel_size=3, stride=1), 5
        else:
            self.cnn_kwargs = self.cnn_kwargs, self.cnn_kwargs, 4

        if self.ngpus > 1:
            self.encoder = nn.DataParallel(Net1(self.nChannels, self.args, self.cnn_kwargs), list(range(ngpus)))
            self.decoder = nn.DataParallel(Net3(self.nChannels, self.args, self.cnn_kwargs), list(range(ngpus)))
        else:
            self.encoder = Encoder(self.nChannels, self.args, self.cnn_kwargs)
            self.decoder = Decoder(self.nChannels, self.args, self.cnn_kwargs)

        if type(recon_loss) == str:
            self.recon_loss = nn.BCELoss(reduction='sum')
            if recon_loss == "mse":
                self.recon_loss = nn.MSELoss(reduction='sum')
            elif recon_loss == "bce":
                self.recon_loss = nn.BCELoss(reduction='sum')
        else:
            self.recon_loss = recon_loss

        self.fc1 = nn.Linear(self.args.x_fdim2, self.args.h_dim)  # mu
        self.fc2 = nn.Linear(self.args.x_fdim2, self.args.h_dim)  # logvar
        self.fc3 = nn.Linear(self.args.h_dim, self.args.x_fdim2)  # upsample from z

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        try:
            esp = torch.randn(*mu.size()).to(self.device)
            z = mu + std * esp
        except:
            esp = torch.randn(*mu.size()).to('cpu')
            z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def representation(self, x):
        return self.bottleneck(self.encoder(x))[0]

    def forward(self, x):
        z, mu, logvar = self.bottleneck(self.encoder(x))

        # loss function ----------
        recon = self.recon_loss(self.decoder(self.fc3(z)).view(-1, self.ipVec_dim),
                                x.view(-1, self.ipVec_dim)) / x.size(0)  # Reconstruction loss

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = self.args.beta * -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

        return recon + KLD, KLD, recon

    def compute_pointwise_energy(self, x):
        z, mu, logvar = self.bottleneck(self.encoder(x))
        loss = nn.MSELoss(reduction='none')

        # loss function ----------
        recon = loss(self.decoder(self.fc3(z)).view(-1, self.ipVec_dim),
                                x.view(-1, self.ipVec_dim))
        recon = torch.sum(recon, dim=list(range(1,len(recon.shape))))

        return recon

    def compute_latent(self, x):
        z, mu, logvar = self.bottleneck(self.encoder(x))
        return mu, logvar

def vae_final_compute(model, args, ct, device=torch.device('cuda')):
    """ Function to compute embeddings of full dataset. """
    if not os.path.exists('oti/'):
        os.makedirs('oti/')

    args.shuffle = False
    x, _, _ = get_dataloader(args=args)  # loading data without shuffle

    for i, sample_batch in enumerate(tqdm(x)):
        z, _, _ = model.bottleneck(model.encoder(sample_batch[0].to(args.proc)))
        torch.save({'oti': z}, 'oti/oti{}_vae_checkpoint.pth_{}.tar'.format(i, ct))

    ot = torch.Tensor([]).to(args.proc)
    for i in range(0, len(x)):
        # print(i)
        ot = torch.cat((ot, torch.load('oti/oti{}_vae_checkpoint.pth_{}.tar'.format(i, ct))['oti']), dim=0)

    return ot
