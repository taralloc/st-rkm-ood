import time
from pathlib import Path
from vae_model import *
from datetime import datetime
from utils import vae_create_dirs
import logging
from dataloader import *
import argparse
from tqdm import tqdm

from vrae_model import VRAEEncoder, VRAEDecoder

""" Specify datatset name (mnist, fashion-mnist, cifar10, dsprites, celeba) and location. 
    Remember to change the architecture according to input size and ranges of input and output."""
#dataset_name = 'fashion-mnist'  # training dataset
#data_root = 'fashion-mnist'

# Model Settings =================================================================================================
parser = argparse.ArgumentParser(description='VAE Model',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset_name', type=str, default='cifar10',
                    help='Dataset name: mnist/fashion-mnist/svhn/dsprites/cifar10/ecg5000')
parser.add_argument('--h_dim', type=int, default=1024, help='Dim of latent vector')
parser.add_argument('--capacity', type=int, default=64, help='Capacity of network. See utils.py')
parser.add_argument('--mb_size', type=int, default=128, help='Mini-batch size. See utils.py')
parser.add_argument('--x_fdim1', type=int, default=1, help='Input x_fdim1. See utils.py')
parser.add_argument('--x_fdim2', type=int, default=1500, help='Input x_fdim2. See utils.py')
parser.add_argument('--beta', type=int, default=1, help='Weight on D_{kl} in beta-vae')
parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint file')

# Training Settings =============================
parser.add_argument('--lr', type=float, default=2e-4, help='Input learning rate for optimizer')
parser.add_argument('--epoch_step', default=[800, 7000], type=int, help='List with epochs to drop lr on')
parser.add_argument('--lr_decay_ratio', default=0.2, type=float)
parser.add_argument('--max_epochs', type=int, default=4000, help='Input max_epoch for cut-off')
parser.add_argument('--proc', type=str, default='cuda', help='Processor type: cuda or cpu')
parser.add_argument('--workers', type=int, default=16, help='# of workers for dataloader')
parser.add_argument('--shuffle', type=bool, default=True, help='shuffle dataset: true or false')
parser.add_argument('--recon_loss', type=str, default="bce", choices=["bce", "mse"], help='reconstruction loss')

opt = parser.parse_args()
# ==================================================================================================================

device = torch.device(opt.proc)

if torch.cuda.is_available():
    torch.cuda.empty_cache()

ct = time.strftime("%Y%m%d-%H%M")
dirs = vae_create_dirs(name=opt.dataset_name, ct=ct)
dirs.create()

# noinspection PyArgumentList
logging.basicConfig(level=logging.INFO,
                    format='%(message)s',
                    handlers=[logging.FileHandler('log/{}/{}_Trained_vae_{}.log'.format(opt.dataset_name, opt.dataset_name, ct)),
                              logging.StreamHandler()])

""" Load Training Data """
vars(opt)['train'] = True
xtrain, ipVec_dim, nChannels = get_dataloader(args=opt)

ngpus = torch.cuda.device_count()

Encoder, Decoder = None, None
if opt.dataset_name == "fashion-mnist":
    Encoder, Decoder = Net1, Net3
elif opt.dataset_name == "cifar10":
    Encoder, Decoder = Net2, Net4
elif opt.dataset_name == "ecg5000":
    Encoder, Decoder = VRAEEncoder, VRAEDecoder
vae = VAE(ipVec_dim=ipVec_dim, args=opt, nChannels=nChannels, recon_loss=opt.recon_loss, ngpus=ngpus, Encoder=Encoder, Decoder=Decoder).to(device)
logging.info(vae)
logging.info(opt)
logging.info('\nN: {}, mb_size: {}'.format(len(xtrain.dataset), opt.mb_size))
logging.info('We are using {} GPU(s)!'.format(ngpus))
logging.info('The number of parameters of model: {}'.format(sum(p.numel() for p in vae.parameters() if p.requires_grad)))

optimizer = torch.optim.Adam(vae.parameters(), lr=opt.lr, weight_decay=0)

# Train =========================================================================================================
start = datetime.now()
Loss_stk = np.empty(shape=[0, 3])
cost, l_cost = np.inf, np.inf  # Initialize cost
is_best = False
t = 1
# hi = np.empty(shape=[N, opt.h_dim, opt.max_epochs])

# Load checkpoint
if opt.checkpoint is not None:
    checkpoint = Path("cp/" + opt.dataset_name + "/" + opt.checkpoint + ".tar")
    if not checkpoint.exists():
        checkpoint = Path("out/" + opt.dataset_name + "/" + opt.checkpoint + ".tar")
    checkpoint = torch.load(str(checkpoint))
    vae.load_state_dict(checkpoint['vae_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    t = checkpoint.get('epochs', 400) + 1
    print(f"Loaded checkpoint at epoch {t}")

while cost > 1e-10 and t <= opt.max_epochs:  # run epochs until convergence
    avg_loss, avg_f1, avg_f2 = 0, 0, 0

    for _, sample_batched in enumerate(tqdm(xtrain, desc="Epoch {}/{}".format(t + 1, opt.max_epochs))):
        loss, f1, f2 = vae(sample_batched[0].to(device))

        optimizer.zero_grad()
        loss.mean().backward()  # 'mean' to handle multi-gpu training
        optimizer.step()

        avg_loss += loss.mean().item()  # 'mean' to handle multi-gpu training
        avg_f1 += f1.mean().item()  # 'mean' to handle multi-gpu training
        avg_f2 += f2.mean().item()  # 'mean' to handle multi-gpu training
    cost = avg_loss

    """ Following 2 lines are temporary code. Remove in future. """
    # hit, _ = final_compute(xtrain.to(device))
    # hi[:, :, t] = hit.detach().cpu().numpy()

    # Remember lowest cost and save checkpoint
    is_best = cost < l_cost
    l_cost = min(cost, l_cost)
    dirs.save_checkpoint({
        'epochs': t,
        'vae': vae,
        'opt': opt,
        'vae_state_dict': vae.state_dict(),
        'optimizer': optimizer.state_dict(),
        'Loss_stk': Loss_stk,
    }, True)

    logging.info('Epoch {}/{}, Loss: [{}], D_kl: [{}], Recon: [{}]'.format(t+1, opt.max_epochs, cost, avg_f1, avg_f2))
    Loss_stk = np.append(Loss_stk, [[cost, avg_f1, avg_f2]], axis=0)
    t += 1

logging.info('Finished Training. Lowest cost: {}'
             '\nLoading best checkpoint [{}] & computing sub-space...'.format(l_cost, dirs.dircp))

# is_best = True
# dircp ='checkpoint.pth_20200312-1339.tar'

# Load the best model and compute the subspace
sd_mdl = torch.load('cp/{}/{}'.format(opt.dataset_name, dirs.dircp))
vae.load_state_dict(sd_mdl['vae_state_dict'])

z = vae_final_compute(model=vae, args=opt, ct=ct)
logging.info("\nTraining complete in: " + str(datetime.now() - start))

# Save Model and Tensors ==================================================================
torch.save({'vae': vae,
            'vae_state_dict': vae.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epochs': t - 1,
            'Loss_stk': Loss_stk,
            'opt': opt,
            # "hi": hi,
            'h': z}, 'out/{}/{}'.format(opt.dataset_name, dirs.dirout))
dirs.remove_checkpoint()
logging.info('\nSaved File: {}'.format(dirs.dirout))

