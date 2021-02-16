import time
from pathlib import Path
import torchvision
from gan_model import Discriminator, Generator
from vae_model import *
from datetime import datetime
from utils import gan_create_dirs
import logging
from dataloader import *
import argparse
from vrae_model import VRAEEncoder, VRAEDecoder

""" Specify datatset name (mnist, fashion-mnist, cifar10, dsprites, celeba) and location. 
    Remember to change the architecture according to input size and ranges of input and output."""
#dataset_name = 'fashion-mnist'  # training dataset
#data_root = 'fashion-mnist'

if __name__ == '__main__':

    # Model Settings =================================================================================================
    parser = argparse.ArgumentParser(description='GAN Model',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset_name', type=str, default='cifar10',
                        help='Dataset name: mnist/fashion-mnist/svhn/dsprites/cifar10/ecg5000')
    parser.add_argument('--h_dim', type=int, default=1024, help='Dim of latent vector')
    parser.add_argument('--capacity', type=int, default=64, help='Capacity of network. See utils.py')
    parser.add_argument('--mb_size', type=int, default=128, help='Mini-batch size. See utils.py')
    parser.add_argument('--x_fdim1', type=int, default=3000, help='Input x_fdim1. See utils.py')
    parser.add_argument('--x_fdim2', type=int, default=1500, help='Input x_fdim2. See utils.py')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint file')

    # Training Settings =============================
    parser.add_argument('--lr', type=float, default=2e-4, help='Input learning rate for optimizer')
    parser.add_argument('--epoch_step', default=[800, 7000], type=int, help='List with epochs to drop lr on')
    parser.add_argument('--lr_decay_ratio', default=0.2, type=float)
    parser.add_argument('--max_epochs', type=int, default=40, help='Input max_epoch for cut-off')
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
    dirs = gan_create_dirs(name=opt.dataset_name, ct=ct)
    dirs.create()

    # noinspection PyArgumentList
    logging.basicConfig(level=logging.INFO,
                        format='%(message)s',
                        handlers=[logging.FileHandler('log/{}/{}_Trained_gan_{}.log'.format(opt.dataset_name, opt.dataset_name, ct)),
                                  logging.StreamHandler()])

    """ Load Training Data """
    vars(opt)['train'] = True
    # image_size = 64
    # vars(opt)['pre_transformations'] = [transforms.Resize(image_size), transforms.CenterCrop(image_size)]
    xtrain, ipVec_dim, nChannels = get_dataloader(args=opt)

    ngpus = torch.cuda.device_count()

    Encoder, Decoder = None, None
    if opt.dataset_name == "fashion-mnist":
        Encoder, Decoder = Net1, Net3
    elif opt.dataset_name == "cifar10":
        Encoder, Decoder = Net2, Net4
    elif opt.dataset_name == "ecg5000":
        Encoder, Decoder = VRAEEncoder, VRAEDecoder
    netD = Discriminator(ipVec_dim=ipVec_dim, args=opt, nChannels=nChannels, Encoder=Encoder).to(device)
    netG = Generator(ipVec_dim=ipVec_dim, args=opt, nChannels=nChannels, Decoder=Decoder).to(device)
    logging.info(netD)
    logging.info(netG)
    logging.info(opt)
    logging.info('\nN: {}, mb_size: {}'.format(len(xtrain.dataset), opt.mb_size))
    logging.info('We are using {} GPU(s)!'.format(ngpus))
    logging.info('The number of parameters of Discriminator: {}'.format(sum(p.numel() for p in netD.parameters() if p.requires_grad)))
    logging.info('The number of parameters of Generator: {}'.format(sum(p.numel() for p in netG.parameters() if p.requires_grad)))

    # Setup Adam optimizers for both G and D
    optimizerD = torch.optim.Adam(netD.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizerG = torch.optim.Adam(netG.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    # Train =========================================================================================================
    # Adapted from https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

    t = 1

    # Load checkpoint
    if opt.checkpoint is not None:
        checkpoint = Path("cp/" + opt.dataset_name + "/" + opt.checkpoint + ".tar")
        if not checkpoint.exists():
            checkpoint = Path("out/" + opt.dataset_name + "/" + opt.checkpoint + ".tar")
        checkpoint = torch.load(str(checkpoint))
        netD.load_state_dict(checkpoint['discriminator_state_dict'])
        netG.load_state_dict(checkpoint['generator_state_dict'])
        optimizerD.load_state_dict(checkpoint['optimizerD'])
        optimizerG.load_state_dict(checkpoint['optimizerG'])
        t = checkpoint.get('epochs', 50) + 1
        print(f"Loaded checkpoint at epoch {t}")

    start = datetime.now()

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(opt.mb_size, opt.h_dim, 1, 1, device=device)

    # Initialize criterion function
    criterion = None
    if opt.recon_loss == 'bce':
        criterion = nn.BCELoss()
    elif opt.recon_loss == 'mse':
        criterion = nn.MSELoss()

    # For each epoch
    for epoch in range(t, opt.max_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(xtrain, 0):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, opt.h_dim, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 1 == 0:
                logging.info('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, opt.max_epochs, i, len(xtrain),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == opt.max_epochs-1) and (i == len(xtrain)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(torchvision.utils.make_grid(fake, padding=2, normalize=True))

            iters += 1

        # Remember lowest cost and save checkpoint
        dirs.save_checkpoint({
            'epochs': epoch,
            'opt': opt,
            'discriminator_state_dict': netD.state_dict(),
            'generator_state_dict': netG.state_dict(),
            'optimizerD': optimizerD.state_dict(),
            'optimizerG': optimizerG.state_dict(),
        }, True)

    logging.info("\nTraining complete in: " + str(datetime.now() - start))

    # Save Model and Tensors ==================================================================
    torch.save({'discriminator_state_dict': netD.state_dict(),
                'generator_state_dict': netG.state_dict(),
                'optimizerD': optimizerD.state_dict(),
                'optimizerG': optimizerG.state_dict(),
                # 'Loss_stk': Loss_stk,
                'opt': opt,
                # "hi": hi,
                'img_list': img_list,
                'G_losses': G_losses,
                'D_losses': D_losses,
                'epochs': epoch - 1,
                }, 'out/{}/{}'.format(opt.dataset_name, dirs.dirout))
    dirs.remove_checkpoint()
    logging.info('\nSaved File: {}'.format(dirs.dirout))

