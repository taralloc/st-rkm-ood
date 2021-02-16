from utils import cnn_create_dirs
from stiefel_rkm_model import *
import logging
import argparse
import time
from datetime import datetime
from wrn_model import WideResNet

# Model Settings =================================================================================================

parser = argparse.ArgumentParser()

parser.add_argument('--dataset_name', type=str, default='cifar10',
                    help='Dataset name: mnist/fashion-mnist/svhn/dsprites/cifar10')
parser.add_argument('--capacity', type=int, default=40, help='Conv_filters of network')
parser.add_argument('--mb_size', type=int, default=128, help='Mini-batch size')
parser.add_argument('--x_fdim1', type=int, default=256, help='Input x_fdim1')
parser.add_argument('--x_fdim2', type=int, default=10, help='Number of classes')
parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint file')

# Training Settings =============================
parser.add_argument('--lr', type=float, default=1e-3, help='Input learning rate for ADAM optimizer')
parser.add_argument('--lrg', type=float, default=1e-4, help='Input learning rate for Cayley_ADAM optimizer')
parser.add_argument('--max_epochs', type=int, default=10, help='Input max_epoch')
parser.add_argument('--proc', type=str, default='cpu', help='device type: cuda or cpu')
parser.add_argument('--workers', type=int, default=0, help='Number of workers for dataloader')
parser.add_argument('--shuffle', type=bool, default=True, help='shuffle dataset: True/False')

opt = parser.parse_args()
# ==================================================================================================================

device = torch.device(opt.proc)

if torch.cuda.is_available():
    torch.cuda.empty_cache()

ct = time.strftime("%Y%m%d-%H%M")
dirs = cnn_create_dirs(name=opt.dataset_name, ct=ct)
dirs.create()

# noinspection PyArgumentList
logging.basicConfig(level=logging.INFO,
                    format='%(message)s',
                    handlers=[logging.FileHandler('log/{}/{}_Trained_cnn_{}.log'.format(opt.dataset_name, opt.dataset_name, ct)),
                              logging.StreamHandler()])

""" Load Training Data """

vars(opt)['train'] = True
xtrain, ipVec_dim, nChannels = get_dataloader(args=opt)

ngpus = torch.cuda.device_count()

logging.info(opt)
logging.info('\nN: {}, mb_size: {}'.format(len(xtrain.dataset), opt.mb_size))
logging.info('We are using {} GPU(s)!'.format(ngpus))

# Define params
cnn_kwargs = dict(kernel_size=4, stride=2, padding=1)
if ipVec_dim <= 28 * 28 * 3:
    cnn_kwargs = cnn_kwargs, dict(kernel_size=3, stride=1), 5
else:
    cnn_kwargs = cnn_kwargs, cnn_kwargs, 4
net = None
if opt.dataset_name == "fashion-mnist":
    net = Net1(nChannels=nChannels, args=opt, cnn_kwargs=cnn_kwargs).to(device)
elif opt.dataset_name == "cifar10":
    opt.x_fdim2 = 128
    net = WideResNet(depth=40, num_classes=10, widen_factor=2, dropRate=0.3).to(device)

optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr, weight_decay=0)
criterion = nn.CrossEntropyLoss()

# Train =========================================================================================================
start = datetime.now()
t = 0

# Load checkpoint
if opt.checkpoint is not None:
    checkpoint = torch.load("cp/" + opt.dataset_name + "/" + opt.checkpoint + ".tar")
    net.load_state_dict(checkpoint['cnn'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    t = checkpoint['epochs'] + 1
    print(f"Loaded checkpoint at epoch {t}")

while t < opt.max_epochs:
    running_loss = 0.0
    running_trainacc = 0.0
    running_valacc = 0.0
    for i, data in enumerate(tqdm(xtrain, desc="Epoch {}/{}".format(t, opt.max_epochs))):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        running_trainacc += torch.sum(labels == torch.max(outputs, dim=1)[1]) / float(inputs.shape[0])

    dirs.save_checkpoint({
        'epochs': t,
        'cnn': net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'opt': opt
    }, True)

    logging.info('Epoch {}/{}, Loss: [{}], Training Accuracy: [{}]'.format(t+1, opt.max_epochs, running_loss / i, running_trainacc / i))
    t += 1
logging.info("\nTraining complete in: " + str(datetime.now() - start))

# Save Model and Tensors ======================================================================================
torch.save({'cnn': net.state_dict(),
            'opt': opt}, 'out/{}/{}'.format(opt.dataset_name, dirs.dirout))
logging.info('\nSaved File: {}'.format(dirs.dirout))
