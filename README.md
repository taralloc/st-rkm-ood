# Unsupervised Energy-based Out-of-distribution Detection using Stiefel-Restricted Kernel Machine

## Abstract

Detecting out-of-distribution (OOD) inputs is an essential requirement for the deployment of machine learning systems in the real world. We propose an energy-based OOD detector trained with an unsupervised learning algorithm called Stiefel-Restricted Kernel Machine (St-RKM). Training requires minimizing an objective function with an autoencoder loss term and the RKM energy where the interconnection matrix lies on the Stiefel manifold. Further, we outline multiple energy function definitions based on the RKM framework and discuss their properties. In the experiments on standard datasets, our method outperforms other deep generative models and energy-based OOD detectors. Through several ablation studies, we further discuss the merit of each proposed energy function on the OOD detection performance.

## Code Structure

- Training a St-RKM model is done in `train_rkm.py`.
  - The architecture of St-RKM and the employed encoder and decoder are defined in `stiefel_rkm_model.py`.
- Training CNN, VAE, GAN and PCA models is done in `train_cnn.py`, `train_vae.py`, `train_gan.py` and `train_pca.py`.
- Evaluation of models is done in `rkm_gen.py`.

## Usage
### Download
First, navigate to the unzipped directory and install required python packages with the provided `environment.yml` file.

### Install packages in conda environment
Run the following in terminal. This will create a conda environment named *rkm_env*.

```
conda env create -f environment.yml
```

For overlapping coefficient computation , install the overlapping R package by running R (command `R`) and executing:

```R
install.packages("overlapping")
```

### Train

Activate the conda environment `conda activate rkm_env` and run one of the following commands, for example:
```
python train_rkm.py --dataset_name fashion-mnist --h_dim 10 --max_epochs 1600
python train_rkm.py --dataset_name cifar10 --h_dim 1024 --max_epochs 4000
```

The following options for training a St-RKM model are available:

```
usage: train_rkm.py [-h] [--dataset_name DATASET_NAME] [--arch ARCH]
                    [--h_dim H_DIM] [--capacity CAPACITY] [--mb_size MB_SIZE]
                    [--x_fdim1 X_FDIM1] [--x_fdim2 X_FDIM2] [--c_accu C_ACCU]
                    [--noise_level NOISE_LEVEL] [--loss LOSS]
                    [--checkpoint CHECKPOINT] [--lr LR] [--lrg LRG]
                    [--max_epochs MAX_EPOCHS] [--cutoff_perc CUTOFF_PERC]
                    [--proc PROC] [--workers WORKERS] [--shuffle SHUFFLE]
                    [--recon_loss {bce,mse}]

St-RKM Model

optional arguments:
  -h, --help            show this help message and exit
  --dataset_name DATASET_NAME
                        Dataset name: mnist/fashion-
                        mnist/svhn/dsprites/3dshapes/cars3d/cifar10/ecg5000
                        (default: cifar10)
  --h_dim H_DIM         Dim of latent vector (default: 1024)
  --capacity CAPACITY   Conv_filters of network (default: 64)
  --mb_size MB_SIZE     Mini-batch size (default: 128)
  --x_fdim1 X_FDIM1     Input x_fdim1 (default: 128)
  --x_fdim2 X_FDIM2     Input x_fdim2 (default: 64)
  --c_accu C_ACCU       Input weight on recons_error (default: 100)
  --noise_level NOISE_LEVEL
                        Noise-level (default: 0.001)
  --loss LOSS           loss type: deterministic/noisyU/splitloss (default:
                        deterministic)
  --checkpoint CHECKPOINT
                        Checkpoint file (default: None)
  --lr LR               Input learning rate for ADAM optimizer (default:
                        0.0002)
  --lrg LRG             Input learning rate for Cayley_ADAM optimizer
                        (default: 0.0001)
  --max_epochs MAX_EPOCHS
                        Input max_epoch (default: 5000)
  --proc PROC           device type: cuda or cpu (default: cuda)
  --workers WORKERS     Number of workers for dataloader (default: 16)
  --shuffle SHUFFLE     shuffle dataset: True/False (default: True)
  --recon_loss {bce,mse}
                        reconstruction loss (default: bce)
```
### Evaluate

To evaluate a given trained model `trained_model.tar` on some OOD datasets, run, for instance, one of the following commands:

```
python rkm_gen.py --dataset_name fashion-mnist --ood_dataset_names mnist dsprites svhn cifar10 --filename trained_model.tar
python rkm_gen.py --dataset_name cifar10 --ood_dataset_names mnist fashion-mnist svhn isun --filename trained_model.tar
```