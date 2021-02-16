import copy
import zipfile
from functools import partial
from aenum import Enum
from typing import List, Tuple, Callable
import pandas
import sklearn
from gan_model import Discriminator, Generator
from utils import eval_clustering, dataset_mean, dataset_std, datasetrgb_mean, datasetrgb_std
from vae_model import VAE
from wrn_model import WideResNet
from torchvision import transforms
import time
import utils
import torch
import argparse
from dataloader import *
from stiefel_rkm_model import final_compute, compute_ot, Net1, Net2, Net3, Net4
from plotnine import *
from vrae_model import VRAEEncoder, VRAEDecoder


class Method(Enum):
    _init_ = 'value string'

    FULL_ENERGY = 0, 'Full\nEnergy'
    ENERGY_BUT_LOSS = 1, 'kPCA\nError'
    LOSS = 2, 'AELoss'
    CORR = 3, 'corr'
    NEG_CORR = 4, 'NegCorr'
    LIU2020A = 5, 'Liu2020'
    LIU2020A_COLOR = 7, 'liu2020_wrn'
    PCA = 14, 'PCA'
    VAE = 15, 'VAE'
    GAN = 16, 'GAN'

    def __str__(self):
        return self.string

def strkm_ood(x, rkm, ot_train_mean, method) -> np.ndarray:
    return rkm.compute_pointwise_energy(x, mean=ot_train_mean, method=method).detach().cpu().numpy()

def liu2020a(x, net: Net1, T=1.0) -> np.ndarray:
    torch.manual_seed(0)
    with torch.no_grad():
        logits = net(x)
        # logits = logits - torch.max(logits, dim=1, keepdim=True)[0] # just some experiment
        return -(T*torch.logsumexp(logits / T, dim=1)).cpu().numpy()

def pca_ood(mean, eigvecs, x) -> np.ndarray:
    x = x.view(x.shape[0], -1)
    x_mean = x - mean
    rec = torch.mm(torch.mm(x_mean, eigvecs.t()), eigvecs) + mean
    return torch.pow(torch.norm(rec-x, dim=1), 2).cpu().numpy()

def vae_ood(vae: VAE, x) -> np.ndarray:
    torch.manual_seed(0)
    with torch.no_grad():
        rec = vae.compute_pointwise_energy(x)
        assert (rec >= 0).all()
        return rec.cpu().numpy()

def gan_ood(x, netD: Discriminator, netG: Generator, h_dim, device, lambd=0.0, seed=0) -> np.ndarray:
    torch.manual_seed(seed)
    with torch.no_grad():
        N = x.shape[0]
        z = torch.randn(N, h_dim, 1, 1, device=device)
        f1 = torch.log(netD(x)).cpu().numpy().flatten()
        f2 = torch.norm(x.view(N,-1)-netG(z).view(N,-1), dim=1).cpu().numpy()
        return -(1-lambd) * f1 + lambd * f2

def ood_compute_scores(xtrain: torch.utils.data.DataLoader, funs: List[Callable]):
    N = len(xtrain.dataset)
    energies = {fun: np.empty(N) for fun in funs}
    batch_size = xtrain.batch_size
    device = "cuda"
    for i, data in enumerate(xtrain):
        inputs = data[0].to(device)
        for fun in funs:
            energies_temp = fun(inputs)
            energies[fun][i*batch_size:i*batch_size+inputs.shape[0]] = energies_temp
    return energies #key: fun, value: energy vector

def ood_eval(input: List[Tuple[str, List[Tuple[torch.utils.data.DataLoader, Callable, bool]]]], sensitivity_analysis=False) -> Tuple[pandas.DataFrame, pandas.DataFrame]:
    """
    Main function for computing scores.
    Input is a list of (training_dataset_name, [(dataloader for ood dataset, fun, in?)]),
    whwre fun is the function that given a test point x returns its score,
    and in? is True if the ood dataset = training dataset, else False.
    Note that fun has an attribute method, which is a number corresponding to the Method enum.
    """
    assert len(set([name for name, _, indistri in input if indistri])) == 1
    pos_scores = {} # key: fun, value: energy vector
    neg_scores = {} # key: dataset name, value: {key: fun, value: energy vector}
    table1 = []
    energy_table = pandas.DataFrame()
    train_dataset_name = [name for name, _, indistri in input if indistri][0]
    for name, evals, indistri in input:
        for xtrain, funs in evals:
            energies = ood_compute_scores(xtrain, funs)
            if indistri:
                pos_scores = utils.merge_two_dicts(pos_scores, energies)
            else:
                if name not in neg_scores:
                    neg_scores[name] = {}
                neg_scores[name] = utils.merge_two_dicts(neg_scores[name], energies)
    # Finally, evaluate classifier for each ood dataset and for each method
    for ood_dataset_name, neg_scores in neg_scores.items():
        for fun, neg_scores_method in neg_scores.items():
            method = fun.method
            auroc, aupr, fpr = utils.get_measures(-np.nan_to_num(pos_scores[fun]), -np.nan_to_num(neg_scores_method))
            overlap = utils.get_overlap(pos_scores[fun], neg_scores_method)
            mmd = utils.get_mmd(pos_scores[fun], neg_scores_method)
            wd = utils.get_wd(pos_scores[fun], neg_scores_method)
            table1_row = {'d_in': train_dataset_name, 'd_out': ood_dataset_name, 'method': method,
                           'fpr95': fpr, 'auroc': auroc, 'aupr': aupr,
                          'overlap': overlap, 'mmd': mmd, 'wd': wd}
            energytable_row = {'energy': np.concatenate([pos_scores[fun], neg_scores_method]),
                               'method': np.tile(method, len(pos_scores[fun]) + len(neg_scores_method)),
                               'out_dataset': np.tile(ood_dataset_name, len(pos_scores[fun]) + len(neg_scores_method)),
                               'in?': np.concatenate([np.tile(True, len(pos_scores[fun])), np.tile(False, len(neg_scores_method))]),
                               'label': fun.label}
            if sensitivity_analysis:
                table1_row['lambda'] = fun.lambd
                energytable_row['lambda'] = fun.lambd
            table1.append(table1_row)
            energy_table = energy_table.append(pandas.DataFrame(energytable_row), ignore_index=True)
    table1 = pandas.DataFrame(table1)
    return table1, energy_table


def save_table1(table: pandas.DataFrame, out_file: str):
    """
    Saves table 1 in a format suitable for the latex table.
    """
    table = table.copy(deep=True)
    datasets = table['d_out'].unique()

    def to_perc(table, col, k=100.0):
        table[col] = table[col].astype('float')
        table[col] = table[col].map(lambda x: '%.3f' % (k * x))
        return table

    table = to_perc(table, 'fpr95_mean')
    table = to_perc(table, 'fpr95_std')
    table = to_perc(table, 'auroc_mean')
    table = to_perc(table, 'auroc_std')
    table = to_perc(table, 'aupr_mean')
    table = to_perc(table, 'aupr_std')
    table = to_perc(table, 'overlap_mean', k=1)
    table = to_perc(table, 'overlap_std', k=1)
    table = to_perc(table, 'mmd_mean', k=1)
    table = to_perc(table, 'mmd_std', k=1)
    table = to_perc(table, 'wd_mean', k=1)
    table = to_perc(table, 'wd_std', k=1)


    # Merge mean and std
    fpr95 = table.xs('fpr95_mean', axis=1).astype(str) + ' +- ' + table.xs('fpr95_std', axis=1).astype(str)
    auroc = table.xs('auroc_mean', axis=1).astype(str) + ' +- ' + table.xs('auroc_std', axis=1).astype(str)
    aupr = table.xs('aupr_mean', axis=1).astype(str) + ' +- ' + table.xs('aupr_std', axis=1).astype(str)
    overlap = table.xs('overlap_mean', axis=1).astype(str) + ' +- ' + table.xs('overlap_std', axis=1).astype(str)
    mmd = table.xs('mmd_mean', axis=1).astype(str) + ' +- ' + table.xs('mmd_std', axis=1).astype(str)
    wd = table.xs('wd_mean', axis=1).astype(str) + ' +- ' + table.xs('wd_std', axis=1).astype(str)
    table['fpr95'] = fpr95
    table['auroc'] = auroc
    table['aupr'] = aupr
    table['overlap'] = overlap
    table['mmd'] = mmd
    table['wd'] = wd
    select = ['d_in', 'd_out', 'method', 'fpr95', 'auroc', 'aupr'] +\
             (['overlap'] if 'overlap' in table.columns else []) + \
             (['mmd'] if 'mmd' in table.columns else []) + \
             (['wd'] if 'wd' in table.columns else []) + \
             (['lambda'] if 'lambda' in table.columns else [])
    table = table[select]

    with zipfile.ZipFile(out_file % ("horizontaltable", "zip"), "w", zipfile.ZIP_DEFLATED) as zipf:
        for dataset in datasets:
            table[table['d_out'] == dataset].transpose().to_csv("table_%s.csv" % dataset, float_format="%.3f")
            zipf.write("table_%s.csv" % dataset)
            os.remove("table_%s.csv" % dataset)


def beautify_table1(table: pandas.DataFrame):
    sensitivity_analysis = 'lambda' in table.columns
    groupby = ['d_in', 'd_out', 'method'] + (['lambda'] if sensitivity_analysis else [])
    table = table.groupby(groupby, as_index=False). \
        agg({'fpr95': ['mean', 'std'], 'auroc': ['mean', 'std'], 'aupr': ['mean', 'std'], 'overlap': ['mean', 'std'],
             'mmd': ['mean', 'std'], 'wd': ['mean', 'std']})
    table.columns = table.columns.map(lambda x: '_'.join(x) if x[1] != '' else x[0])
    table = table.fillna(0.0)

    # Rename method
    table['method'] = table['method'].astype('category')
    table['method'] = table['method'].cat.rename_categories(dict(list(enumerate(map(str, list(Method))))))
    # Rename datasets
    rename_dict = {'mnist': "MNIST", 'svhn': "SVHN", "cifar10": "CIFAR-10", "dsprites": "dSprites", "fashion-minst": "Fashion-MNIST", "isun": "iSUN"}
    table['d_in'], table['d_out'] = table['d_in'].astype('category'), table['d_out'].astype('category')
    table['d_in'], table['d_out'] = table['d_in'].cat.rename_categories(rename_dict), table['d_out'].cat.rename_categories(rename_dict)

    return table

def beautify_energytable(energytable: pandas.DataFrame):
    # Preprocess energies
    methods, ood_dataset_names, labels = energytable['method'].unique(), energytable['out_dataset'].unique(), energytable['label'].unique()
    for method, ood_dataset_name, label in zip(methods, ood_dataset_names, labels):
        selector = (energytable['method'] == method) & (energytable['out_dataset'] == ood_dataset_name)\
                   & (energytable['label'] == label)
        if len(energytable[selector]) > 0:
            pos_scores = energytable[selector & (energytable['in?'] == True)]['energy']
            neg_scores = energytable[selector & (energytable['in?'] == False)]['energy']
            x = sklearn.preprocessing.scale(np.concatenate([pos_scores, neg_scores]))
            energytable.loc[selector & (energytable['in?'] == True), 'energy'] = x[:len(pos_scores)]
            energytable.loc[selector & (energytable['in?'] == False), 'energy'] = x[len(pos_scores):]
    # Reorder
    energytable['out_dataset'] = energytable['out_dataset'].astype('category')
    target_order = ['mnist', 'dsprites', 'svhn', 'cifar10', 'fashion-mnist', 'isun']
    actual_order = energytable['out_dataset'].unique()
    energytable['out_dataset'] = energytable['out_dataset'].cat.reorder_categories([dataset for dataset in target_order if dataset in actual_order])
    energytable['method'] = energytable['method'].astype('category')
    target_order = [0,1,2,3,4,5,14,15,16]
    actual_order = energytable['method'].unique()
    energytable['method'] = energytable['method'].cat.reorder_categories([method for method in target_order if method in actual_order])
    # Rename method
    energytable['method'] = energytable['method'].astype('category')
    energytable['method'] = energytable['method'].cat.rename_categories(dict(list(enumerate(map(str, list(Method))))))
    # Rename datasets
    rename_dict = {'mnist': "MNIST", 'svhn': "SVHN", "cifar10": "CIFAR-10", "dsprites": "dSprites", "fashion-mnist": "Fashion-MNIST", "isun": "iSUN"}
    energytable['out_dataset'] = energytable['out_dataset'].astype('category')
    energytable['out_dataset'] = energytable['out_dataset'].cat.rename_categories(rename_dict)

    energytable['in?'] = energytable['in?'].astype('bool')
    return energytable


def plot1(energytable: pandas.DataFrame) -> ggplot:
    # works with plotine 0.6.0, matplotlib 3.1.1
    # Query for centered visualization
    energytable = energytable[(energytable['out_dataset'] == 'mnist') & (energytable['energy'] > -5) & (energytable['energy'] < 5) |
                              (energytable['out_dataset'] == 'svhn') & (energytable['energy'] > -7) & (energytable['energy'] < 7) |
                              (energytable['out_dataset'] == 'dsprites') & (energytable['energy'] > -3.5) & (energytable['energy'] < 3) |
                              (energytable['out_dataset'] == 'cifar10') & (energytable['energy'] > -3) & (energytable['energy'] < 3) |
                              (energytable['out_dataset'] == 'fashion-mnist') & (energytable['energy'] > -100) & (energytable['energy'] < 100) |
                              (energytable['out_dataset'] == 'isun') & (energytable['energy'] > -1e10) & (energytable['energy'] < 1e10)]
    energytable = energytable[energytable['method'].isin([0, 1, 2, 3, 4, 5, 14, 15, 16])]

    # Make all energies positive because of https://github.com/matplotlib/matplotlib/issues/17007
    minima = [(out_dataset, np.min(energytable[energytable['out_dataset'] == out_dataset]['energy'])) for out_dataset in energytable['out_dataset'].unique()]
    for out_dataset, minimum in minima:
        energytable.loc[energytable["out_dataset"] == out_dataset, ["energy"]] += -minimum

    energytable = beautify_energytable(energytable)

    if len(energytable['out_dataset'].unique()) == 1 and energytable['out_dataset'].unique()[0] == "Fashion-MNIST":
        #Figure 3
        return ggplot(energytable, aes(x='energy', color='in?', fill='in?')) + \
               geom_density(alpha=0.1) + \
               coord_cartesian(xlim=(0.25, 7.25)) + \
               facet_grid('out_dataset ~ method', scales='free') + \
               theme(figure_size=(6.5, 3), strip_text=element_text(margin={'r': 8, 'l': 7, 't': 4, 'b': 4, 'units': 'pt'})) + \
               labs(x="Energy", y="Density", color="in-distribution?", fill="in-distribution?") + \
               theme(text=element_text(family="CMU Serif"))  # font from https://www.fontsquirrel.com/fonts/computer-modern
    else:
        #Figure 4
        return ggplot(energytable, aes(x='energy', color='in?', fill='in?')) + \
               geom_density(alpha=0.1) + \
               scale_y_log10() + \
               facet_grid('method ~ out_dataset', scales='free') + \
               theme(figure_size=(6.2, 6), strip_text=element_text(margin={'r': 8, 'l': 7, 't': 4, 'b': 4, 'units': 'pt'})) + \
               labs(x="Energy", y="Density", color="in-distribution?", fill="in-distribution?") + \
               theme(text=element_text(family="CMU Serif"))  # font from https://www.fontsquirrel.com/fonts/computer-modern

def plot_outlier(energytable: pandas.DataFrame, train_dataset_name, ood_dataset_names, opt, out_file, method=Method.ENERGY_BUT_LOSS, n=50):
    # n is number of images to plot

    def save_images(outlier_in, outlier_out, outlier_in_labels, outlier_out_labels, ood_dataset_name):
        from PIL import Image
        if outlier_out is None:
            outlier_out = torch.tensor([])
        if outlier_in is None:
            outlier_in = torch.tensor([])
        if type(outlier_out) != torch.Tensor:
            outlier_out = torch.from_numpy(outlier_out)
        with zipfile.ZipFile(out_file % (f"outlierimages_{ood_dataset_name}", "zip"), "w", zipfile.ZIP_DEFLATED) as zipf:
            for i in range(outlier_in.shape[0]):
                img = Image.fromarray((outlier_in[i]).astype('uint8'), 'RGB')
                name = "%s%02d-%02d_%s.png" % ("in", i, outlier_in_labels[i], train_dataset_name)
                img.save(name)
                zipf.write(name)
                os.remove(name)
            for i in range(outlier_out.shape[0]):
                if outlier_out.shape[1] == 3 or outlier_out.shape[3] == 3:

                    img = Image.fromarray((outlier_out[i].detach().numpy()).astype('uint8'), 'RGB')
                else:
                    img = Image.fromarray((outlier_out[i].detach().numpy()).astype('uint8'), 'L')
                name = "%s%02d-%02d_%s.png" % ("out", i, outlier_out_labels[i], ood_dataset_name)
                img.save(name)
                zipf.write(name)
                os.remove(name)

    myopt = copy.deepcopy(opt)
    method = method.value
    energytable = energytable[energytable['method'] == method]

    for ood_dataset_name in ood_dataset_names:
        assert ood_dataset_name in energytable['out_dataset'].unique()
        sensitivity_analysis = ood_dataset_name == train_dataset_name
        energytable_single_ood_dataset = energytable[energytable['out_dataset'] == ood_dataset_name]
        en_minout = energytable_single_ood_dataset[energytable_single_ood_dataset['in?'] == False]['energy'].min()
        en_maxin = energytable_single_ood_dataset[energytable_single_ood_dataset['in?'] == True]['energy'].max()
        assert en_minout < en_maxin

        outlier_in, outlier_out = None, None
        outlier_in_labels, outlier_out_labels = None, None

        if not sensitivity_analysis:
            energytable_in = energytable_single_ood_dataset[energytable_single_ood_dataset['in?'] == True]
            en_n_in = np.sort(energytable_in['energy'].to_numpy())[-n]
            mask_in = np.logical_and((energytable_in['energy'] >= en_n_in).to_numpy(), (energytable_in['energy'] < en_maxin).to_numpy())
            vars(myopt)["train"] = False
            vars(myopt)["dataset_name"] = train_dataset_name
            xtrainin, _, _ = get_dataloader(args=myopt)
            outlier_in = xtrainin.dataset.data[mask_in]
            outlier_in_labels = np.array(xtrainin.dataset.targets)[mask_in]
            assert outlier_in.shape[0] <= n

        energytable_out = energytable_single_ood_dataset[energytable_single_ood_dataset['in?'] == False]
        if not sensitivity_analysis:
            en_n_out = np.sort(energytable_out['energy'].to_numpy())[n]
            mask_out = np.logical_and((energytable_out['energy'] >= en_minout).to_numpy(), (energytable_out['energy'] < en_n_out).to_numpy())
        else:
            en_n_out = np.sort(energytable_out['energy'].to_numpy())[-n]
            mask_out = (energytable_out['energy'] >= en_n_out).to_numpy()
        vars(myopt)["train"] = False
        vars(myopt)["dataset_name"] = ood_dataset_name
        xtrainout, _, _ = get_dataloader(args=myopt)
        outlier_out = xtrainout.dataset.data[mask_out]
        try:
            outlier_out_labels = np.array(xtrainout.dataset.targets)[mask_out]
        except:
            outlier_out_labels = xtrainout.dataset.labels[mask_out]
        assert outlier_out.shape[0] <= n
        save_images(outlier_in, outlier_out, outlier_in_labels, outlier_out_labels, ood_dataset_name)

if __name__ == '__main__':
    # Enter path to load the pre-trained model (see README.md file)
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--filename', nargs="+", type=str, default=['trained_model'], help='Enter Filename')
    parser.add_argument('--dataset_name', default='fashion-mnist', type=str, help='Training dataset name')
    parser.add_argument('--ood_dataset_names', nargs="*", default=["mnist", "svhn", "cifar10", "dsprites"], type=str, help='OOD datasets names')
    parser.add_argument('--filename_cnn', nargs="*", type=str, default=[], help='Enter Filename')
    parser.add_argument('--filename_pca', nargs="*", type=str, default=[], help='Enter Filename')
    parser.add_argument('--filename_vae', nargs="*", type=str, default=[], help='Enter Filename')
    parser.add_argument('--filename_gan', nargs="*", type=str, default=[], help='Enter Filename')
    parser.add_argument("--sanity_check", help="Tests on in-distribution dataset", action="store_true")
    parser.add_argument('--filename_energytable', type=str, default=None, help='Path to energytable to restore evaluation')
    parser.add_argument('--filename_table', type=str, default=None, help='Path to table to restore evaluation')
    parser.add_argument("--sens_analysis", help="Sensitivity analysis on lambda", action="store_true")
    parser.add_argument("--visualize_outlier", help="Visualize outlier. To use with filename_energytable.", action="store_true")
    opt_gen = parser.parse_args()
    if opt_gen.sanity_check:
        opt_gen.ood_dataset_names = [opt_gen.dataset_name]  # if sanity check, ood dataset is training dataset
    if opt_gen.sens_analysis:
        opt_gen.filename_cnn = []
        opt_gen.filename_pca = []
        opt_gen.filename_vae = []
        opt_gen.filename_gan = []

    # Define model architecture
    sd_mdl = torch.load('out/{}/{}.tar'.format(opt_gen.dataset_name, opt_gen.filename[0]),
                        map_location=torch.device('cpu'))
    opt = sd_mdl['opt']
    Encoder, Decoder = None, None
    if opt_gen.dataset_name in ["fashion-mnist", "mnist"]:
        Encoder, Decoder = Net1, Net3
    elif opt_gen.dataset_name == "cifar10":
        Encoder, Decoder = Net2, Net4
    elif opt_gen.dataset_name in ["ecg5000"]:
        Encoder, Decoder = VRAEEncoder, VRAEDecoder
    opt.workers = 16
    opt.shuffle = False
    opt.proc = "cuda"
    opt.__dict__.pop("dataset_name")
    vars(opt)['train'] = False  # selects test sets
    opt = argparse.Namespace(**vars(opt), **vars(opt_gen))
    xtrain, ipVec_dim, nChannels = get_dataloader(args=opt)

    def my_get_dataloader(train_dataset_name, dataset_name, method=0, _in=False):
        assert not _in or train_dataset_name == dataset_name
        myopt = copy.deepcopy(opt)
        myopt.dataset_name = dataset_name
        myopt.train = True if (_in and opt_gen.sanity_check) else False
        if train_dataset_name != "ecg5000":
            if method == 6:
                # Normalize dataset for LIU2020A_NORM method
                vars(myopt)["post_transformations"] += [transforms.Normalize([dataset_mean[train_dataset_name]], [dataset_std[train_dataset_name]])]
            elif method == 7:
                vars(myopt)["pre_transformations"] += [utils.ChannelTransform(rgb=True), transforms.Resize(32)]
                vars(myopt)["post_transformations"] += [transforms.Normalize(datasetrgb_mean[train_dataset_name], datasetrgb_std[train_dataset_name])]
            else:
                if train_dataset_name in ["fashion-mnist", "mnist"]:
                    # Transform all datasets to grayscale and resize to 28x28
                    vars(myopt)["pre_transformations"] += [utils.ChannelTransform(rgb=False), transforms.Resize(28)]
                elif train_dataset_name == "cifar10":
                    # Transform all datasets to rgb and resize to 32x32
                    vars(myopt)["pre_transformations"] += [utils.ChannelTransform(rgb=True), transforms.Resize(32)]
        x, _, _ = get_dataloader(args=myopt)
        return x

    if opt_gen.visualize:
        U = sd_mdl.get('U', None)
        h = sd_mdl.get('h', None)
        rkm = sd_mdl['rkm'].to(opt.proc)
        if h is None and U is None:
            h, U, _, ot_train_mean = final_compute(model=rkm, args=opt, ct="a", device=opt.proc)
        U = U.to(opt.proc)
        ot_train_mean = sd_mdl.get('ot_mean', None)
        if ot_train_mean is None:
            _, ot_train_mean = compute_ot(rkm, opt, "a", train_mean=ot_train_mean)

        visualize_model(opt, rkm, U, ot_train_mean, xtrain, nChannels)
        exit()

    if opt_gen.filename_energytable is None and opt_gen.filename_table is None:
        # Load St-RKM
        def load_rkm(filename_rkm):
            sd_mdl = torch.load('out/{}/{}.tar'.format(opt_gen.dataset_name, filename_rkm),
                                map_location=torch.device('cpu'))
            rkm = sd_mdl['rkm']
            rkm.encoder = rkm.encoder.eval()
            rkm.decoder = rkm.decoder.eval()
            opt = sd_mdl['opt']
            ot_train_mean = sd_mdl.get('ot_mean', None)
            opt.workers = 0 if os.name == 'nt' else 16  # 0 workers for Windows
            opt.shuffle = False
            opt.proc = "cpu" if os.name == 'nt' else "cuda"  # cpu for Windows
            if type(rkm.decoder) == VRAEDecoder:
                rkm.decoder.myto(opt.proc)
            opt_dict = vars(opt)
            opt_dict.pop("dataset_name")
            opt = argparse.Namespace(**opt_dict, **vars(opt_gen))
            vars(opt)['train'] = False  # selects test sets
            ct = time.strftime("%Y%m%d-%H%M")
            if ot_train_mean is None:
                _, ot_train_mean = compute_ot(rkm, opt, ct, train_mean=ot_train_mean)
            ot_train_mean = ot_train_mean.to(opt.proc)
            rkm = rkm.to(opt.proc)
            def strkm_fun_factory(method):
                strkm_fun = partial(strkm_ood, rkm=rkm, ot_train_mean=ot_train_mean, method=method)
                strkm_fun.method = Method(method).value
                strkm_fun.lambd = opt.c_accu
                strkm_fun.label = filename_rkm
                return strkm_fun
            strkm_funs = [strkm_fun_factory(method) for method in [0,1,2,3,4]]
            return strkm_funs
        rkm_funs = sum([load_rkm(filename) for filename in opt_gen.filename], [])

        # Load comparison model that liu2020a shared on GitHub
        sd_mdl_cmp_liu2020a = torch.load('out/liu2020a/cifar10_wrn_pretrained_epoch_99.pt', map_location=torch.device('cpu'))
        net_liu2020a = WideResNet(depth=40, num_classes=10, widen_factor=2, dropRate=0.3)
        net_liu2020a.load_state_dict(sd_mdl_cmp_liu2020a)
        net_liu2020a.eval()
        net_liu2020a = net_liu2020a.to(opt.proc)
        liu2020_7_fun = partial(liu2020a, net=net_liu2020a)
        liu2020_7_fun.method = Method.LIU2020A_COLOR.value
        liu2020_7_fun.label = "cifar10_wrn_pretrained_epoch_99"

        # Load our pre-trained model for liu2020a
        def load_cnn(filename_cnn):
            sd_mdl_cmp = torch.load('out/{}/{}.tar'.format(opt_gen.dataset_name, filename_cnn),
                                map_location=lambda storage, loc: storage)
            cnn_kwargs = dict(kernel_size=4, stride=2, padding=1)
            if ipVec_dim <= 28 * 28 * 3:
                cnn_kwargs = cnn_kwargs, dict(kernel_size=3, stride=1), 5
            else:
                cnn_kwargs = cnn_kwargs, cnn_kwargs, 4
            net = Encoder(nChannels=nChannels, args=sd_mdl_cmp['opt'], cnn_kwargs=cnn_kwargs)
            net.load_state_dict(sd_mdl_cmp['cnn'])
            net.eval()
            net = net.to(opt.proc)
            liu2020_5_fun = partial(liu2020a, net=net)
            liu2020_5_fun.method = Method.LIU2020A.value
            liu2020_5_fun.label = filename_cnn
            return liu2020_5_fun
        cnn_funs = [load_cnn(filename) for filename in opt_gen.filename_cnn]

        # Load PCA model
        def load_pca(filename_pca):
            sd_mdl_pca = torch.load('out/{}/{}.tar'.format(opt_gen.dataset_name, filename_pca), map_location=torch.device('cpu'))
            pca_fun = partial(pca_ood, sd_mdl_pca['mean'].to(opt.proc), sd_mdl_pca['eigvec'].to(opt.proc))
            pca_fun.method = Method.PCA.value
            pca_fun.label = filename_pca
            return pca_fun
        pca_funs = [load_pca(filename) for filename in opt_gen.filename_pca]

        # Load VAE model
        def load_vae(filename_vae):
            sd_mdl_vae = torch.load('out/{}/{}.tar'.format(opt_gen.dataset_name, filename_vae), map_location=torch.device('cpu'))
            if "recon_loss" not in vars(sd_mdl_vae['opt']):
                vars(sd_mdl_vae['opt'])['recon_loss'] = "bce"
            vae = VAE(ipVec_dim=ipVec_dim, args=sd_mdl_vae['opt'], nChannels=nChannels, recon_loss=sd_mdl_vae['opt'].recon_loss, Encoder=Encoder, Decoder=Decoder)
            vae.load_state_dict(sd_mdl_vae['vae_state_dict'])
            vae.eval()
            vae = vae.to(opt.proc)
            vae_fun = partial(vae_ood, vae)
            vae_fun.method = Method.VAE.value
            vae_fun.label = filename_vae
            return vae_fun
        vae_funs = [load_vae(filename) for filename in opt_gen.filename_vae]

        # Load GAN model
        def load_gan(filename_gan):
            sd_mdl_gan = torch.load('out/{}/{}.tar'.format(opt_gen.dataset_name, filename_gan), map_location=torch.device('cpu'))
            netD = Discriminator(ipVec_dim=ipVec_dim, args=opt, nChannels=nChannels, Encoder=Encoder).to(opt.proc)
            netG = Generator(ipVec_dim=ipVec_dim, args=opt, nChannels=nChannels, Decoder=Decoder).to(opt.proc)
            netD.load_state_dict(sd_mdl_gan['discriminator_state_dict'])
            netD.eval()
            netD = netD.to(opt.proc)
            netG.load_state_dict(sd_mdl_gan['generator_state_dict'])
            netG.eval()
            netG = netG.to(opt.proc)
            gan_fun = partial(gan_ood, netD=netD, netG=netG, h_dim=opt.h_dim, device=opt.proc, lambd=0.2)
            gan_funs = [partial(gan_fun, seed=seed) for seed in range(5)] #repeat evaluation 5 times
            for gan_fun in gan_funs:
                gan_fun.method = Method.GAN.value
                gan_fun.label = filename_gan
            return gan_funs
        gan_funs = sum([load_gan(filename) for filename in opt_gen.filename_gan], [])

        # Define input for OOD detection evaluation
        funs_0 = rkm_funs + pca_funs + vae_funs + cnn_funs + gan_funs
        funs_1 = [liu2020_7_fun] if opt_gen.dataset_name not in ["ecg5000"] and not opt_gen.sens_analysis else []
        in_dataset = [(opt_gen.dataset_name, [(my_get_dataloader(opt_gen.dataset_name, opt_gen.dataset_name, _in=True), funs_0)], True)]
        if funs_1:
            in_dataset += [(opt_gen.dataset_name, [(my_get_dataloader(opt_gen.dataset_name, opt_gen.dataset_name, 7, _in=True), funs_1)], True)]
        ood_datasets = [(ood_dataset_name, [(my_get_dataloader(opt_gen.dataset_name, ood_dataset_name), funs_0)], False)
                        for ood_dataset_name in opt_gen.ood_dataset_names]
        if funs_1:
            ood_datasets += [(ood_dataset_name, [(my_get_dataloader(opt_gen.dataset_name, ood_dataset_name, 7), funs_1)], False)
                         for ood_dataset_name in opt_gen.ood_dataset_names]

    # EVALUATION ================================================
    with torch.no_grad():

        out_dir = "ood_res/%s/" % opt_gen.dataset_name
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        ct = time.strftime("%Y%m%d-%H%M")
        out_file = out_dir + f"%s_{ct}.%s"
        if opt_gen.filename_energytable is None and opt_gen.filename_table is None:
            # Evaluate OOD performance
            print("Evaluating OOD performance")
            table, energytable = ood_eval(in_dataset + ood_datasets, sensitivity_analysis=opt_gen.sens_analysis)
            table1 = beautify_table1(table)
            with pandas.option_context('display.max_rows', None, 'display.max_columns', None, 'expand_frame_repr', False):
                print(table1)
            table.to_csv(out_file % ("table", "csv"), index=False)
            table1.to_csv(out_file % ("table1", "csv"), index=False)
            print("Saving energy table")
            energytable.to_csv(out_file % ("energytable", "csv"), index=False, compression='zip')
            save_table1(table1, out_file)
        elif opt_gen.filename_energytable is not None and opt_gen.visualize_outlier:
            print("Producing outlier images")
            energytable = pandas.read_csv(out_dir + opt_gen.filename_energytable + ".csv", compression="zip")
            plot_outlier(energytable, opt_gen.dataset_name, opt_gen.ood_dataset_names, opt, out_file)
        elif opt_gen.filename_energytable is not None:
            energytable = pandas.read_csv(out_dir + opt_gen.filename_energytable + ".csv", compression="zip")
            print("Producing plot for Figure 3")
            plot = plot1(energytable)
            plot.save(out_file % ("plot1", "png"), dpi=600)
            save_as_pdf_pages([plot], out_file % ("plot1", "pdf"))

