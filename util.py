from tqdm import tqdm, trange
import numpy as np
import os, pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from copy import deepcopy
from glob import glob
from pprint import pprint
from tqdm import tqdm


from design_bench.datasets.continuous.superconductor_dataset import SuperconductorDataset
from design_bench.datasets.continuous.ant_morphology_dataset import AntMorphologyDataset
from design_bench.datasets.continuous.hopper_controller_dataset import HopperControllerDataset
from design_bench.datasets.continuous.dkitty_morphology_dataset import DKittyMorphologyDataset

from design_bench.datasets.discrete.tf_bind_8_dataset import TFBind8Dataset
from design_bench.datasets.discrete.gfp_dataset import GFPDataset
from design_bench.datasets.discrete.utr_dataset import UTRDataset
from design_bench.datasets.discrete.chembl_dataset import ChEMBLDataset
from design_bench.datasets.discrete.tf_bind_10_dataset import TFBind10Dataset
from design_bench.datasets.discrete.cifar_nas_dataset import CIFARNASDataset
torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

TASKS = { 'tfbind8': 'TFBind8-Exact-v0', # requires morphing-agents
          'gfp': 'GFP-GP-v0',
          'utr': 'UTR-ResNet-v0',
          'hopper': 'HopperController-Exact-v0', # requires morphing-agents
          'rf': 'Superconductor-RandomForest-v0',
          'ant': 'AntMorphology-Exact-v0', # requires morphing-agents
          'dkitty': 'DKittyMorphology-Exact-v0' # requires morphing-agents
        }
DATASETS = {
    'superconductor': lambda: SuperconductorDataset(),
    'tf-bind-8': lambda: TFBind8Dataset(),
    'tf-bind-10': lambda: TFBind10Dataset(),
    'hopper': lambda: HopperControllerDataset(),
    'dkitty': lambda: DKittyMorphologyDataset(),
    'ant': lambda: AntMorphologyDataset(),
    'gfp': lambda: GFPDataset(),
    'utr': lambda: UTRDataset(),
}



def freeze(model):
    assert isinstance(model, nn.Module)
    for p in model.parameters():
        p.requires_grad = False

def unfreeze(model):
    assert isinstance(model, nn.Module)
    for p in model.parameters():
        p.requires_grad = True


def init_weights(m, method = 'kaiming'):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
        if method == 'kaiming':
            torch.nn.init.kaiming_uniform_(m.weight)
        else:
            torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.00)

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch_seed = seed
    np_seed = seed
    np.random.seed(np_seed)
    torch.manual_seed(torch_seed)

def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output)

def load_object(filename):
    with open(filename, 'rb') as output:
        return pickle.load(output)

def check(FOLDER):
    if not os.path.exists(FOLDER):
        os.makedirs(FOLDER)