import unc_sdl

from pylearn2.autoencoder import Autoencoder, DenoisingAutoencoder
from pylearn2.models.rbm import GaussianBinaryRBM
from pylearn2.corruption import BinomialCorruptor
from pylearn2.corruption import GaussianCorruptor
from pylearn2.training_algorithms.sgd import SGD
from pylearn2.costs.autoencoder import MeanSquaredReconstructionError
from pylearn2.training_algorithms.sgd import EpochCounter
from pylearn2.classifier import LogisticRegressionLayer
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.datasets import preprocessing
from pylearn2.energy_functions.rbm_energy import GRBM_Type_1
from pylearn2.base import StackedBlocks
from pylearn2.datasets.transformer_dataset import TransformerDataset
from pylearn2.costs.ebm_estimation import SMD
from pylearn2.termination_criteria import MonitorBased
from pylearn2.training_algorithms.sgd import MonitorBasedLRAdjuster
from pylearn2.train_extensions import TrainExtension
from pylearn2.costs.cost import CrossEntropy
from pylearn2.train import Train
import pylearn2.utils.serial as serial

import numpy
import numpy.random
import sys
import os

def get_autoencoder(structure):
    n_input, n_output = structure
    config = {
        'nhid': n_output,
        'nvis': n_input,
        'tied_weights': True,
        'act_enc': 'tanh',
        'act_dec': 'sigmoid',
        'irange': 0.001,
        }
    return Autoencoder(**config)

def get_denoising_autoencoder(structure):
    n_input, n_output = structure
    corruptor = BinomialCorruptor(corruption_level=0.5)
    config = {
        'corruptor': corruptor,
        'nvis': n_input,
        'nhid': n_output,
        'tied_weights': True,
        'act_enc': 'tanh',
        'act_dec': 'sigmoid',
        'irange': 0.001,
        }
    return DenoisingAutoencoder(**config)

def get_grbm(structure):
    n_input, n_output = structure
    config = {
        'nvis': n_input,
        'nhid': n_output,
        'irange': 0.05,
        'energy_function_class': GRBM_Type_1,
        'learn_sigma': True,
        'init_sigma': 0.4,
        'init_bias_hid': -2.0,
        'mean_vis': False,
        'sigma_lr_scale': 1e-3,
        }
    return GaussianBinaryRBM(**config)

def get_logistic_regressor(structure):
    n_input, n_output = structure
    return LogisticRegressionLayer(nvis=n_input, nclasses=n_output)

def get_layer_trainer_logistic(layer, trainset):
    config = {
        'learning_rate': 0.1,
        'cost': CrossEntropy(),
        'batch_size': 10,
        #'monitoring_batches': 10,
        #'monitoring_dataset': None,
        'termination_criterion': EpochCounter(max_epochs=10),
        }
    train_algo = SGD(**config)
    model = layer
    extensions = None
    return Train(model = model,
            dataset = trainset,
            algorithm = train_algo,
            extensions = extensions)

def get_layer_trainer_sgd_autoencoder(layer, trainset):
    config = {
        'learning_rate': 0.1,
        'cost': MeanSquaredReconstructionError(),
        'batch_size': 10,
        #'monitoring_batches': 10,
        #'monitoring_dataset': None,
        'termination_criterion': EpochCounter(max_epochs=1),
        }
    train_algo = SGD(**config)
    model = layer
    extensions = None
    return Train(model = model,
            algorithm = train_algo,
            extensions = extensions,
            dataset = trainset)

def get_layer_trainer_sgd_rbm(layer, trainset):
    config = {
        'learning_rate': 0.1,
        'cost': SMD(corruptor=GaussianCorruptor(stdev=0.4)),
        'batch_size': 5,
        'monitoring_batches': 20,
        'monitoring_dataset': trainset,
        'termination_criterion': EpochCounter(max_epochs=1),
        # another option:
        # 'termination_criterion': MonitorBased(prop_decrease=0.01, N=10),
        }
    train_algo = SGD(**config)
    model = layer
    extensions = [MonitorBasedLRAdjuster()]
    return Train(model = model,
            algorithm = train_algo,
            extensions = extensions,
            dataset = trainset)

def main():
    fullset = unc_sdl.UNC_SDL()
    # TODO: split train, test
    trainset = fullset
    testset = fullset

    n_input = trainset.get_design_matrix().shape[1]
    layers = []
    structure = [[n_input, 400],[400,50],[50,100],[100,2]]
    # layer 0: gaussianRBM
    layers.append(get_grbm(structure[0]))
    # layer 1: denoising AE
    layers.append(get_denoising_autoencoder(structure[1]))
    # layer 2: AE
    layers.append(get_autoencoder(structure[2]))
    # layer 3: logistic regression
    layers.append(get_logistic_regressor(structure[3]))

    # construct training sets for different layers
    trainset = [ trainset,
                TransformerDataset(raw = trainset, transformer = layers[0]),
                TransformerDataset(raw = trainset, transformer = StackedBlocks(layers[0:2])),
                TransformerDataset(raw = trainset, transformer = StackedBlocks(layers[0:3])) ]
    # construct layer trainers
    layer_trainers = []
    layer_trainers.append(get_layer_trainer_sgd_rbm(layers[0], trainset[0]))
    layer_trainers.append(get_layer_trainer_sgd_autoencoder(layers[1], trainset[1]))
    layer_trainers.append(get_layer_trainer_sgd_autoencoder(layers[2], trainset[2]))
    layer_trainers.append(get_layer_trainer_logistic(layers[3], trainset[3]))

    # unsupervised pretraining
    for layer_trainer in layer_trainers[0:3]:
        layer_trainer.main_loop()

    # supervised training
    layer_trainers[-1].main_loop()

if __name__ == '__main__':
    main()

