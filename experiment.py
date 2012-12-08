import unc_sdl

from pylearn2.autoencoder import Autoencoder, DenoisingAutoencoder
from pylearn2.models.rbm import RBM, GaussianBinaryRBM
from pylearn2.corruption import BinomialCorruptor
from pylearn2.corruption import GaussianCorruptor
from pylearn2.training_algorithms.sgd import SGD, AnnealedLearningRate
from pylearn2.costs.autoencoder import MeanSquaredReconstructionError
from pylearn2.training_algorithms.sgd import EpochCounter
from pylearn2.classifier import LogisticRegressionLayer
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.datasets import preprocessing
from pylearn2.energy_functions.rbm_energy import GRBM_Type_1
from pylearn2.base import Block, StackedBlocks
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
import cPickle as pickle

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

def get_brbm(structure):
    n_input, n_output = structure
    config = {
        'nvis': n_input,
        'nhid': n_output,
        'irange': 0.05,
        }
    return RBM(**config)

def get_grbm(structure):
    n_input, n_output = structure
    config = {
        'nvis': n_input,
        'nhid': n_output,
        'irange': 0.05,
        'energy_function_class': GRBM_Type_1,
        'learn_sigma': True,
        'init_sigma': 0.4,
        #'init_bias_hid': -2.0,
        'mean_vis': False,
        'sigma_lr_scale': 1e-3,
        }
    return GaussianBinaryRBM(**config)

def get_logistic_regressor(structure):
    n_input, n_output = structure
    return LogisticRegressionLayer(nvis=n_input, nclasses=n_output)

def get_layer_trainer_logistic(layer, trainset, save_path):
    config = {
        'learning_rate': 0.1,
        'cost': CrossEntropy(),
        'batch_size': 50,
        'monitoring_batches': 10,
        'monitoring_dataset': trainset,
        #'termination_criterion': EpochCounter(max_epochs=10),
        'termination_criterion': MonitorBased(prop_decrease=0.01, N=10),
        'update_callbacks': [AnnealedLearningRate(1)]
        }
    train_algo = SGD(**config)
    model = layer
    extensions = []
    return Train(model = model,
            dataset = trainset,
            algorithm = train_algo,
            extensions = extensions,
            save_path = save_path,
            save_freq = 1)

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

def get_layer_trainer_sgd_rbm(layer, trainset, save_path):
    config = {
        'learning_rate': 0.1,
        'cost': SMD(corruptor=GaussianCorruptor(stdev=0.1)),
        'batch_size': 50,
        'monitoring_batches': 10,
        'monitoring_dataset': trainset,
        'termination_criterion': EpochCounter(max_epochs=1),
        #'termination_criterion': MonitorBased(prop_decrease=0.01, N=10),
        }
    train_algo = SGD(**config)
    model = layer
    extensions = [MonitorBasedLRAdjuster()]
    return Train(model = model,
            algorithm = train_algo,
            extensions = extensions,
            dataset = trainset,
            save_path = save_path,
            save_freq = 1)

def train(trainset):
    #n_input = trainset.get_batch_design(1).shape[1]
    n_input = 300*300
    structure = [[n_input,1000],[1000,1000],[1000,1000],[1000,2]]
    layers = []
    layers.append(get_grbm(structure[0]))
    layers.append(get_brbm(structure[1]))
    layers.append(get_brbm(structure[2]))
    layers.append(get_logistic_regressor(structure[3]))

    # construct training sets for different layers
    trainset = [ trainset,
                TransformerDataset(raw = trainset, transformer = StackedBlocks([layers[0]])),
                TransformerDataset(raw = trainset, transformer = StackedBlocks(layers[0:2])),
                TransformerDataset(raw = trainset, transformer = StackedBlocks(layers[0:3])) ]
    layer_trainers = []
    layer_trainers.append(get_layer_trainer_sgd_rbm(layers[0], trainset[0], 'layer0.pck'))
    layer_trainers.append(get_layer_trainer_sgd_rbm(layers[1], trainset[1], 'layer1.pck'))
    layer_trainers.append(get_layer_trainer_sgd_rbm(layers[2], trainset[2], 'layer2.pck'))
    layer_trainers.append(get_layer_trainer_logistic(layers[3], trainset[3], 'layer3.pck'))

    for layer_trainer in layer_trainers[0:-1]
        print '*** Unsupervised pretraining: next layer'
        layer_trainer.main_loop()

    print '*** Supervised training: final layer'
    layer_trainers[-1].main_loop()

class CastAndScale(Block):
    def perform(self, X):
        self._params = None
        return X.astype('float32') / 255.0

def main2(trainset, testset):
    layers = []
    for i in xrange(0,4):
        layers.append(pickle.load(file('layer%d.pck'%i,'r')))
    model = StackedBlocks(layers)
    z = numpy.zeros(testset.y.shape)
    for i in xrange(0,testset.num_examples):
        z[i] = model.perform(testset.X[i].reshape((1,)+testset.X[i].shape))
    pickle.dump(z,file('z.pck','w'))

if __name__ == '__main__':
    fullset = unc_sdl.UNC_SDL()
    #fullset = TransformerDataset(raw=fullset, transformer=CastAndScale())
    # TODO: split train, test
    trainset = fullset
    testset = fullset

    train(trainset)

