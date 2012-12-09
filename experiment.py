import numpy
import scipy
import theano.tensor as T
from theano import shared
# NB: cPickle fucks up some Theano structures here, apparently
import pickle
import os

from deeplearning.DBN import DBN

def load_datasets(path):
    """Returns [train_set, valid_set, test_set, pretrain_set].
    Each is a tuple (X,y) except for pretrain_set, which is just an X."""
    datasets = 4*[None]

    X = numpy.load(path+'/X.npy', mmap_mode='r')
    y = numpy.load(path+'/y.npy')

    # deeplearning.logistic_sgd expects int32
    y = y.astype('int32')

    assert X.shape[0] == y.shape[0]
    assert len(y.shape) == 1

    ####
    each_set_size = 500
    pretraining_size = 4000
    ####

    # Flatten all but the first dimension (examples).
    # We don't make use of the 2D topology of the input.
    X = X.reshape((X.shape[0],numpy.product(X.shape[1:])))

    print 'selecting training examples'
    # Use all the positive examples, then fill up with random examples
    # TODO: hold out some positive examples
    selection = y[:]==1
    selection[numpy.random.choice(y.shape[0], each_set_size-sum(selection), replace=False)] = True
    datasets[0] = (shared(X[selection],borrow=True), shared(y[selection],borrow=True))

    print 'selecting validation examples'
    # Use all the positive examples, then fill up with random examples
    selection = y[:]==1
    selection[numpy.random.choice(y.shape[0], each_set_size-sum(selection), replace=False)] = True
    datasets[1] = (shared(X[selection],borrow=True), shared(y[selection],borrow=True))

    print 'selecting test examples'
    # Use all the positive examples, then fill up with random examples
    selection = y[:]==1
    selection[numpy.random.choice(y.shape[0], each_set_size-sum(selection), replace=False)] = True
    datasets[2] = (shared(X[selection],borrow=True), shared(y[selection],borrow=True))

    print 'selecting unsupervised pretraining examples'
    # Use all the positive examples, then fill up with random examples
    selection = y[:]==1
    selection[numpy.random.choice(y.shape[0], pretraining_size-sum(selection), replace=False)] = True
    datasets[3] = shared(X[selection],borrow=True)

    return datasets

def pretrain(datasets):
    ####
    batch_size = 50
    hidden_layers_sizes = [2000,2000,2000,2000]
    max_epochs = [20,20,20,20]
    numpy_rng = numpy.random.RandomState(seed=42)
    pretrain_lr = 0.1
    gibbs_steps = 1
    ####
    
    train_set_x = datasets[3]

    n_train_batches = int(train_set_x.get_value(borrow=True).shape[0] / batch_size)
    
    print 'constructing dbn'
    dbn = DBN(numpy_rng,
            n_ins = train_set_x.get_value(borrow=True).shape[1],
            hidden_layers_sizes = hidden_layers_sizes,
            n_outs = 2)
    
    print 'compiling pretraining functions'
    pretraining_fns = dbn.pretraining_functions(train_set_x=train_set_x, batch_size=batch_size, k=gibbs_steps)
    
    last_cost = -1e10
    for layer_index in xrange(dbn.n_layers):
        print 'pretraining layer %d'%(layer_index+1)
        for epoch in xrange(max_epochs[layer_index]):
            c = []
            for batch_index in xrange(n_train_batches):
                c.append(pretraining_fns[layer_index](index=batch_index,lr=pretrain_lr))
                #print 'batch %d/%d cost %f'%(batch_index+1,n_train_batches,c[-1])
            c = numpy.mean(c)
            print 'layer %d, epoch %d. mean cost: %f'%(layer_index+1,epoch+1,c)
            # TODO: stop if cost doesn't improve
            last_cost = c

    return dbn

def finetune(dbn,datasets):
    ####
    batch_size = 50
    finetune_lr = 0.1
    training_epochs = 10
    ####

    print 'compiling finetuning functions'
    train_fn, validate_model, test_model = dbn.build_finetune_functions(
            datasets=datasets[0:3], batch_size=batch_size,
            learning_rate=finetune_lr)
    n_train_batches = int(datasets[0][0].get_value(borrow=True).shape[0] / batch_size)

    # early-stopping parameters
    patience = 4 * n_train_batches  # look as this many examples regardless
    patience_increase = 2.    # wait this much longer when a new best is
                              # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatches before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    test_score = 0.

    done_looping = False
    epoch = 0

    while (epoch < training_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            minibatch_avg_cost = train_fn(minibatch_index)
            iteration = epoch * n_train_batches + minibatch_index

            if (iteration + 1) % validation_frequency == 0:

                validation_losses = validate_model()
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' % \
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if (this_validation_loss < best_validation_loss *
                        improvement_threshold):
                        patience = max(patience, iteration * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iteration

                    # test it on the test set
                    test_losses = test_model()
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iteration:
                done_looping = True
                break

    print(('Optimization complete with best validation score of %f %%,'
           'with test performance %f %%') %
                 (best_validation_loss * 100., test_score * 100.))
    return dbn

if __name__ == '__main__':
    path = '/srv/data/apnea'
    
    datasets = load_datasets(path)

    if os.path.isfile(path+'/model-pretrained.pkl'):
        print 'loading model'
        dbn = pickle.load(file(path+'/model-pretrained.pkl','r'))
    else:
        dbn = pretrain(datasets)
        print 'saving model'
        pickle.dump(dbn, file(path+'/model-pretrained.pkl','w'))

    finetune(dbn,datasets)
    print 'saving model'
    pickle.dump(dbn, file(path+'/model-finetuned.pkl','w'))

