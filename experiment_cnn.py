import theano
from theano import shared, function
import theano.tensor as T
import numpy
import pickle as pickle
import time
import sys
import os
from deeplearning.convolutional_mlp import LeNetConvPoolLayer
from deeplearning.mlp import HiddenLayer
from deeplearning.logistic_sgd import LogisticRegression
from unc_sdl import load_datasets

# Rprop algorithm. Hooray for Rprop!
def rprop_updates(cost, params):
    initial_rprop_rate=0.005
    minimum_rprop_rate=1e-6
    maximum_rprop_rate=50
    rprop_eta_n = 0.5
    rprop_eta_p = 1.2

    rprop_values = [shared(initial_rprop_rate*numpy.ones(p.get_value(borrow=True).shape,dtype=theano.config.floatX)) for p in params]
    rprop_signs = [shared(numpy.zeros(p.get_value(borrow=True).shape,dtype=theano.config.floatX)) for p in params]
    updates = []
    for param, value, sign in zip(params, rprop_values, rprop_signs):
        grad = T.grad(cost, param)
        sign_new = T.sgn(grad)
        sign_changed = T.neq(sign, sign_new)
        updates.append((param, T.switch(sign_changed, param, param - value*sign_new)))
        updates.append((value, T.clip(T.switch(sign_changed, rprop_eta_n*value, rprop_eta_p*value), minimum_rprop_rate, maximum_rprop_rate)))
        updates.append((sign, sign_new))
    return updates

def train(datasets, batch_size = 200, save_path=None):
    ####
    max_epochs=100

    ishape = (300, 300)

    num_convpool_layers = 3
    nkerns = [4,16,64]
    filtersize = [15,6,5]
    poolsize = [2,2,2]
    tanh_output_size = 500
    n_classes = 9
    ####
    outsize = []
    outsize.append((ishape[0]-filtersize[0]+1)/poolsize[0])
    for i in xrange(1,num_convpool_layers):
        outsize.append((outsize[i-1]-filtersize[i]+1)/poolsize[i])

    print 'layer output sizes: ', outsize, tanh_output_size

    rng = numpy.random.RandomState()

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size
    n_valid_batches /= batch_size
    n_test_batches /= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print 'building the model'

    # Reshape matrix of rasterized images of shape (batch_size,x*y)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    layer0_input = x.reshape((batch_size, 1, ishape[0], ishape[1]))
    layers = []

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to ishape-filtersize+1
    # maxpooling reduces this further by 1/layer0_
    # 4D output tensor is thus of shape (batch_size,nkerns[0],_,_)
    layers.append(LeNetConvPoolLayer(rng, input=layer0_input,
            image_shape=(batch_size, 1, ishape[0], ishape[0]),
            filter_shape=(nkerns[0], 1, filtersize[0], filtersize[0]), poolsize=(poolsize[0],poolsize[0])))

    for i in xrange(1,num_convpool_layers):
        layers.append(LeNetConvPoolLayer(rng, input=layers[-1].output,
                image_shape=(batch_size, nkerns[i-1], outsize[i-1], outsize[i-1]),
                filter_shape=(nkerns[i], nkerns[i-1], filtersize[i], filtersize[i]), poolsize=(poolsize[i], poolsize[i])))

    # the TanhLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size,num_pixels) (i.e matrix of rasterized images).
    layers.append(HiddenLayer(rng, input=layers[-1].output.flatten(2), n_in=nkerns[-1] * outsize[-1] * outsize[-1],
                         n_out=tanh_output_size, activation=T.tanh))

    # classify the values of the fully-connected sigmoidal layer
    layers.append(LogisticRegression(input=layers[-1].output, n_in=tanh_output_size, n_out=n_classes))

    # the cost we minimize during training is the NLL of the model
    cost = layers[-1].negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function([index], layers[-1].errors(y),
             givens={
                x: test_set_x[index * batch_size: (index + 1) * batch_size],
                y: test_set_y[index * batch_size: (index + 1) * batch_size]})

    validate_model = theano.function([index], layers[-1].errors(y),
            givens={
                x: valid_set_x[index * batch_size: (index + 1) * batch_size],
                y: valid_set_y[index * batch_size: (index + 1) * batch_size]})

    # create a list of all model parameters to be fit by gradient descent
    params = sum([layer.params for layer in layers], [])

    # train_model is a function that updates the model parameters
    updates = rprop_updates(cost, params)    
    train_model = theano.function([index], cost, updates=updates,
            givens={ x: train_set_x[index * batch_size: (index + 1) * batch_size],
                     y: train_set_y[index * batch_size: (index + 1) * batch_size]})

    # evaluate_model returns the predicted classes on the test set
    evaluate_model = theano.function([index], layers[-1].y_pred,
            givens={x: test_set_x[index * batch_size: (index + 1) * batch_size]})

    ###############
    # TRAIN MODEL #
    ###############
    print 'training'
    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatches before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False

    while (epoch < max_epochs) and (not done_looping):
        epoch = epoch + 1

        # Shuffle training set.
        rng_state = numpy.random.get_state()
        numpy.random.shuffle(train_set_x.get_value(borrow=True))
        numpy.random.set_state(rng_state)
        numpy.random.shuffle(train_set_y.get_value(borrow=True))

        for minibatch_index in xrange(n_train_batches):
            print 'epoch %i, minibatch %i/%i ...' % (epoch, minibatch_index+1, n_train_batches)

            iter = epoch * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print 'training @ iter = ', iter
            cost_ij = train_model(minibatch_index)
            print 'LL ~', cost_ij

            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' % \
                      (epoch, minibatch_index + 1, n_train_batches, \
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [test_model(i) for i in xrange(n_test_batches)]
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of best '
                           'model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

                    # compute a confusion matrix
                    # (row,column) <-> (true,predicted)
                    test_pred_y = numpy.concatenate([evaluate_model(i) for i in xrange(n_test_batches)])
                    confusion = numpy.bincount(n_classes*test_set_y.get_value() + test_pred_y, minlength=n_classes*n_classes).reshape((n_classes,n_classes))
                    print '     confusion matrix of best model on test data:\n', confusion

                    if save_path is not None:
                        print 'saving best model...'
                        pickle.dump(layers, file(save_path,'w'))

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i,'\
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

if __name__ == '__main__':
    #path = '/srv/data/apnea/synthetic'
    #import unc_sdl
    #unc_sdl.build_synthetic_dataset(path)
    #datasets = load_datasets(path, sizes=(500,100,100,0))

    path = '/srv/data/apnea'
    datasets = load_datasets(path, sizes=(6000,1000,1000,0))

    train(datasets, batch_size=500, save_path=path+'/model-cnn.pkl')


