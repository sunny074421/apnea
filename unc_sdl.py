
import sys
import os
import numpy
import scipy
import scipy.signal
from scipy.io import wavfile, loadmat
from datetime import datetime, timedelta
from numpy.lib.format import open_memmap

# Signals of interest (and their class IDs)
classes = { 'RERA': 1
          #, 'OAa':  2
          #, 'CAa':  3
          #, 'OHa':  4
          #, 'CHa':  5
          #, 'OA': 6
          #, 'CA': 7
          #, 'OH': 8
          # no examples of CH anyway
          #, 'CH': 9
          }
sample_rate = 12000
stft_width = 150/1000.0
stft_stride = 50/1000.0
stft_take_coefficients = 300
downsample_factor = 1
# in seconds
stft_window_width = 14.0
window_padding = 10 # pixels
stft_window_stride = 1.0
# Events occurring closer than this many seconds to the window's borders are ignored.
stft_window_tolerance = 0.0 # disabled for now, since it seems to screw up training on the synth set

def load_datasets(path, sizes=(800,800,800,2000), topological=False):
    """Returns [train_set, valid_set, test_set, pretrain_set].
    Each is a tuple (X,y) except for pretrain_set, which is just an X."""
    from theano import shared

    datasets = 4*[None]

    X = numpy.load(path+'/X.npy', mmap_mode='r')
    y = numpy.load(path+'/y.npy')

    # deeplearning.logistic_sgd expects int32
    y = y.astype('int32')

    assert X.shape[0] == y.shape[0]
    assert len(y.shape) == 1

    if not topological:
        # Flatten all but the first dimension (examples).
        X = X.reshape((X.shape[0],numpy.product(X.shape[1:])))

    # Try to choose an equal number of examples from each class.
    def choose(size):
        c = numpy.concatenate([numpy.random.choice((y==i).nonzero()[0], min(size/(len(classes)+1), sum(y==i))) if len((y==i).nonzero()[0]) > 0 else numpy.zeros(0,dtype='int') for i in [0]+classes.values()])
        # fill up the quota with negative examples
        if len(c) < size:
            c = numpy.concatenate([c, numpy.random.choice((y==0).nonzero()[0], size-len(c))])
        return c

    print '%d total, %d positive, %d negative' % (y.shape[0],(y!=0).nonzero()[0].shape[0],(y==0).nonzero()[0].shape[0])
    print 'counts by class: ', numpy.bincount(y,minlength=len(classes))

    print 'selecting training examples'
    selection = choose(sizes[0])
    numpy.random.shuffle(selection)
    datasets[0] = (shared(X[selection],borrow=True), shared(y[selection],borrow=True))

    print 'selecting validation examples'
    selection = choose(sizes[1])
    numpy.random.shuffle(selection)
    datasets[1] = (shared(X[selection],borrow=True), shared(y[selection],borrow=True))

    print 'selecting test examples'
    selection = choose(sizes[2])
    numpy.random.shuffle(selection)
    datasets[2] = (shared(X[selection],borrow=True), shared(y[selection],borrow=True))

    print 'selecting unsupervised pretraining examples'
    # Randomly select examples
    selection = numpy.random.choice(y.shape[0], sizes[3])
    numpy.random.shuffle(selection)
    datasets[3] = shared(X[selection],borrow=True)

    return datasets

def compute_windows(samples, times):
    # Take the Short-Time Fourier Transform (STFT) of the samples.
    print >> sys.stderr, "Computing STFT..."
    xs = numpy.array(stft(samples, sample_rate, framesz=stft_width, hop=stft_stride))
    del samples

    print >> sys.stderr, "Creating windows of STFT..."
    # discard DC offset (1st FFT coefficient)
    xs = xs[:,1:]
    # discard all but the first n coefficients
    xs = xs[:,0:stft_take_coefficients]
    # get normalization parameters
    xs_min = numpy.min(xs)
    xs_max = numpy.max(xs)
    # each row of xs summarizes stft_width seconds of data
    # adjacent rows start stft_stride seconds apart
    window_width_rows = int(stft_window_width/stft_stride)
    window_stride_rows = int(stft_window_stride/stft_stride)
    window_start_indices = xrange(0, xs.shape[0]-window_width_rows, window_stride_rows)
    windows = numpy.zeros((len(window_start_indices), 2*window_padding+window_width_rows/downsample_factor, xs.shape[1]/downsample_factor), dtype='float32')
    labels = numpy.zeros(len(window_start_indices), dtype='int')
    print >> sys.stderr, "Shape: ", windows.shape
    j = 0
    for i in window_start_indices:
        windows[j] = numpy.pad((downsample(xs[i:i+window_width_rows],downsample_factor) - xs_min)/(xs_max - xs_min), ((window_padding,window_padding),(0,0)), mode='constant', constant_values=0)
        # label is True if any event occurs during this window and not within a certain distance of the edges
        window_start_time = stft_stride*i + stft_window_tolerance
        window_end_time = stft_stride*(i+window_width_rows) - stft_window_tolerance
        labels[j] = 0
        for signal in times:
            if any([window_start_time <= e and e <= window_end_time for e in times[signal]]):
                if labels[j] != 0:
                    print >> sys.stderr, "Warning: overlapping events %d, %d near %f s"%(labels[j],classes[signal],window_start_time)
                labels[j] = classes[signal]
        j = j+1
    return (windows, labels)

def build_synthetic_dataset(path='/srv/data/apnea/synthetic'):
    nights = ['synth1-allnegative', 'synth1-apnea']
    event_times = {
            'synth1-allnegative': [],
            'synth1-apnea': sum([map(lambda t: t+300*i,[33, 48, 142, 220, 276]) for i in range(10)],[])
            }
    all_windows = []
    all_labels = []

    for night in nights:
        basename = path+'/'+night
        wav_name = basename+'.wav'

        print >> sys.stderr, "Reading samples from %s" % wav_name
        rate, samples = wavfile.read(wav_name)
        assert rate == sample_rate, "File has wrong sample rate: %s (is %d, should be %d)" % (wav_name,rate,sample_rate)
        assert samples.ndim == 1, "Expected mono audio only: %s" % wav_name
        assert samples.dtype == numpy.dtype('int16'), "Expected 16-bit samples: %s" % wav_name

        # Each element here is an offset (in seconds) from the beginning of the wav file.
        times = {'RERA': event_times[night]}
        
        windows, labels = compute_windows(samples, times)
        del samples
        all_windows.append(windows)
        all_labels.append(labels)

    print >> sys.stderr, "Gathering all examples..."
    windows = numpy.concatenate(all_windows)
    labels = numpy.concatenate(all_labels)
    del all_windows
    del all_labels

    print >> sys.stderr, "Saving X.npy, y.npy"
    numpy.save(path+'/X.npy',windows)
    numpy.save(path+'/y.npy',labels)
    return (windows, labels)

# Sleep apnea data from the Sleep Disorder Laboratory at the University of North Carolina.
def build_dataset(path='/srv/data/apnea'):
    nights = ['302-adjust', '302-nopap', '303-adjust', '303-nopap', '304-adjust', '304-nopap', '305-adjust', '305-nopap', '306-adjust', '306-nopap', '307-adjust', '307-nopap', '309-adjust', '309-nopap', '310-adjust', '310-nopap', '311-adjust', '311-nopap', '312-adjust', '312-nopap', '313-adjust', '313-nopap', '314-adjust', '314-nopap', '315-adjust', '316-adjust', '316-nopap', '317-adjust', '317-nopap']
    labeled_nights = ['302-adjust', '302-nopap', '303-adjust', '303-nopap', '304-nopap', '309-adjust', '310-adjust', '310-nopap', '311-adjust', '312-adjust', '312-nopap', '316-nopap', '317-adjust', '317-nopap']
    # These are the "nominal" start and end times for each WAV file.
    # In fact, however, each WAV file is padded out with unlabeled data,
    # at the beginning (!), to an exact multiple of two minutes in length.
    nominal_times = {
            '302-adjust': ('2011-07-07 22:42:50', '2011-07-08 06:04:04'),
            '302-nopap': ('2011-07-11 22:46:36', '2011-07-12 06:51:09'),
            '303-adjust': ('2011-07-06 00:16:58', '2011-07-06 06:38:52'),
            '303-nopap': ('2011-07-27 22:20:45', '2011-07-28 06:26:15'),
            '304-adjust': ('2011-07-19 21:41:07', '2011-07-20 06:20:42'),
            '304-nopap': ('2011-07-26 22:49:09', '2011-07-27 06:00:30'),
            '305-adjust': ('2011-08-03 23:22:44', '2011-08-04 06:39:41'),
            '305-nopap': ('2011-08-04 23:48:46', '2011-08-05 07:09:23'),
            '306-adjust': ('2011-08-18 22:45:03', '2011-08-19 06:34:35'),
            '306-nopap': ('2011-08-19 22:13:58', '2011-08-20 06:47:19'),
            '307-adjust': ('2011-08-23 22:32:23', '2011-08-24 05:48:02'),
            '307-nopap': ('2011-08-30 22:16:11', '2011-08-31 06:04:09'),
            '309-adjust': ('2011-11-13 22:22:38', '2011-11-14 05:57:54'),
            '309-nopap': ('2011-11-14 22:30:07', '2011-11-15 05:14:17'),
            '310-adjust': ('2011-11-22 23:59:34', '2011-11-23 06:44:19'),
            '310-nopap': ('2011-11-29 00:47:19', '2011-11-29 06:51:23'),
            '311-adjust': ('2011-11-09 22:59:20', '2011-11-10 06:17:11'),
            '311-nopap': ('2011-11-17 22:37:49', '2011-11-18 06:30:28'),
            '312-adjust': ('2011-12-09 23:16:22', '2011-12-10 06:20:20'),
            '312-nopap': ('2011-12-11 22:28:14', '2011-12-12 05:23:21'),
            '313-adjust': ('2011-12-05 22:37:40', '2011-12-06 06:19:57'),
            '313-nopap': ('2011-12-06 21:58:02', '2011-12-07 05:53:00'),
            '314-adjust': ('2012-02-12 23:18:14', '2012-02-13 05:21:12'),
            '314-nopap': ('2012-02-19 22:47:06', '2012-02-20 05:50:21'),
            '315-adjust': ('2012-04-20 23:13:48', '2012-04-21 06:50:22'),
            '316-adjust': ('2012-03-21 23:44:59', '2012-03-22 08:36:15'),
            '316-nopap': ('2012-03-22 22:58:45', '2012-03-23 07:43:39'),
            '317-adjust': ('2012-04-16 00:23:34', '2012-04-16 08:29:43'),
            '317-nopap': ('2012-04-30 00:06:28', '2012-04-30 07:56:16')
            }

    def parse_time(s): return datetime.strptime(s,'%Y-%m-%d %H:%M:%S')
    end_times = {x: parse_time(nominal_times[x][1]) for x in nominal_times}

    # TODO: use unlabeled nights, too
    window_shape = None
    total_examples = 0
    X_names = []
    y_names = []
    for night in labeled_nights:

        basename = path+'/'+night
        wav_name = basename+'.wav'
        X_name = basename+'-X.npy'
        y_name = basename+'-y.npy'

        print >> sys.stderr, "Reading samples from %s" % wav_name
        rate, samples = wavfile.read(wav_name)
        assert rate == sample_rate, "File has wrong sample rate: %s (is %d, should be %d)" % (wav_name,rate,sample_rate)
        assert samples.ndim == 1, "Expected mono audio only: %s" % wav_name
        assert samples.dtype == numpy.dtype('int16'), "Expected 16-bit samples: %s" % wav_name

        mat = loadmat(basename+'.mat')

        actual_length      = timedelta(seconds=len(samples)/float(sample_rate))
        nominal_start_time = parse_time(nominal_times[night][0])
        end_time           = parse_time(nominal_times[night][1])
        actual_start_time  = end_time - actual_length
        assert actual_start_time < nominal_start_time

        # elements here are offsets in seconds from the beginning of the wav file.
        def to_seconds(t): return (parse_time(t) - actual_start_time).total_seconds()
        times = {signal: map(to_seconds, numpy.hstack(mat[signal].flatten()))
                         if len(mat[signal]) > 0 else []
                    for signal in classes}
        
        X, y = compute_windows(samples, times)
        del samples

        assert window_shape is None or window_shape == X.shape[1:]
        window_shape = X.shape[1:]

        numpy.save(X_name, X)
        numpy.save(y_name, y)
        total_examples += X.shape[0]
        X_names.append(X_name)
        y_names.append(y_name)
        del X, y
        # end for night

    print >> sys.stderr, "Gathering all examples..."
    X = open_memmap(path+'/X.npy', mode='w+', dtype='float32', shape=(total_examples,)+window_shape)
    ys = []

    i = 0
    for X_name,y_name in zip(X_names,y_names):
        x1 = numpy.load(X_name, mmap_mode='r')
        ys.append(numpy.load(y_name))
        X[i:i+x1.shape[0]] = x1
    y = numpy.concatenate(ys)
    numpy.save(path+'/y.npy', y)
    return (X,y)

def stft(x, fs, framesz, hop):
    """
    stft(samples, sample_rate, width, stride)
    width and stride are in seconds.
    returns iterator of fft-coefficient arrays.
    """
    framesamp = int(framesz*fs)
    hopsamp = int(hop*fs)
    w = scipy.hamming(framesamp)
    return [numpy.absolute(numpy.fft.rfft(w*x[i:i+framesamp])) for i in xrange(0, len(x)-framesamp, hopsamp)]

def downsample(X, sampling_factor):
    """Basic downsample using a mean kernel."""
    assert X.ndim == 2
    assert X.dtype == 'float32' or X.dtype == 'float64'
    if sampling_factor == 1:
        return X
    kernel = numpy.ones((sampling_factor,sampling_factor))
    kernel /= kernel.size
    return scipy.signal.convolve2d(X,kernel,mode='same')[::sampling_factor,::sampling_factor]

if __name__ == "__main__":
    # Rebuild both datasets
    build_synthetic_dataset()
    build_dataset()


