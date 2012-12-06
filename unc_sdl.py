
import sys
import os
import numpy
import scipy
import scipy.signal
from scipy.io import wavfile, loadmat
from datetime import datetime, timedelta
import cPickle as pickle

import pylearn2
from pylearn2.datasets import dense_design_matrix

# Sleep apnea data from the Sleep Disorder Laboratory at the University of North Carolina.
class UNC_SDL(dense_design_matrix.DenseDesignMatrix):
    def __init__(self
            , path="/srv/data/apnea"
            ):

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
        sample_rate = 12000
        stft_width = 50/1000.0
        stft_stride = 50/1000.0
        stft_sample_width = int(stft_width*sample_rate)
        downsample_factor = 3

        # in seconds
        stft_window_width = 15.0
        stft_window_stride = stft_window_width/2.0

        def parse_time(s): return datetime.strptime(s,'%Y-%m-%d %H:%M:%S')
        # Signals of interest:
        signals = ['RERA' # Respiratory Effort Related Arousal
                  ,'OAa'  # Obstructive Apnea with arousal
                  ,'CAa'  # Central Apnea with arousal
                  ,'OHa'  # Obstructive Hypopnea with arousal
                  ,'CHa'  # Central Hypopnea with arousal
                  ]
        end_times = {x: parse_time(nominal_times[x][1]) for x in nominal_times}

        all_meta_name = path+'/all-meta.pck'
        all_windows_name = path+'/all-windows.dat'
        if os.path.isfile(all_meta_name) and os.path.isfile(all_windows_name):
            all_meta = pickle.load(file(all_meta_name,'r'))
            all_windows = numpy.memmap(all_windows_name, dtype='float32', mode='r', shape=all_meta['all_windows.shape'])
            vc = SimpleViewConverter(all_windows.shape[1:])
            X = vc.topo_view_to_design_mat(all_windows)
            y = all_meta['all_labels']
            print >> sys.stderr, "Mapped UNC_SDL dataset from %s" % all_windows_name
            super(UNC_SDL,self).__init__(X=X, y=y, view_converter=vc)
            return
        # Otherwise we have to recompute some or all of the windows.

        # TODO: use unlabeled nights, too
        all_windows = []
        all_labels = []
        for night in labeled_nights:

            basename = path+'/'+night
            wav_name = basename+'.wav'
            windows_name = basename+'-windows.dat'
            meta_name = basename+'-meta.pck'

            if os.path.isfile(meta_name) and os.path.isfile(windows_name):
                print >> sys.stderr, "Mapping precomputed windows from %s" % windows_name
                meta = pickle.load(file(meta_name, 'r'))
                windows = numpy.memmap(windows_name, dtype='float32', mode='r', shape=meta['windows.shape'])
            else:
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
    
                # For now, just glob together all the signals.
                # This gives us the data to predict the Respiratory Disturbance Index (RDI):
                #   RDI = (RERAs + Hypopneas + Apneas) / Total Sleep Time (in hours)
                times = sum([map(parse_time, numpy.hstack(mat[signal].flatten())) 
                                if len(mat[signal]) > 0 else []
                             for signal in signals], [])
                # Each element here is an offset (in seconds) from the beginning of the wav file.
                events = [(t - actual_start_time).total_seconds() for t in times]
                
                # Take the Short-Time Fourier Transform (STFT) of the samples.
                print >> sys.stderr, "Computing STFT..."
                xs = numpy.array(stft(samples, sample_rate, framesz=stft_width, hop=stft_stride))
                del samples
                # discard DC offset (1st FFT coefficient)
                xs = xs[:,1:]
                # get normalization parameters
                xs_min = numpy.min(xs)
                xs_max = numpy.max(xs)
    
                print >> sys.stderr, "Creating windows of STFT..."
                w = int(stft_window_width/stft_stride)
                s = int(stft_window_stride/stft_stride)
                r = xrange(0, xs.shape[0]-w, s)
                windows = numpy.memmap(windows_name, dtype='float32', mode='w+', shape=(len(r), w/downsample_factor, xs.shape[1]/downsample_factor))
                labels = numpy.zeros((windows.shape[0],2), dtype='float32')
                print >> sys.stderr, "Shape: ", windows.shape
                j = 0
                for i in xrange(0, xs.shape[0]-w, s):
                    windows[j] = (downsample(xs[i:i+w],downsample_factor) - xs_min)/(xs_max - xs_min)
                    # label is True if any event occurs during this window
                    # TODO: may want to make it the middle third, and overlap windows
                    if any([stft_stride*i <= e and e <= stft_stride*(i+w)+stft_width for e in events]):
                        labels[j] = [1,0]
                    else:
                        labels[j] = [0,1]
                    j = j+1
                del xs
    
                meta = {'windows.shape': windows.shape, 'labels': labels}
                pickle.dump(meta, file(meta_name, 'w'))
            all_windows.append(windows)
            all_labels.append(meta['labels'])
            del windows
            del meta
            # end for night
        print >> sys.stderr, "Gathering all windows..."
        topo_view = numpy.concatenate(all_windows)
        labels = numpy.concatenate(all_labels)
        del all_windows
        del all_labels

        print >> sys.stderr, "Saving precomputed windows to %s..." % all_windows_name
        all_meta = {'all_windows.shape': topo_view.shape, 'all_labels': labels}
        pickle.dump(all_meta, file(all_meta_name, 'w'))
        mm = numpy.memmap(all_windows_name, dtype='float32', mode='w+', shape=topo_view.shape)
        mm[:] = topo_view[:]
        topo_view = mm
        vc = SimpleViewConverter(topo_view.shape[1:])
        super(UNC_SDL,self).__init__(X=vc.topo_view_to_design_mat(topo_view), y=labels, view_converter=vc)

class SimpleViewConverter(object):
    def __init__(self,shape):
        self.shape = shape
    def view_shape(self): return self.shape
    def weights_view_shape(self): return self.shape

    def design_mat_to_topo_view(self, X):
        #x = X.view()
        #x.shape = (X.shape[0],)+self.shape
        #return x
        return numpy.reshape(X, (X.shape[0],)+self.shape)

    def design_mat_to_weights_view(self, X):
        return self.design_mat_to_topo_view(X)

    def topo_view_to_design_mat(self, X):
        #x = X.view()
        #x.shape = (X.shape[0],numpy.product(self.shape))
        #return x
        return numpy.reshape(X, (X.shape[0],numpy.product(self.shape)))

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
    kernel = numpy.ones((sampling_factor,sampling_factor))
    kernel /= kernel.size
    return scipy.signal.convolve2d(X,kernel,mode='same')[::sampling_factor,::sampling_factor]

if __name__ == "__main__":
    u = UNC_SDL()
    scipy.io.savemat('/srv/data/apnea/data.mat',{'X':u.X, 'y':u.y})


