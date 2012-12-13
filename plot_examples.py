import numpy
import pylab
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as pyplot

X = numpy.load('/srv/data/apnea/X.npy', mmap_mode='r')
y = numpy.load('/srv/data/apnea/y.npy', mmap_mode='r')

fig = pyplot.figure(1, (4., 4.))
grid = ImageGrid(fig, 111, nrows_ncols=(3,3), axes_pad=0.1)

print X.shape

for i in range(9):
    j = numpy.random.choice((y==i).nonzero()[0])[0]
    grid[i].imshow(X[j,:,:])
pyplot.show()

