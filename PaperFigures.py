import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.io as sio
from SlidingWindow import *
from Utilities import *
from sklearn.decomposition import PCA

def makeTorusFigure():
    """
    Show an example of a time series that yields a torus
    Plot ground truth phases, as well as Laplacian estimated phases
    and a persistence diagram
    """
    subsample = 4.0
    t = np.arange(500)/subsample
    theta = np.mod(0.5*(1+np.sqrt(5))*t, 2*np.pi)
    phi = np.mod(t, 2*np.pi)
    x = np.cos(theta) + 0.8*np.cos(phi)


    #Sliding window
    X = getSlidingWindow(x, 10, subsample, 1)
    c = plt.get_cmap('Spectral')
    C = c(np.array(np.round(np.linspace(0, 255, X.shape[0])), dtype=np.int32))
    C = C[:, 0:3]
    plt.subplot(142)
    plt.scatter(theta[0:X.shape[0]], phi[0:X.shape[0]], 20, c=C, edgecolor='none')
    ax = plt.gca()
    ax.set_axis_bgcolor((0.15, 0.15, 0.15))
    plt.xlabel("$(1+\\sqrt{5})t/2$")
    plt.ylabel("$t$")
    plt.title("Ground Truth Phases")
    plotSlidingWindowResults(x, X, projType=CIRC_COORDS_LAPLACIAN, doTDA=True, subsample=subsample, p=2)
    plt.subplot(141)
    plt.title("$g(t) = 0.8 \\cos \\left( \\frac{1+\\sqrt{5}}{2} t \\right) + \\cos(t)$")
    plt.show()


if __name__ == '__main__':
    makeTorusFigure()