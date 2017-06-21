import numpy as np
from Utilities import *

def getSphereTimeSeries(theta, phi, geodesic = False):
    N = phi.size
    X = np.zeros((N, 3))
    X[:, 0] = np.cos(theta)*np.cos(phi)
    X[:, 1] = np.sin(theta)*np.cos(phi)
    X[:, 2] = np.sin(phi)
    
    u = np.random.randn(3, 1)
    u = u/np.sqrt(np.sum(u**2))
    
    d = X.dot(u)
    d[d < -1] = -1
    d[d > 1] = 1
    
    x = d
    if geodesic:
        x = obsFn(d)
    return x.flatten()


def getRP2TimeSeries(theta, phi):
    N = phi.size
    X = np.zeros((N, 3))
    X[:, 0] = np.cos(theta)*np.cos(phi)
    X[:, 1] = np.sin(theta)*np.cos(phi)
    X[:, 2] = np.sin(phi)
    
    u = np.random.randn(3, 1)
    u = u/np.sqrt(np.sum(u**2))
    
    d = X.dot(u)
    d[d < -1] = -1
    d[d > 1] = 1

    x = np.arccos(np.abs(d))
    return x.flatten()

def getBulgingSphereTimeSeries(theta, phi):
    N = phi.size
    X = np.zeros((N, 3))
    X[:, 0] = np.cos(theta)*(0.5+np.cos(phi))
    X[:, 1] = np.sin(theta)*(0.5+np.cos(phi))
    X[:, 2] = np.sin(phi)
    
    u = np.random.randn(3, 1)
    u = u/np.sqrt(np.sum(u**2))
    
    d = X.dot(u)
    d[d < -1] = -1
    d[d > 1] = 1

    return (X, d.flatten())


if __name__ == '__main__':
    np.random.seed(100)
    N = 6000
    NPeriods = 50
    S = np.zeros((N, 3))
    theta = np.linspace(0, 2*np.pi*NPeriods, N)
    phi = np.pi*np.linspace(-0.5, 0.5, N)

    #Observation function
    x = getSphereTimeSeries(theta, phi)
    #x = getRP2TimeSeries(theta, phi)

    #Sliding window
    X = getSlidingWindowNoInterp(x, int(N/NPeriods))

    Y = plotSlidingWindowResults(x, X)
    plt.savefig("SphereTimeSeries.svg", bbox_inches='tight')
    
    
    Z = np.zeros((Y.shape[0], 4))
    Z[:, 0:3] = Y[:, 0:3]
    Z[:, 3] = x[0:Z.shape[0]]
    savePCOff(Y, "Sphere.off")

    
