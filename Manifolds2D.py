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

"""
    NPeriods : int
        Number of periods wrapping around the circle part of the Klein bottle as
        it goes from bottom to top
"""

def getKleinTimeSeries(T1, slope, eps = 0.02):
    """
    Make a Klein bottle time series
    
    Parameters
    ----------
    T1 : int
        The number of samples per period on the circle part
    slope : float
        Slope of the trajectory along principal domain of the Klein bottle
    eps : float
        Fuzz close to the boundary in the y direction.  Or if negative,
        the number of periods to complete
    """
    NPeriods = 1.0/slope
    N = T1*NPeriods
    print("NPeriods = %i, N = %i"%(NPeriods, N))
    if eps < 0:
        print("Expanding period")
        N *= -eps
        y = np.linspace(0, np.pi*(-eps), N)
    else:
        y = np.linspace(0, np.pi, N)
    x = np.arange(N)*2*np.pi/T1
    
    if eps > 0:
        idx = (y>eps)*(y<np.pi-eps) #Exlude points close to the boundary
        x = x[idx]
        y = y[idx]
    return np.cos(2*x) + np.cos(x)*np.sin(y) + np.cos(y)

def doSphereExample():
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

def doKleinExample():
    x = getKleinTimeSeries(40, 0.05)
    plt.plot(x)
    plt.show()

if __name__ == '__main__':
    doKleinExample()