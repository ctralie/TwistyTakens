import numpy as np
from Utilities import *

def getSphereTimeSeries(theta, phi, u, geodesic = False):
    N = phi.size
    X = np.zeros((N, 3))
    X[:, 0] = np.cos(theta)*np.cos(phi)
    X[:, 1] = np.sin(theta)*np.cos(phi)
    X[:, 2] = np.sin(phi)
    
    d = X.dot(u)
    d[d < -1] = -1
    d[d > 1] = 1

    x = d
    if geodesic:
        x = np.arccos(d)
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
    

def getTorusDistance(x, theta, phi, alpha_theta = 1.0, alpha_phi = 1.0, L1 = False):
    """
    Get a distance from points to an observation point x on the torus
    Parameters
    ----------
    x : ndarray (2)
        Position of observation point (theta, phi)
    theta : ndarray (N)
        Theta (x) coordinates of points on the flow
    phi : ndarray (N)
        Phi (y) coordinates of points on the flow
    alpha_theta : float
        Weight of metric along the x direction
    alpha_phi : float
        Weight of metric along the y direction
    """
    dx = np.abs(x[0]-theta)
    dx = np.minimum(dx, np.abs(x[0]+2*np.pi-theta))
    dx = np.minimum(dx, np.abs(x[0]-2*np.pi-theta))
    dy = np.abs(x[1]-phi)
    dy = np.minimum(dy, np.abs(x[1]+2*np.pi-phi))
    dy = np.minimum(dy, np.abs(x[1]-2*np.pi-phi))
    dx = alpha_theta*dx 
    dy = alpha_phi*dy 
    if L1:
        dist = dx + dy
    else:
        dist = np.sqrt(dx**2 + dy**2)
    return dist

def getKleinDistance(x1, theta, phi, alpha_theta = 1.0, alpha_phi = 1.0, L1 = False):
    """
    Get a distance from points to an observation point x on the Klein bottle, where
    the points are specified on its double cover on [0, 2*pi] x [0, 2*pi] and the
    identification is [x, y] ~ [x + pi, -y]
    x1 : ndarray (2)
        Position of observation point on the double cover (theta, phi)
    theta : ndarray (N)
        Theta (x) coordinates of points on the flow
    phi : ndarray (N)
        Phi (y) coordinates of points on the flow
    alpha_theta : float
        Weight of metric along the x direction
    alpha_phi : float
        Weight of metric along the y direction
    """
    x2 = [x1[0]+np.pi, -x1[1]] #Quotient map
    x2 = np.mod(x2, 2*np.pi)
    d1 = getTorusDistance(x1, theta, phi, alpha_theta, alpha_phi, L1)
    d2 = getTorusDistance(x2, theta, phi, alpha_theta, alpha_phi, L1)
    return np.minimum(d1, d2)

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