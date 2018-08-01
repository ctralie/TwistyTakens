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

def intersectSegments2D(A, B, C, D, countEndpoints = True):
    """
    Find the intersection of two lines segments in a numerically stable
    way by looking at them parametrically
    """
    denomDet = (D[0]-C[0])*(A[1]-B[1]) - (D[1]-C[1])*(A[0]-B[0])
    if (denomDet == 0): #Segments are parallel
        return np.array([])
    num_t = (A[0]-C[0])*(A[1]-B[1]) - (A[1]-C[1])*(A[0]-B[0])
    num_s = (D[0]-C[0])*(A[1]-C[1]) - (D[1]-C[1])*(A[0]-C[0])
    t = float(num_t) / float(denomDet)
    s = float(num_s) / float(denomDet)
    if (s < 0 or s > 1):
        return np.array([]) #Intersection not within the bounds of segment 1
    if (t < 0 or t > 1):
        return np.array([]) #Intersection not within the bounds of segment 2
    #Don't count intersections that occur at the endpoints of both segments
    #if the user so chooses
    if ((t == 0 or t == 1) and (s == 0 or s == 1) and (not countEndpoints)):
        return np.array([])
    ret = A + s*(B-A)
    return ret

def get2HoledTorusTraj(x0, dx, NPoints):
    """
    Come up with a trajectory on the unit octagon representation
    of the 2-holed torus
    Parameters
    ----------
    x0 : ndarray (2, 1)
        Initial position on the 2-holed torus
    dx : ndarray (2, 1)
        Vector between adjacent points on the trajectory
    NPoints : int
        Number of points on the trajectory
    """
    x0 = np.array(x0)
    dx = np.array(dx)
    thetas = np.linspace(0, 2*np.pi, 9) - np.pi/8
    endpts = np.zeros((9, 2))
    endpts[:, 0] = np.cos(thetas)
    endpts[:, 1] = np.sin(thetas)
    normals = endpts[1::, :] - endpts[0:-1, :]
    normals[:, 0], normals[:, 1] = normals[:, 1], -normals[:, 0]
    normals = normals/np.sqrt(np.sum(normals**2, 1))[:, None]
    width = endpts[0, 0] - endpts[5, 0]

    X = [x0]
    for i in range(1, NPoints):
        x1 = X[i-1]
        x2 = x1 + dx
        # Check if out of bounds of torus
        k = 0
        while k < 8:
            res = intersectSegments2D(x1, x2, endpts[k, :], endpts[k+1, :])
            if res.size > 0:
                x1 = res - width*normals[k, :] #Go to other side of octagon
                x2 = x1 + (x2 - res)
                x1 = x1+1e-10*normals[k, :]
                k = 0
                continue
            k += 1
        X.append(x2)
    X = np.array(X)
    return {'X':X, 'endpts':endpts}


def get2HoledTorusDist(X, x0, endpts):
    """
    Compute the distance from a set of points to a chosen point on 
    the 2-holed torus, using the flat Euclidean metric on the octagon
    Parameters
    ----------
    X: ndarray (N, 2)
        A set of points inside of the octagon
    x0: ndarray (2)
        A point to which to measure distances
    endpts: ndarray (9, 2)
        Endpoints on the octagon model
    """
    offsets = endpts[1:9, :] + endpts[0:8, :]
    Y = x0 + offsets
    Y = np.concatenate((x0[None, :], Y), 0)
    XSqr = np.sum(X**2, 1)
    YSqr = np.sum(Y**2, 1)
    D = XSqr[:, None] + YSqr[None, :] - 2*X.dot(Y.T)
    distSqr = np.min(D, 1)
    distSqr[distSqr < 0] = 0
    return distSqr


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
    x0 = [0.1, 0.1]
    dx = 3*np.array([0.02*(1+np.sqrt(5))/2, 0.04])
    res = get2HoledTorusTraj(x0, dx, 1000)
    endpts, X = res['endpts'], res['X']

    c = plt.get_cmap('Spectral')
    C = c(np.array(np.round(np.linspace(0, 255, X.shape[0])), dtype=np.int32))
    C = C[:, 0:3]

    x0 = np.array([0.1, 0.1])
    y = get2HoledTorusDist(X, x0, endpts)

    plt.subplot(121)
    plt.plot(endpts[:, 0], endpts[:, 1])
    plt.scatter(x0[0], x0[1], 80, 'k')
    plt.scatter(X[:, 0], X[:, 1], 20, c=C)
    plt.axis('equal')

    plt.subplot(122)
    plt.plot(y)
    plt.show()