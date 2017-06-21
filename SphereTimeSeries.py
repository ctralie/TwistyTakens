import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.io as sio
from TDA import *
from SlidingWindow import *
from sklearn.decomposition import PCA
import scipy.sparse as sparse
import scipy.special

def drawLineColored(idx, x, C):
    plt.hold(True)
    for i in range(len(x)-1):
        plt.plot(idx[i:i+2], x[i:i+2], c=C[i, :])

def getGreedyPerm(D):
    N = D.shape[0]
    #By default, takes the first point in the list to be the
    #first point in the permutation, but could be random
    perm = np.zeros(N, dtype=np.int64)
    lambdas = np.zeros(N)
    ds = D[0, :]
    for i in range(1, N):
        print("%i of %i"%(i, N))
        idx = np.argmax(ds)
        perm[i] = idx
        lambdas[i] = ds[idx]
        ds = np.minimum(ds, D[idx, :])
    return (perm, lambdas)

def doFurthestPointSampleGeodesic(X, NNeighbs, NPoints):
    N = X.shape[0]
    #Now make sparse matrix
    D = np.sum(X**2, 1)[:, None]
    D = D + D.T - 2*X.dot(X.T)
    D[D < 0] = 0
    D = 0.5*(D + D.T)
    D = np.sqrt(D)

    J = np.argpartition(D, NNeighbs, 1)[:, 0:NNeighbs].flatten()
    I = np.tile(np.arange(N)[:, None], (1, NNeighbs)).flatten()
    V = D[I, J]
    DSparse = sparse.coo_matrix((V, (I, J)), shape=(N, N))
    D2 = sparse.csgraph.dijkstra(DSparse, False)
    (perm, lambdas) = getGreedyPerm(D2)
    return X[perm[0:NPoints], :]

def doFurthestPointSample(X, NPoints):
    N = X.shape[0]
    XSqr = np.sum(X**2, 1)
    #By default, takes the first point in the list to be the
    #first point in the permutation, but could be random
    perm = np.zeros(NPoints, dtype=np.int64)
    ds = XSqr[0] + XSqr - 2*X[0, :].dot(X.T)
    for i in range(1, NPoints):
        idx = np.argmax(ds)
        perm[i] = idx
        nextds = XSqr[idx] + XSqr - 2*X[idx, :].dot(X.T)
        ds = np.minimum(ds, nextds)
    return X[perm, :]

def getSphereTimeSeries(theta, phi):
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
    #x = d
    return x.flatten()

if __name__ == '__main__':
    np.random.seed(100)
    N = 600
    NPeriods = 20
    S = np.zeros((N, 3))
    theta = np.linspace(0, 2*np.pi*NPeriods, N)
    phi = np.pi*np.linspace(-0.5, 0.5, N)
    
    S = np.zeros((N, 3))
    S[:, 0] = np.cos(theta)*np.cos(phi)
    S[:, 1] = np.sin(theta)*np.cos(phi)
    S[:, 2] = np.sin(phi)
    
    #Observation function
    x = getSphereTimeSeries(theta, phi)
    x = x - np.min(x)
    x = x/np.max(x)
    
    #Save trajectory and observation function
    fout = open("sphere.off", "w")
    fout.write("COFF\n%i 0 0\n"%(len(x)))
    c = plt.get_cmap('spectral')
    C = c(np.array(np.round(255*x), dtype=np.int32))
    C = C[:, 0:3]
    for i in range(len(x)):
        fout.write("%g %g %g %g %g %g\n"%(S[i, 0], S[i, 1], S[i, 2], C[i, 0], C[i, 1], C[i, 2]))
    fout.close()

    X = getSlidingWindowNoInterp(x, int(N/NPeriods))
    x = x - np.min(x)
    x = x/np.max(x)
    x = -2*(x - 0.5)

    #Make color array
    c = plt.get_cmap('Spectral')
    C = c(np.array(np.round(np.linspace(0, 255, X.shape[0])), dtype=np.int32))
    C = C[:, 0:3]

    #Perform PCA down to 2D for visualization
    pca = PCA(n_components = 3)
    Y = pca.fit_transform(X)
    sio.savemat("Sphere.mat", {"Y":Y, "x":x})
    eigs = pca.explained_variance_
    
    fig = plt.figure(figsize=(18, 5))
    plt.subplot(131)
    drawLineColored(np.arange(X.shape[0]), x[0:X.shape[0]], C)
    plt.ylim([-1.1, 1.1])
    ax = plt.gca()
    plotbgcolor = (0.15, 0.15, 0.15)
    ax.set_axis_bgcolor(plotbgcolor)
    plt.title("Original Signal")
    plt.xlabel("t")

    ax2 = fig.add_subplot(132, projection = '3d')
    plt.title("PCA of Sliding Window Embedding\n%.3g%s Variance Explained"%(100*np.sum(pca.explained_variance_ratio_), "%"))
    ax2.scatter(Y[:, 0], Y[:, 1], Y[:, 2], c=C, edgecolor='none')


    plt.subplot(133)
    #Do TDA
    PDs2 = doRipsFiltration(X, 2, thresh = -1, coeff = 3)
    H1 = plotDGM(PDs2[1], color = np.array([1.0, 0.0, 0.2]), label = 'H1', sz = 50, axcolor = np.array([0.8]*3))
    plt.hold(True)
    H2 = plotDGM(PDs2[2], color = np.array([0.43, 0.67, 0.27]), marker = 'x', sz = 50, label = 'H2', axcolor = np.array([0.8]*3))
    #plt.legend(handles=[H1, H2])

    ax = plt.gca()
    ax.set_axis_bgcolor(plotbgcolor)
    plt.title('Persistence Diagrams $\mathbb{Z}_3$')

    
    plt.savefig("SphereTimeSeries.svg", bbox_inches='tight')
