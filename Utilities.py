import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.io as sio
from SlidingWindow import *
from sklearn.decomposition import PCA
import scipy.sparse as sparse
import scipy.special
from ripser import Rips
import sys 
sys.path.append("DREiMac/")
from Laplacian import getTorusCoordinates

"""
Furthest Point Sampling
"""

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






"""
Input/Output
"""

def savePCOff(X, filename):   
    #Save trajectory and observation function
    fout = open(filename, "w")
    fout.write("COFF\n%i 0 0\n"%(X.shape[0]))
    c = plt.get_cmap('afmhot')
    
    x = np.ones(X.shape[0])
    C = 127*np.ones((X.shape[0], 3))
    if X.shape[1] > 3:
        x = X[:, 3]
        x = x - np.min(x)
        x = x/np.max(x)
        #vals = np.array(np.floor(255*x), dtype=np.int32)
        vals = x
        plt.clf()
        plt.plot(vals)
        plt.show()
        C = c(vals)
        C = C[:, 0:3]
        
    for i in range(len(x)):
        fout.write("%g %g %g %g %g %g\n"%(X[i, 0], X[i, 1], X[i, 2], C[i, 0], C[i, 1], C[i, 2]))
    fout.close()



"""
Plotting Tools
"""

def drawLineColored(idx, x, C):
    plt.hold(True)
    for i in range(len(x)-1):
        plt.plot(idx[i:i+2], x[i:i+2], c=C[i, :])

CIRC_COORDS, CIRC_COORDS_LAPLACIAN, PROJ_COORDS, PCA_COORDS = 0, 1, 2, 3
def plotSlidingWindowResults(x, X, projType = PCA_COORDS, doTDA = False, p=47, subsample=1.0):
    xmin = np.min(x)
    xmax = np.max(x)
    
    #Make color array
    c = plt.get_cmap('Spectral')
    C = c(np.array(np.round(np.linspace(0, 255, X.shape[0])), dtype=np.int32))
    C = C[:, 0:3]
    
    if doTDA:
        #fig = plt.figure(figsize=(18, 5))
        plt.subplot(141)
    else:
        fig = plt.figure(figsize=(12, 5))
        plt.subplot(121)
    
    drawLineColored(np.arange(X.shape[0])/subsample, x[0:X.shape[0]], C)
    plt.ylim([xmin - (xmax - xmin)*0.1, xmax + (xmax - xmin)*0.1])
    ax = plt.gca()
    plotbgcolor = (0.15, 0.15, 0.15)
    ax.set_axis_bgcolor(plotbgcolor)
    plt.title("Original Signal")
    plt.xlabel("t")
    

    if doTDA:
        plt.subplot(144)
        #Do TDA
        r = Rips(coeff=p, maxdim=2, do_cocycles=True)
        r.fit_transform(X)
        r.plot(show=False)
        plt.title('Persistence Diagrams $\mathbb{Z}_{%i}$'%p)
    
    if projType == PCA_COORDS:
        if doTDA:
            ax2 = fig.add_subplot(142, projection = '3d')
        else:
            ax2 = fig.add_subplot(122, projection = '3d')
        #Perform PCA down to 2D for visualization
        pca = PCA(n_components = 3)
        Y = pca.fit_transform(X)
        sio.savemat("Sphere.mat", {"Y":Y, "x":x})
        eigs = pca.explained_variance_
        plt.title("PCA of Sliding Window Embedding\n%.3g%s Variance Explained"%(100*np.sum(pca.explained_variance_ratio_), "%"))
        ax2.scatter(Y[:, 0], Y[:, 1], Y[:, 2], c=C, edgecolor='none')
        return Y
    elif projType == CIRC_COORDS and doTDA:
        plt.subplot(143)
        #Get two maps to the circle from the highest two persistence points
        dgm1 = r.dgm_[1]
        idx = np.argsort(dgm1[:, 0]-dgm1[:, 1])
        ccl1 = r.cocycles_[1][idx[0]]
        thresh1 = dgm1[idx[0],1]-0.001
        ccl2 = r.cocycles_[1][idx[1]]
        thresh2 = dgm1[idx[1],1]-0.001
        theta, _ = getCircularCoordinates(X, ccl1, p, thresh1)
        phi, _ = getCircularCoordinates(X, ccl2, p, thresh2)
        #plt.scatter(theta, phi, 20, c=C, edgecolor='none')
        ax = plt.gca()
        ax.set_axis_bgcolor(plotbgcolor)
        return theta, phi
    elif projType == CIRC_COORDS_LAPLACIAN and doTDA:
        plt.subplot(143)
        dgm1 = r.dgm_[1]
        #Find second most persistent H1 dot
        idx = np.argsort(dgm1[:, 0]-dgm1[:, 1])[1]
        birthfac = 0.1
        thresh = birthfac*dgm1[idx, 0] + (1.0-birthfac)*dgm1[idx, 1]
        print("thresh = ", thresh)
        res = getTorusCoordinates(X, thresh, weighted=True)
        theta, phi = res['theta'] + np.pi, res['phi'] + np.pi
        plt.scatter(theta, phi, 20, c=C, edgecolor='none')
        plt.xlabel("Laplacian $\\theta$")
        plt.ylabel("Laplacian $\\phi$")
        plt.title("Sliding Window Estimated Phases")
        ax = plt.gca()
        ax.set_axis_bgcolor(plotbgcolor)
        return theta, phi
