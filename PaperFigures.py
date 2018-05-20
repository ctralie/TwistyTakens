import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.io as sio
from SlidingWindow import *
from Utilities import *
from sklearn.decomposition import PCA
from Manifolds2D import *
from ripser import Rips
from matplotlib.patches import Polygon
import sys 
sys.path.append("DREiMac")
from ProjectiveCoordinates import *

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

    Y = getDiffusionMap(getSSM(X), 0.1)
    plt.subplot(143)
    phi = np.arctan2(Y[:, -2], Y[:, -3])
    theta = np.arctan2(Y[:, -4], Y[:, -5])
    plt.scatter(theta, phi, 20, c=C, edgecolor='none')
    ax = plt.gca()
    ax.set_axis_bgcolor((0.15, 0.15, 0.15))
    plt.show()

def makeKleinProjCoordsFigure():
    T = 30
    slope = 0.02
    #eps = -4
    eps = 0.02
    #slope = 0.2
    x = getKleinTimeSeries(T, slope, eps)
    X = getSlidingWindowNoInterp(x, T)
    c = plt.get_cmap('Spectral')
    C = c(np.array(np.round(np.linspace(0, 255, X.shape[0])), dtype=np.int32))
    C = C[:, 0:3]
    M = X.shape[0]
    cocycle_idx = [0, 1]
    NLandmarks = 200

    fig = plt.figure(figsize=(6, 6))

    res = ProjCoords(X, NLandmarks, cocycle_idx = cocycle_idx, proj_dim=2, perc = 0.9)
    SFinal = getStereoProjCodim1(res['X'])

    plt.subplot(141)
    res["rips"].plot(show = False)
    dgm1 = res["dgm1"]
    idx = res["idx_p1"]
    #plt.scatter(dgm1[idx, 0]*2, dgm1[idx, 1]*2, 40, 'r')

    plt.subplot(142)
    drawLineColored(np.arange(M), x[0:M], C)

    plt.subplot(143)
    N = int(SFinal.shape[0]/2)
    SFinal1 = SFinal[0:N, :]
    C1 = C[0:N, :]
    SFinal2 = SFinal[N::, :]
    C2 = C[N::, :]
    plotRP2Stereo(SFinal1, C1)
    ax = plt.gca()
    AW = 0.02
    AWV = 0.04
    AXW = 0.002
    for i in range(0, C1.shape[0], 7):
        x1 = SFinal[i, :]
        rx = SFinal[i+1, :] - x1
        if np.sqrt(np.sum(rx**2)) > 0.5:
            continue
        ax.arrow(x1[0], x1[1], rx[0], rx[1], head_width = AW, head_length = AW, fc = 'k', ec = 'k', width = AXW)

    """
    plt.subplot(144)
    plotRP2Stereo(SFinal2, C2)

    plt.title("Cocycles %s"%cocycle_idx)
    """

    """
    ax = fig.add_subplot(133, projection='3d')
    plotRP3Stereo(ax, SFinal, C)
    plt.title("$\\theta$")
    """



    """
    plt.subplot(132)
    r = Rips(coeff=2, maxdim=2, do_cocycles=True)
    r.fit_transform(X)
    r.plot(show=False)
    plt.title("$\mathbb{Z} / 2$")

    plt.subplot(133)
    r = Rips(coeff=3, maxdim=2, do_cocycles=True)
    r.fit_transform(X)
    r.plot(show=False)
    plt.title("$\mathbb{Z} / 3$")
    """

    plt.show()
    #plt.savefig("Klein.svg", bbox_inches='tight')

def makeTorusDistanceFigure(ratio = np.sqrt(2)):
    res = 400
    np.random.seed(200)
    x0 = 2*np.pi*np.random.rand(2)
    
    # Step 1: Compute distance on a grid torus
    theta = np.linspace(0, 2*np.pi, res)
    [theta, phi] = np.meshgrid(theta, theta)
    dist = getTorusDistance(x0, theta, phi)

    #Step 2: Make torus time series
    subsample = 4.0
    t = np.arange(500)/subsample
    theta = np.mod(ratio*t, 2*np.pi)
    phi = np.mod(t, 2*np.pi)

    x = getTorusDistance(x0, theta, phi)
    X = getSlidingWindow(x, 10, subsample, 1)
    c = plt.get_cmap('Spectral')
    C = c(np.array(np.round(np.linspace(0, 255, X.shape[0])), dtype=np.int32))
    C = C[:, 0:3]

    plt.subplot(141)
    plt.imshow(dist, extent = (0, 2*np.pi, 2*np.pi, 0), cmap = 'afmhot', interpolation = 'none')
    plt.scatter(x0[0], x0[1], 20)
    plt.title("Distance Function")
    plt.gca().invert_yaxis()

    plt.subplot(142)
    plt.imshow(dist, cmap = 'afmhot', extent = (0, 2*np.pi, 0, 2*np.pi), interpolation = 'none')
    plt.scatter(theta, phi, c=C)
    plt.title("Distance Function with Trajectory")

    plt.subplot(143)
    drawLineColored(np.arange(X.shape[0]), x[0:X.shape[0]], C)
    plt.title("Time Series")

    plt.subplot(144)
    r = Rips(maxdim=2)
    r.fit_transform(X)
    r.plot(show=False)

    plt.show()

def makeKleinFigure(distance = True):
    eps = 0.02
    T1 = 20
    slope = 0.05
    NPeriods = 1.0/slope
    N = int(T1*NPeriods)
    print("N = %i"%N)
    theta = np.mod(np.arange(N)*2*np.pi/T1, 2*np.pi)
    phi = np.linspace(0, np.pi, N)
    theta = theta[(phi > eps)*(phi < np.pi-eps)]
    phi = phi[(phi > eps)*(phi < np.pi-eps)]

    alpha_theta = 1.0 
    alpha_phi = 0.5
    L1 = False
    np.random.seed(3)
    #Define observation point
    x1 = np.array([4.5, 2.5])
    x2 = np.array(x1)
    x2[0] = np.pi + x2[0]
    x2[1] *= -1
    x2 = np.mod(x2, 2*np.pi)
    #x1 = [np.pi/2, np.pi/2]
    print(x1)

    # Get the correct distance under the quotient
    if distance:
        x = getKleinDistance(x1, theta, phi, alpha_theta, alpha_phi, L1)
    else:
        x = np.cos(2*theta) + np.cos(theta)*np.sin(phi) + np.cos(phi)
    X = getSlidingWindowNoInterp(x, T1*2)

    c = plt.get_cmap('Spectral')
    C = c(np.array(np.round(np.linspace(0, 255, X.shape[0])), dtype=np.int32))
    C = C[:, 0:3]

    plt.figure(figsize=(20, 5))
    #Plot distance function sampled on a grid
    plt.subplot(141)
    res = 400
    thetagrid = np.linspace(0, 2*np.pi, res)
    [thetagrid, phigrid] = np.meshgrid(thetagrid, thetagrid)
    if distance:
        obsfn = getKleinDistance(x1, thetagrid, phigrid, alpha_theta, alpha_phi, L1)
    else:
        obsfn = np.cos(2*thetagrid) + np.cos(thetagrid)*np.sin(phigrid) + np.cos(phigrid)
    plt.imshow(obsfn, cmap = 'gray', extent = (0, 2*np.pi, 2*np.pi, 0), interpolation = 'none')
    #Show observation points
    if distance:
        plt.scatter([x1[0], x2[0]], [x1[1], x2[1]])
    plt.plot([0, 2*np.pi], [np.pi, np.pi], lineWidth=2, linestyle='--')
    
    AW = 0.05
    AXW = 0.005
    ax = plt.gca()
    print(theta[0])
    #for i in np.arange(X.shape[0]):#np.random.permutation(X.shape[0])[0:50]:
    Theta = np.zeros((phi.size, 2))
    Theta[:, 0] = theta
    Theta[:, 1] = phi
    Theta = Theta[0:X.shape[0], :]
    idxperm = getGreedyPermEuclidean(Theta, 100)['perm']

    for i in idxperm:
        p1 = np.array([theta[i], phi[i]])
        p2 = np.array([theta[i+1], phi[i+1]])
        rx = 0.5*(p2 - p1)
        if np.sqrt(np.sum(rx**2)) > 1:
            continue
        ax.arrow(p1[0], p1[1], rx[0], rx[1], head_width = AW, head_length = AW, fc = C[i, :], ec = C[i, :], width = AXW)
    plt.xticks([0, 2*np.pi], ["0", "$2\\pi$"])
    plt.yticks([0, np.pi, 2*np.pi], ["0", "$\\pi$", "$2\\pi$"])
    plt.gca().invert_yaxis()
    plt.title("Observation Function")

    #plt.subplot2grid((1, 4), (0, 1), colspan=3)
    plt.subplot(142)
    drawLineColored(np.arange(X.shape[0]), x[0:X.shape[0]], C)
    ax = plt.gca()
    ax.set_axis_bgcolor((0.15, 0.15, 0.15))
    plt.title("Time Series")
    plt.xlabel("Sample Number")
    plt.ylabel("Observation Function")

    plt.subplot(143)
    r = Rips(maxdim=2)
    r.fit_transform(X)
    r.plot(show=False)
    plt.title("$\\mathbb{Z}/2$")

    plt.subplot(144)
    r = Rips(maxdim=2, coeff=3)
    r.fit_transform(X)
    r.plot(show=False)
    plt.title("$\\mathbb{Z}/3$")

    if distance:
        plt.savefig("KleinDist.svg", bbox_inches='tight')
    else:
        plt.savefig("KleinFourier.svg", bbox_inches='tight')


if __name__ == '__main__':
    #makeTorusFigure()
    #makeKleinProjCoordsFigure()
    #makeTorusDistanceFigure()
    makeKleinFigure(distance = True)