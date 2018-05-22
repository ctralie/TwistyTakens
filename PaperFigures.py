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

def makeTorusCircularCoordsFigure():
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

    fig = plt.figure(figsize=(20, 5))

    res = ProjCoords(X, NLandmarks, cocycle_idx = cocycle_idx, proj_dim=2, perc = 0.9)
    SFinal = getStereoProjCodim1(res['X'])

    plt.subplot2grid((1, 4), (0, 0), colspan=2)
    drawLineColored(np.arange(M), x[0:M], C)
    ax = plt.gca()
    ax.set_axis_bgcolor((0.15, 0.15, 0.15))
    plt.xlabel("Sample Number")
    plt.ylabel("Observation")
    plt.title("Klein Bottle Time Series")

    plt.subplot(143)
    N = int(SFinal.shape[0]/2)
    C1 = C[0:N, :]
    SFinal1 = SFinal[0:N, :]
    SFinal2 = SFinal[N::, :]
    C2 = C[N::, :]
    #plotRP2Stereo(SFinal1, C1)
    ax = plt.gca()
    AW = 0.03
    AXW = 0.005
    for i in range(0, C1.shape[0]-1):
        x1 = SFinal[i, :]
        rx = SFinal[i+1, :] - x1
        rx = 0.5*rx
        if np.sqrt(np.sum(rx**2)) > 0.5:
            continue
        ax.arrow(x1[0], x1[1], rx[0], rx[1], head_width = AW, head_length = AW, fc = C1[i, :], ec = C1[i, :], width = AXW)
    t = np.linspace(0, 2*np.pi, 200)
    plt.plot(np.cos(t), np.sin(t), 'c')
    plt.axis('equal')
    ax = plt.gca()
    ax.set_axis_bgcolor((0.15, 0.15, 0.15))
    ax.set_xticks([])
    ax.set_yticks([])
    plt.title("Projective Coordinates First Half")

    plt.subplot(144)
    #plotRP2Stereo(SFinal2, C2)
    ax = plt.gca()
    for i in range(0, C2.shape[0]-1):
        x1 = SFinal2[i, :]
        rx = SFinal2[i+1, :] - x1
        rx = 0.5*rx
        if np.sqrt(np.sum(rx**2)) > 0.5:
            continue
        ax.arrow(x1[0], x1[1], rx[0], rx[1], head_width = AW, head_length = AW, fc = C2[i, :], ec = C2[i, :], width = AXW)
    plt.plot(np.cos(t), np.sin(t), 'c')
    plt.axis('equal')
    ax = plt.gca()
    ax.set_axis_bgcolor((0.15, 0.15, 0.15))
    ax.set_xticks([])
    ax.set_yticks([])
    plt.title("Projective Coordinates Second Half")

    plt.savefig("KleinProj.svg", bbox_inches='tight')


    X = getSlidingWindow(x, 10, subsample, 1)


def makeTorusFigure(distance = True, ratio = np.sqrt(2)):
    subsample = 4.0
    t = np.arange(500)/subsample
    theta = np.mod(ratio*t, 2*np.pi)
    phi = np.mod(t, 2*np.pi)
    T = 30
    Win = T

    np.random.seed(4)
    x1 = [6, np.pi]

    # Get the correct distance under the quotient
    if distance:
        x = getTorusDistance(x1, theta, phi)
    else:
        x = np.cos(theta) + np.cos(phi)
    X = getSlidingWindowNoInterp(x, Win)

    c = plt.get_cmap('Spectral')
    C = c(np.array(np.round(np.linspace(0, 255, X.shape[0])), dtype=np.int32))
    C = C[:, 0:3]

    plt.figure(figsize=(14, 4))
    #Plot distance function sampled on a grid
    plt.subplot(141)
    res = 400
    thetagrid = np.linspace(0, 2*np.pi, res)
    [thetagrid, phigrid] = np.meshgrid(thetagrid, thetagrid)
    if distance:
        obsfn = getTorusDistance(x1, thetagrid, phigrid)
    else:
        obsfn = np.cos(thetagrid) + np.cos(phigrid)
    plt.imshow(obsfn, cmap = 'gray', extent = (0, 2*np.pi, 2*np.pi, 0), interpolation = 'none')
    #Show observation points
    if distance:
        plt.scatter([x1[0]], [x1[1]])
    
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
    plt.xticks([0, np.pi, 2*np.pi], ["0", "$\\pi$", "$2\\pi$"])
    plt.yticks([0, np.pi, 2*np.pi], ["0", "$\\pi$", "$2\\pi$"])
    plt.gca().invert_yaxis()
    plt.xlabel("$X$")
    plt.ylabel("$Y$")
    plt.title("Observation Function")

    #plt.subplot2grid((1, 4), (0, 1), colspan=3)
    plt.subplot(132)
    ax = plt.gca()
    drawLineColored(np.arange(X.shape[0]), x[0:X.shape[0]], C)
    #Draw sliding window
    y1, y2 = np.min(x), np.max(x)
    pad = 0.05*(y2-y1)
    c = np.array([1.0, 0.737, 0.667])
    ax.arrow(Win, y2+0.3*pad, 20, 0, head_width = AW, head_length = 10, fc = c, ec = c, width = AXW)
    ax.arrow(Win, y1-0.3*pad, 20, 0, head_width = AW, head_length = 10, fc = c, ec = c, width = AXW)
    y1 = y1 - pad
    y2 = y2 + pad
    plt.plot([0, Win], [y1, y1], c=c)
    plt.plot([0, Win], [y2, y2], c=c)
    plt.plot([0, 0], [y1, y2], c=c)
    plt.plot([Win, Win], [y1, y2], c=c)
    ax.set_axis_bgcolor((0.15, 0.15, 0.15))
    plt.title("Time Series")
    plt.xlabel("Sample Number")
    plt.ylabel("Observation Function")

    plt.subplot(133)
    r = Rips(maxdim=2)
    r.fit_transform(X)
    r.plot(show=False)
    plt.title("$\\mathbb{Z}/2$")

    if distance:
        plt.savefig("TorusDist.svg", bbox_inches='tight')
    else:
        plt.savefig("TorusFourier.svg", bbox_inches='tight')

def makeKleinFigure(distance = True):
    eps = 0.02
    T1 = 20
    Win = T1*2
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
    X = getSlidingWindowNoInterp(x, Win)

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
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Observation Function")

    #plt.subplot2grid((1, 4), (0, 1), colspan=3)
    plt.subplot(142)
    ax = plt.gca()
    drawLineColored(np.arange(X.shape[0]), x[0:X.shape[0]], C)
    #Draw sliding window
    y1, y2 = np.min(x), np.max(x)
    pad = 0.05*(y2-y1)
    c = np.array([1.0, 0.737, 0.667])
    ax.arrow(Win, y2+0.3*pad, 20, 0, head_width = AW, head_length = 10, fc = c, ec = c, width = AXW)
    ax.arrow(Win, y1-0.3*pad, 20, 0, head_width = AW, head_length = 10, fc = c, ec = c, width = AXW)
    y1 = y1 - pad
    y2 = y2 + pad
    plt.plot([0, Win], [y1, y1], c=c)
    plt.plot([0, Win], [y2, y2], c=c)
    plt.plot([0, 0], [y1, y2], c=c)
    plt.plot([Win, Win], [y1, y2], c=c)

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

def makeSphereFigure():
    np.random.seed(6)
    u = np.random.randn(3, 1)
    u = u/np.sqrt(np.sum(u**2))

    T = 30
    NPeriods = 30
    Win = T
    NPeriods = 30
    N = T*NPeriods
    theta = np.linspace(0, 2*np.pi*NPeriods, N)
    theta = np.mod(theta, 2*np.pi)
    phi = np.linspace(-np.pi/2, np.pi/2, N)

    N = phi.size
    XTraj= np.zeros((N, 3))
    XTraj[:, 0] = np.cos(theta)*np.cos(phi)
    XTraj[:, 1] = np.sin(theta)*np.cos(phi)
    XTraj[:, 2] = np.sin(phi)

    x = getSphereTimeSeries(theta, phi, u, geodesic=True)
    X = getSlidingWindowNoInterp(x, T)
    pca = PCA(n_components=3)
    Y = pca.fit_transform(X)
    fig = plt.figure(figsize=(15, 10))

    c = plt.get_cmap('Spectral')
    C = c(np.array(np.round(np.linspace(0, 255, X.shape[0])), dtype=np.int32))
    C = C[:, 0:3]
    sio.savemat("SphereData.mat", {"XTraj":XTraj, "u":u, "C":C})

    plt.subplot(234)
    plt.title("Vector Field / Distance View 1")
    plt.axis('off')

    plt.subplot(235)
    plt.title("Vector Field / Distance View 2")
    plt.axis('off')

    plt.subplot2grid((2, 3), (0, 0), colspan=2)
    drawLineColored(np.arange(X.shape[0]), x[0:X.shape[0]], C)
    ax = plt.gca()
    #Draw sliding window
    AW = 0.05
    AXW = 0.005
    y1, y2 = np.min(x), np.max(x)
    pad = 0.05*(y2-y1)
    c = np.array([1.0, 0.737, 0.667])
    ax.arrow(Win, y2+0.3*pad, 20, 0, head_width = AW, head_length = 10, fc = c, ec = c, width = AXW)
    ax.arrow(Win, y1-0.3*pad, 20, 0, head_width = AW, head_length = 10, fc = c, ec = c, width = AXW)
    y1 = y1 - pad
    y2 = y2 + pad
    plt.plot([0, Win], [y1, y1], c=c)
    plt.plot([0, Win], [y2, y2], c=c)
    plt.plot([0, 0], [y1, y2], c=c)
    plt.plot([Win, Win], [y1, y2], c=c)
    ax.set_axis_bgcolor((0.15, 0.15, 0.15))
    plt.title("Time Series")
    plt.xlabel("Sample Number")
    plt.ylabel("Observation Function")

    plt.subplot(233)
    r = Rips(maxdim=2)
    r.fit_transform(X)
    r.plot(show=False)
    plt.title("$\\mathbb{Z}/2$")

    ax = fig.add_subplot(236, projection='3d')
    ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], c=C)
    plt.axis('equal')
    ax.set_axis_bgcolor((0.15, 0.15, 0.15))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    var = 100*np.sum(pca.explained_variance_ratio_[0:3])
    plt.title("3D PCA %.3g%s Variance Explained"%(var, "%"))
    plt.savefig("SphereDist.svg", bbox_inches='tight')


def makeRP2Figure(randomSeed = -1):
    np.random.seed(6)
    u = np.random.randn(3, 1)
    u = u/np.sqrt(np.sum(u**2))

    cocycle_idx = [0]
    NLandmarks=200
    T = 30
    Win = T
    NPeriods = 30
    N = T*NPeriods
    theta = np.linspace(0, 2*np.pi*NPeriods, N)
    theta = np.mod(theta, 2*np.pi)
    phi = np.linspace(-np.pi/2, 0, N)

    N = phi.size
    XTraj= np.zeros((N, 3))
    XTraj[:, 0] = np.cos(theta)*np.cos(phi)
    XTraj[:, 1] = np.sin(theta)*np.cos(phi)
    XTraj[:, 2] = np.sin(phi)
    x = np.dot(XTraj, u)
    x = np.abs(x)
    x[x > 1] = 1
    x = np.arccos(x).flatten()
    print(len(x))

    X = getSlidingWindowNoInterp(x, Win)
    pca = PCA(n_components=3)
    Y = pca.fit_transform(X)
    fig = plt.figure(figsize=(15, 10))

    c = plt.get_cmap('Spectral')
    C = c(np.array(np.round(np.linspace(0, 255, X.shape[0])), dtype=np.int32))
    C = C[:, 0:3]

    #Draw vector field
    plt.subplot(231)
    plt.title("Vector Field")
    ax = plt.gca()
    thetagrid = np.linspace(0, 2*np.pi, 200)
    phigrid = np.linspace(-np.pi/2, 0, 200)
    [thetagrid, phigrid] = np.meshgrid(thetagrid, phigrid)
    Y = np.zeros((thetagrid.size, 3))
    thetagrid, phigrid = thetagrid.flatten(), phigrid.flatten()
    Y[:, 0] = np.cos(thetagrid)*np.cos(phigrid)
    Y[:, 1] = np.sin(thetagrid)*np.cos(phigrid)
    Y[:, 2] = np.sin(phigrid)
    y = np.dot(Y, u)
    y = np.abs(y)
    y[y > 1] = 1
    y = np.arccos(y).flatten()
    y = y/np.max(y)
    c = plt.get_cmap('gray')
    CObj = c(np.array(np.round(y*255), dtype=np.int32))
    CObj = CObj[:, 0:3]
    idxperm = getGreedyPermEuclidean(Y[:, 0:2], 3000)['perm']
    plt.scatter(Y[idxperm, 0], Y[idxperm, 1], c=CObj[idxperm, :])
    
    #Now draw vector field
    #plt.scatter(XTraj[0:X.shape[0], 0], XTraj[0:X.shape[0], 1], c=C)
    AW = 0.05
    AXW = 0.005
    idxperm = getGreedyPermEuclidean(XTraj[0:X.shape[0], 0:2], 200)['perm']
    for i in idxperm:
        p1 = XTraj[i, 0:2]
        p2 = XTraj[i+1, 0:2]
        rx = 0.6*(p2 - p1)
        if np.sqrt(np.sum(rx**2)) > 1:
            continue
        ax.arrow(p1[0], p1[1], rx[0], rx[1], head_width = AW, head_length = AW, fc = C[i, :], ec = C[i, :], width = AXW)
    ax.arrow(-0.1, 1.05, 0.001, 0, head_width = 0.15, head_length = 0.2, fc = 'c', ec = 'c', width = 0)
    ax.arrow(0.1, -1.05, -0.001, 0, head_width = 0.15, head_length = 0.2, fc = 'c', ec = 'c', width = 0)
    t = np.linspace(0, 2*np.pi, 200)
    plt.plot(1.05*np.cos(t), 1.05*np.sin(t), 'c')
    plt.axis('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.scatter(u[0], u[1], 100, 'r')

    ax.set_axis_bgcolor((0.15, 0.15, 0.15))

    plt.subplot2grid((2, 3), (0, 1), colspan=2)
    drawLineColored(np.arange(X.shape[0]), x[0:X.shape[0]], C)
    ax = plt.gca()
    #Draw sliding window
    AW = 0.05
    AXW = 0.005
    y1, y2 = np.min(x), np.max(x)
    pad = 0.05*(y2-y1)
    c = np.array([1.0, 0.737, 0.667])
    ax.arrow(Win, y2+0.3*pad, 20, 0, head_width = AW, head_length = 10, fc = c, ec = c, width = AXW)
    ax.arrow(Win, y1-0.3*pad, 20, 0, head_width = AW, head_length = 10, fc = c, ec = c, width = AXW)
    y1 = y1 - pad
    y2 = y2 + pad
    plt.plot([0, Win], [y1, y1], c=c)
    plt.plot([0, Win], [y2, y2], c=c)
    plt.plot([0, 0], [y1, y2], c=c)
    plt.plot([Win, Win], [y1, y2], c=c)
    ax.set_axis_bgcolor((0.15, 0.15, 0.15))
    plt.title("Time Series")
    plt.xlabel("Sample Number")
    plt.ylabel("Observation Function")

    plt.subplot(234)
    res = ProjCoords(X, NLandmarks, cocycle_idx = cocycle_idx, proj_dim=2, perc = 0.9)
    SFinal = getStereoProjCodim1(res['X'], randomSeed)
    #plotRP2Stereo(SFinal, C)
    ax = plt.gca()

    idxperm = getGreedyPermEuclidean(SFinal[0:-1, :], 200)['perm']
    for i in idxperm:
        p1 = SFinal[i, 0:2]
        p2 = SFinal[i+1, 0:2]
        rx = 0.6*(p2 - p1)
        if np.sqrt(np.sum(rx**2)) > 1:
            continue
        ax.arrow(p1[0], p1[1], rx[0], rx[1], head_width = AW, head_length = AW, fc = C[i, :], ec = C[i, :], width = AXW)
    ax.arrow(-0.1, 1.05, 0.001, 0, head_width = 0.15, head_length = 0.2, fc = 'c', ec = 'c', width = 0)
    ax.arrow(0.1, -1.05, -0.001, 0, head_width = 0.15, head_length = 0.2, fc = 'c', ec = 'c', width = 0)
    t = np.linspace(0, 2*np.pi, 200)
    plt.plot(1.05*np.cos(t), 1.05*np.sin(t), 'c')
    plt.axis('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_axis_bgcolor((0.15, 0.15, 0.15))

    plt.title("Sliding Window Projective Coordinates")

    idxperm = getGreedyPermEuclidean(X, 400)['perm']
    X = X[idxperm, :]

    plt.subplot(235)
    r = Rips(maxdim=2)
    r.fit_transform(X)
    r.plot(show=False)
    plt.title("$\\mathbb{Z}/2$")

    plt.subplot(236)
    r = Rips(maxdim=2, coeff=3)
    r.fit_transform(X)
    r.plot(show=False)
    plt.title("$\\mathbb{Z}/3$")

    plt.savefig("ProjDist%i.svg"%randomSeed, bbox_inches='tight')


if __name__ == '__main__':
    #makeTorusCircularCoordsFigure()
    #makeKleinProjCoordsFigure()
    #makeTorusFigure(distance = True)
    #makeTorusFigure(distance = False)
    #makeKleinFigure(distance = True)
    #makeSphereFigure()
    makeRP2Figure(randomSeed = 48)