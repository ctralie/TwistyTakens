import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.io as sio
from SlidingWindow import *
from Utilities import *
from sklearn.decomposition import PCA
from Manifolds2D import *
from ripser import Rips, ripser, plot_dgms
from matplotlib.patches import Polygon
import sys 
import time
sys.path.append("DREiMac")
from ProjectiveCoordinates import *
from CircularCoordinates import *

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
    ax.set_facecolor((0.15, 0.15, 0.15))
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
    ax.set_facecolor((0.15, 0.15, 0.15))
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
    ax.set_facecolor((0.15, 0.15, 0.15))
    ax.set_xticks([])
    ax.set_yticks([])
    plt.title("Projective Coordinates Second Half")

    plt.savefig("KleinProj.svg", bbox_inches='tight')

def cocyclestr(ccl):
    s = "%i"%ccl[0]
    for i in range(1, len(ccl)):
        s = s + " + %i"%ccl[i]
    return s

def makeTorusFigure(distance = True, ratio = np.sqrt(2), do_scatterplots = False):
    prows = 1
    if do_scatterplots:
        prows = 2
    T1 = 30
    NPeriods = 30
    thetagt = np.linspace(0, 2*np.pi*NPeriods, T1*NPeriods)
    phigt = thetagt * np.sqrt(2)
    thetagt = np.mod(thetagt, 2*np.pi)
    phigt = np.mod(phigt, 2*np.pi)
    Win = int(T1*1.5)
    x1 = [6, np.pi]
    prime = 41
    cocycle1 = [0]
    cocycle2 = [0, 1]

    # Get the correct distance under the quotient
    if distance:
        x = getTorusDistance(x1, thetagt, phigt)
    else:
        x = np.cos(thetagt) + np.cos(phigt)
    X = getSlidingWindowNoInterp(x, Win)

    c = plt.get_cmap('Spectral')
    C = c(np.array(np.round(np.linspace(0, 255, X.shape[0])), dtype=np.int32))
    C = C[:, 0:3]

    plt.figure(figsize=(18, 4*prows))
    #Plot distance function sampled on a grid
    plt.subplot(prows, 4, 1)
    res = 100
    thetagrid = np.linspace(0, 2*np.pi, res)
    [thetagrid, phigrid] = np.meshgrid(thetagrid, thetagrid)
    if distance:
        obsfn = getTorusDistance(x1, thetagrid, phigrid)
    else:
        obsfn = np.cos(thetagrid) + np.cos(phigrid)
    plt.imshow(obsfn, cmap = 'gray', extent = (0, 2*np.pi, 2*np.pi, 0), interpolation = 'none')
    
    AW = 0.05
    AXW = 0.005
    ax = plt.gca()
    """
    Theta = np.zeros((phigt.size, 2))
    Theta[:, 0] = thetagt
    Theta[:, 1] = phigt
    Theta = Theta[0:X.shape[0], :]
    idxperm = getGreedyPermEuclidean(Theta, 400)['perm']
    """
    for i in range(X.shape[0] - 1):
        p1 = np.array([thetagt[i], phigt[i]])
        p2 = np.array([thetagt[i+1], phigt[i+1]])
        rx = 0.5*(p2 - p1)
        if np.sqrt(np.sum(rx**2)) > 1:
            continue
        ax.arrow(p1[0], p1[1], rx[0], rx[1], head_width = AW, head_length = AW, fc = C[i, :], ec = C[i, :], width = AXW)
    
    #Show observation points
    if distance:
        plt.scatter([x1[0]], [x1[1]])
    
    plt.xticks([0, np.pi, 2*np.pi], ["0", "$\\pi$", "$2\\pi$"])
    plt.yticks([0, np.pi, 2*np.pi], ["0", "$\\pi$", "$2\\pi$"])
    plt.gca().invert_yaxis()
    plt.xlabel("$X$")
    plt.ylabel("$Y$")
    plt.title("Observation Function")

    #plt.subplot2grid((1, 4), (0, 1), colspan=3)
    plt.subplot(prows, 4, 2)
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
    ax.set_facecolor((0.15, 0.15, 0.15))
    plt.title("Time Series")
    plt.xlabel("Sample Number")
    plt.ylabel("Observation Function")

    #Now compute circular coordinates
    NLandmarks = 400
    res = CircularCoords(X, NLandmarks, prime=prime, maxdim=2, cocycle_idx = cocycle1, perc=0.5)
    rips = res["rips"]
    theta = res["thetas"]
    idx_p1 = res["idx_p1"]
    dgm1 = 2*res["dgm1"]
    phi = CircularCoords(X, NLandmarks, prime=prime, cocycle_idx = cocycle2, perc=0.5)["thetas"]

    plt.subplot(prows, 4, 3)
    rips.plot(show=False)
    plt.text(dgm1[idx_p1[0], 0], dgm1[idx_p1[0], 1], "0")
    plt.text(dgm1[idx_p1[1], 0], dgm1[idx_p1[1], 1], "1")
    plt.title("$\\mathbb{Z} / %i$, %i Landmarks"%(prime, NLandmarks))

    # Unwrap circular coordinates and shift to start at (0, 0) with positive slope
    theta = np.unwrap(theta) + 1
    phi = np.unwrap(phi) + 1
    avgslope = (phi[-1]-0 + 1)/(theta[-1]-theta[0])
    avgslope = np.abs(avgslope)
    if theta[-1] < theta[0] + 1:
        theta *= -1
    theta = np.mod(theta-theta[0], 2*np.pi)
    if phi[-1] < phi[0]:
        phi *= -1
    phi = np.mod(phi-phi[0], 2*np.pi)
    plt.subplot(prows, 4, 4)
    ax = plt.gca()
    for i in range(theta.size - 1):
        p1 = np.array([theta[i], phi[i]])
        p2 = np.array([theta[i+1], phi[i+1]])
        rx = 0.5*(p2 - p1)
        if np.sqrt(np.sum(rx**2)) > 1:
            continue
        ax.arrow(p1[0], p1[1], rx[0], rx[1], head_width = AW, head_length = AW, fc = C[i, :], ec = C[i, :], width = AXW)
    plt.xticks([0, 2*np.pi], ["0", "$2\\pi$"])
    plt.yticks([0, 2*np.pi], ["0", "$2\\pi$"])
    plt.xlabel("Cocycle %s"%cocyclestr(cocycle1))
    plt.ylabel("Cocycle %s"%cocyclestr(cocycle2))
    plt.title("Circular Coordinates, Slope = %.3g"%avgslope)
    ax.set_facecolor((0.15, 0.15, 0.15))

    theta, thetagt = np.unwrap(theta), np.unwrap(thetagt)
    phi, phigt = np.unwrap(phi), np.unwrap(phigt)

    if do_scatterplots:
        plt.subplot(245)
        plt.scatter(thetagt[0:theta.size], theta)
        plt.xlabel("X")
        plt.ylabel("Cocycle %s"%cocyclestr(cocycle1))
        plt.subplot(246)
        plt.scatter(phigt[0:phi.size], phi)
        plt.xlabel("Y")
        plt.ylabel("Cocycle %s"%cocyclestr(cocycle2))

    if distance:
        plt.savefig("TorusDist_%i.svg"%NLandmarks, bbox_inches='tight')
    else:
        plt.savefig("TorusFourier_%i.svg"%NLandmarks, bbox_inches='tight')
    


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
        #x = np.cos(theta)*np.sin(phi) + np.cos(phi)
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
        #obsfn = np.cos(thetagrid)*np.sin(phigrid) + np.cos(phigrid)
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

    ax.set_facecolor((0.15, 0.15, 0.15))
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
    ax.set_facecolor((0.15, 0.15, 0.15))
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
    ax.set_facecolor((0.15, 0.15, 0.15))
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

    ax.set_facecolor((0.15, 0.15, 0.15))

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
    ax.set_facecolor((0.15, 0.15, 0.15))
    plt.title("Time Series")
    plt.xlabel("Sample Number")
    plt.ylabel("Observation Function")

    plt.subplot(234)
    prime = 41
    res = ProjCoords(X, NLandmarks, prime=prime, cocycle_idx = cocycle_idx, proj_dim=2, perc = 0.9)
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
    ax.set_facecolor((0.15, 0.15, 0.15))

    plt.title("Sliding Window Projective Coordinates")

    idxperm = getGreedyPermEuclidean(X, 400)['perm']
    X = X[idxperm, :]

    plt.subplot(235)
    r = Rips(maxdim=2)
    r.fit_transform(X)
    r.plot(show=False)
    plt.title("$\\mathbb{Z}/%i$"%prime)

    plt.subplot(236)
    r = Rips(maxdim=2, coeff=3)
    r.fit_transform(X)
    r.plot(show=False)
    plt.title("$\\mathbb{Z}/%i$"%prime)

    plt.savefig("ProjDist%i.svg"%randomSeed, bbox_inches='tight')



def make2HoledTorusFigure():
    x0 = np.array([0.2, 0.2])
    dx = 0.1*np.array([1.0, np.sqrt(3)/2])
    res = get2HoledTorusTraj(x0, dx, 750)
    Win = 40
    endpts, X = res['endpts'], res['X']

    c = plt.get_cmap('Spectral')
    C = c(np.array(np.round(np.linspace(0, 255, X.shape[0])), dtype=np.int32))
    C = C[:, 0:3]
    y = get2HoledTorusDist(X, x0, endpts)

    Y = getSlidingWindowNoInterp(y, Win)
    sio.savemat("2HoledTorus.mat", {"X":Y})
    tic = time.time()
    dgms = ripser(Y, maxdim=2)['dgms']
    print("Elapsed Time: %.3g"%(time.time()-tic))

    # Create distance image
    normals = endpts[1::, :] - endpts[0:-1, :]
    normals[:, 0], normals[:, 1] = normals[:, 1], -normals[:, 0]
    pix = np.linspace(-1.1, 1.1, 200)
    [I, J] = np.meshgrid(pix, pix)
    XDist = np.zeros((I.size, 2))
    XDist[:, 0] = I.flatten()
    XDist[:, 1] = J.flatten()
    dist = get2HoledTorusDist(XDist, x0, endpts)
    dist = np.reshape(dist, I.shape)
    insideOctagon = np.ones(XDist.shape[0])
    for i in range(8):
        dot = (XDist - endpts[i, :]).dot(normals[i, :])
        insideOctagon *= (dot < 0)
    insideOctagon = np.reshape(insideOctagon, dist.shape)
    dist[insideOctagon == 0] = 0


    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.imshow(dist, cmap = 'gray', extent = [pix[0], pix[-1], pix[-1], pix[0]])
    plt.scatter(x0[0], x0[0], 100, 'r', zorder=10)
    ax = plt.gca()
    AW = 0.05
    AXW = 0.0025
    idxperm = getGreedyPermEuclidean(X[0:-1, :], 200)['perm']
    for i in idxperm:
        p1 = X[i, :]
        p2 = X[i+1, :]
        rx = 0.6*(p2 - p1)
        if np.sqrt(np.sum(rx**2)) > 0.5:
            continue
        ax.arrow(p1[0], p1[1], rx[0], rx[1], head_width = AW, head_length = AW, fc = C[i, :], ec = C[i, :], width = AXW)
    # Plot octagon
    AW = 0.1
    AXW = 0.005
    for i in range(8):
        p1 = endpts[i, :] + 0.5*(endpts[i+1, :] - endpts[i, :])
        p2 = endpts[i, :] + 0.51*(endpts[i+1, :] - endpts[i, :])
        rx = p2 - p1
        if i >= 4:
            rx *= -1
            p1, p2 = p2, p1
        k = i%4
        plt.plot(endpts[i:i+2, 0], endpts[i:i+2, 1], c='C%i'%k, linewidth=2)
        ax.arrow(p1[0], p1[1], rx[0], rx[1], head_width = AW, head_length = AW, fc = 'C%i'%k, ec = 'C%i'%k, width = AXW, zorder=10)
    plt.xlim([-1.1, 1.1])
    plt.ylim([-1.1, 1.1])
    plt.axis('equal')
    plt.title("Observation Function")

    plt.subplot(132)
    drawLineColored(np.arange(y.size), y, C)
    ax = plt.gca()
    ax = plt.gca()
    #Draw sliding window
    AW = 0.05
    AXW = 0.005
    y1, y2 = np.min(y), np.max(y)
    pad = 0.05*(y2-y1)
    c = np.array([1.0, 0.737, 0.667])
    ax.arrow(Win, y2+0.3*pad, 20, 0, head_width = AW, head_length = 20, fc = c, ec = c, width = AXW)
    ax.arrow(Win, y1-0.3*pad, 20, 0, head_width = AW, head_length = 20, fc = c, ec = c, width = AXW)
    y1 = y1 - pad
    y2 = y2 + pad
    plt.plot([0, Win], [y1, y1], c=c)
    plt.plot([0, Win], [y2, y2], c=c)
    plt.plot([0, 0], [y1, y2], c=c)
    plt.plot([Win, Win], [y1, y2], c=c)
    ax.set_facecolor((0.15, 0.15, 0.15))
    plt.title("Time Series")
    plt.xlabel("Sample Number")
    plt.ylabel("Observation Function")

    plt.subplot(133)
    plot_dgms(dgms)
    plt.title("Persistence Diagrams $\mathbb{Z}/2$")

    plt.savefig("2HoledTorus.svg", bbox_inches='tight')


def makeCircularCoordsFigure():
    np.random.seed(2)
    prime = 49
    N = 10000 #Number of initial points in (theta, phi) space
    NPoints = 1000 #Number of points to evenly subsample in 3D
    R = 6
    r = 2
    theta = np.random.rand(N)*2*np.pi
    phi = np.random.rand(N)*2*np.pi
    X = np.zeros((N, 3))
    X[:, 0] = (R + r*np.cos(theta))*np.cos(phi)
    X[:, 1] = (R + r*np.cos(theta))*np.sin(phi)
    X[:, 2] = r*np.sin(theta)
    # Evenly subsample the points geometrically
    X = getGreedyPermEuclidean(X, NPoints)['Y']
    xr = [np.min(X.flatten()), np.max(X.flatten())]

    NLandmarks = 400
    res = CircularCoords(X, NLandmarks, prime=prime, maxdim=2, cocycle_idx = [0], perc=0.5)
    rips = res["rips"]
    theta = res["thetas"]
    idx_p1 = res["idx_p1"]
    dgm1 = 2*res["dgm1"]
    phi = CircularCoords(X, NLandmarks, prime=prime, cocycle_idx = [1], perc=0.5)["thetas"]

    plt.figure(figsize=(18, 4))
    ax = plt.subplot(141, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=5)
    ax.set_xlim(xr)
    ax.set_ylim(xr)
    ax.set_zlim(xr)
    ax.view_init(elev=42, azim=-46)
    plt.title("Point Cloud")

    plt.subplot(142)
    rips.plot(show=False)
    plt.text(dgm1[idx_p1[0], 0]+0.02, dgm1[idx_p1[0], 1], "0")
    plt.text(dgm1[idx_p1[1], 0]+0.02, dgm1[idx_p1[1], 1], "1")
    plt.title("Persistence Diagram")

    theta = theta - np.min(theta)
    theta = theta/np.max(theta)
    phi = phi - np.min(phi)
    phi = phi/np.max(phi)
    phi = np.mod(phi + 0.75, 1)
    c = plt.get_cmap('afmhot')
    C1 = c(np.array(np.round(255*theta), dtype=np.int32))
    C1 = C1[:, 0:3]
    C2 = c(np.array(np.round(255*phi), dtype=np.int32))
    C2 = C2[:, 0:3]

    ax = plt.subplot(143, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=5, c = C1)
    ax.set_xlim(xr)
    ax.set_ylim(xr)
    ax.set_zlim(xr)
    ax.view_init(elev=42, azim=-46)
    plt.title("Circular Coordinates Cocycle 0")

    ax = plt.subplot(144, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=5, c = C2)
    ax.set_xlim(xr)
    ax.set_ylim(xr)
    ax.set_zlim(xr)
    ax.view_init(elev=42, azim=-46)
    plt.title("Circular Coordinates Cocycle 1")

    plt.savefig("CircularCoordinatesExample.svg", bbox_inches='tight')

def drawKleinIdentifications(arrowcolor):
    hw = 0.3*1.5
    hl = 0.4*1.5
    pad = 0.2
    ax = plt.gca()
    plt.plot([0, 2*np.pi], [0, 0], c=arrowcolor, linewidth=4)
    plt.plot([0, 2*np.pi], 2*[2*np.pi], c=arrowcolor, linewidth=4)
    plt.plot([0, 0], [0, 2*np.pi], c=arrowcolor, linewidth=4)
    plt.plot(2*[2*np.pi], [0, 2*np.pi], c=arrowcolor, linewidth=4)
    plt.plot([0, 2*np.pi], 2*[np.pi], c=arrowcolor, linewidth=2, linestyle='--')

    ax.arrow(np.pi-pad, 0, pad, 0, head_width = hw, head_length = hl, fc = arrowcolor, ec = arrowcolor, width = 0)
    ax.arrow(np.pi-pad, 2*np.pi, pad, 0, head_width = hw, head_length = hl, fc = arrowcolor, ec = arrowcolor, width = 0)
    ax.arrow(0, np.pi-pad, 0, pad/2, head_width = hw, head_length = hl, fc = arrowcolor, ec = arrowcolor, width = 0)
    ax.arrow(2*np.pi, np.pi-pad, 0, pad/2, head_width = hw, head_length = hl, fc = arrowcolor, ec = arrowcolor, width = 0)
    


def makeProjCoordsFigure(seed1, perc = 0.9, cocycle_idx = [0, 1]):
    np.random.seed(seed1)
    N = 20000 #Number of initial points in (theta, phi) space
    NLandmarks=200
    bgcolor = (0.2, 0.2, 0.2)

    theta = 2*np.pi*np.random.rand(N)
    phi = np.pi*np.random.rand(N)
    D = np.zeros((N, N))
    for i in range(D.shape[0]):
        D[i, :] = getKleinDistance(np.array([theta[i], phi[i]]), theta, phi)

    # Come up with a colormap that varies along width and length
    cmap1 = plt.get_cmap('bone')
    z = theta - np.min(theta)
    z = z/np.max(z)
    C1 = cmap1(np.array(np.round(255*z), dtype=np.int32))
    C1 = C1[:, 0:3]
    cmap2 = plt.get_cmap('gist_heat')
    z = phi - np.min(phi)
    z = z/np.max(z)
    C2 = cmap2(np.array(np.round(255*z), dtype=np.int32))
    C2 = C2[:, 0:3]
    arrowcolor1 = cmap2(170)
    arrowcolor2 = 'c'

    # Do projective coordinates
    res = ProjCoords(D, NLandmarks, distance_matrix=True, cocycle_idx = cocycle_idx, proj_dim=2, perc = perc)
    rips = res["rips"]
    idx_p1 = res["idx_p1"]
    dgm1 = 2*res["dgm1"]


    plt.figure(figsize=(13, 10))

    plt.subplot(341)
    plt.scatter(theta, phi, 20, c=C1)
    plt.scatter(np.mod(theta+np.pi, 2*np.pi), np.mod(-phi, 2*np.pi), 20, c=C1)
    drawKleinIdentifications(arrowcolor1)
    plt.title("Fund. Domain $x$ Color")
    plt.subplot(342)
    plt.scatter(theta, phi, 20, c=C2)
    plt.scatter(np.mod(theta+np.pi, 2*np.pi), np.mod(-phi, 2*np.pi), 20, c=C2)
    drawKleinIdentifications(arrowcolor2)
    plt.title("Fund. Domain $y$ Color")

    plt.subplot(343)
    rips.plot(show=False)
    plt.text(dgm1[idx_p1[0], 0]+0.02, dgm1[idx_p1[0], 1]+0.02, "0")
    plt.text(dgm1[idx_p1[1], 0]+0.02, dgm1[idx_p1[1], 1]-0.02, "1")
    plt.title("Persistence Diagram $\\mathbb{Z}/2$")
    
    for i, seed2 in enumerate([35, 23]):
        S = getStereoProjCodim1(res['X'], seed2)
        
        plt.subplot(3, 4, 5+i*2)
        plotRP2Stereo(S[phi < np.pi/2, :], C1[phi < np.pi/2, :], arrowcolor=arrowcolor1)
        plt.title("Stereo %i, $x$ color, $y <= \\pi/2$"%(i+1))
        plt.subplot(3, 4, 6+i*2)
        plotRP2Stereo(S[phi > np.pi/2, :], C1[phi > np.pi/2, :], arrowcolor=arrowcolor1)
        plt.title("Stereo %i, $x$ color, $y >= \\pi/2$"%(i+1))

        plt.subplot(3, 4, 9+i*2)
        plotRP2Stereo(S[phi < np.pi/2, :], C2[phi < np.pi/2, :], arrowcolor=arrowcolor2)
        plt.title("Stereo %i, $y$ color, $y < \\pi/2$"%(i+1))
        plt.subplot(3, 4, 10+i*2)
        plotRP2Stereo(S[phi >= np.pi/2, :], C2[phi >= np.pi/2, :], arrowcolor=arrowcolor2)
        plt.title("Stereo %i, $y$ color, $y > \\pi/2$"%(i+1))

    for i in range(1, 13):
        plt.subplot(3, 4, i)
        if i < 3:
            plt.xlabel("$x$")
            plt.ylabel("$y$")
            plt.xticks([0, 2*np.pi], ["0", "$2 \\pi$"])
            plt.yticks([0, np.pi/2, np.pi, 2*np.pi], ["0", "$\\pi/2$", "$\\pi$", "$2\\pi$"])
        elif i < 5:
            continue
        ax = plt.gca()
        ax.set_facecolor(bgcolor)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.savefig("ProjectiveCoordinatesExample.png", bbox_inches='tight')


if __name__ == '__main__':
    #makeKleinProjCoordsFigure()
    #makeTorusFigure(distance = True)
    #makeTorusFigure(distance = False)
    #makeKleinFigure(distance = False)
    #makeSphereFigure()
    #makeRP2Figure(randomSeed = 48)
    #make2HoledTorusFigure()
    #makeCircularCoordsFigure()

    #seed1 = 16, perc = 0.9, cocycle_idx = [0, 1]: 14, 99
    makeProjCoordsFigure(seed1 = 16, perc = 0.7)
