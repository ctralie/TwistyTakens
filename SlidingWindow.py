import numpy as np
import scipy.interpolate as interp
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def getSlidingWindow(x, dim, Tau, dT):
    N = len(x)
    NWindows = int(np.floor((N-dim*Tau)/dT))
    X = np.zeros((NWindows, dim))
    idx = np.arange(N)
    for i in range(NWindows):
        idxx = dT*i + Tau*np.arange(dim)
        start = int(np.floor(idxx[0]))
        end = int(np.ceil(idxx[-1]))+2
        if end >= len(x):
            X = X[0:i, :]
            break
        X[i, :] = interp.spline(idx[start:end+1], x[start:end+1], idxx)
    return X

def getSlidingWindowNoInterp(x, dim):
    N = len(x)
    NWindows = N - dim + 1
    X = np.zeros((NWindows, dim))
    idx = np.arange(N)
    for i in range(NWindows):
        X[i, :] = x[i:i+dim]
    return X
        

if __name__ == '__main__':
    T = 40 #The period in number of samples
    NPeriods = 4 #How many periods to go through
    N = T*NPeriods #The total number of samples
    t = np.linspace(0, 2*np.pi*NPeriods, N+1)[0:N] #Sampling indices in time
    x = np.cos(t) #The final signal
    
    dim = 10
    Tau = 0.2
    dT = 0.1
    X = getSlidingWindow(x, dim, Tau, dT)
    pca = PCA(n_components = 2)
    Y = pca.fit_transform(X)
    eigs = pca.explained_variance_

#    for i in range(X.shape[0]):
#        plt.clf()
#        idxx = dT*i + Tau*np.arange(dim)
#        plt.stem(idxx, X[i, :], 'r')
#        plt.hold(True)
#        start = int(np.floor(idxx[0]))
#        end = int(np.ceil(idxx[-1]))
#        plt.plot(start + np.arange(end-start+1), x[start:end+1])
#        plt.savefig("Window%i.png"%i)

    #Step 4: Plot original signal and PCA of the embedding
    c = plt.get_cmap('Spectral')
    C = c(np.array(np.round(np.linspace(0, 255, Y.shape[0])), dtype=np.int64))
    C = C[:, 0:3]
    plt.figure(figsize=(12, 6))
    ax = plt.subplot(121)
    ax.plot(t, x)
    ax.set_ylim((-2, 2))
    ax.set_title("Original Signal")
    ax.set_xlabel("Phase")
    ax2 = plt.subplot(122)
    ax2.set_title("PCA of Sliding Window Embedding")
    ax2.scatter(Y[:, 0], Y[:, 1], c=C)
    ax2.set_aspect('equal', 'datalim')
    plt.show()
