import numpy as np
from scipy.io import wavfile 
import matplotlib.pyplot as plt 
from Manifolds2D import *

def makeTorusSound(ratio, basefreq = 440, nsec = 3, Fs = 44100):
    """
    Make two pure tones on the torus with a particular ratio
    """
    t = np.arange(nsec*Fs)
    x = np.cos(2*np.pi*basefreq*t/float(Fs))
    x += np.cos(2*np.pi*basefreq*ratio*t/float(Fs))
    x = 0.8*x/np.max(np.abs(x))
    wavfile.write("torus.wav", Fs, x)

def makeKleinSound(basefreq = 440, Fs = 44100, time = 1, eps = 0.02):
    T = int(np.round(Fs/float(basefreq)))
    slope = 1.0/(basefreq*time)
    print("slope = %.3g"%slope)
    x = getKleinTimeSeries(T, slope, eps=eps)
    x = 0.8*x/np.max(np.abs(x))
    wavfile.write("klein.wav", Fs, x)

if __name__ == '__main__':
    #phi = (1+np.sqrt(5))/2
    #phi = 1.0/phi
    phi = np.sqrt(2)
    makeTorusSound(phi)
    makeKleinSound(eps=0.005)