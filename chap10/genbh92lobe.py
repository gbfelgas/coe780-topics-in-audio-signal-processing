import numpy as np

def D(x,N):
    '''
    Calculate rectangular window transform (Dirichlet kernel)
    '''
    y = np.sin(N*x/2)/np.sin(x/2);
    y = np.where(y!=y,N,y)                                #avoid NaN if x==0
    return y

def genbh92lobe(x,N = 512):
    '''
    Authors: J. Bonada, X. Serra, X. Amatriain, A. Loscos
    Calculate transform of the Blackman-Harris 92dB window
    x: bin positions to compute (real values)
    y: transform values
    '''
    f = x*np.pi*2/N;                                  # frequency sampling
    df = 2*np.pi/N;
    y = np.zeros(len(x));                                # initialize window
    consts = [.35875, .48829, .14128, .01168];        # window constants
    for m in range(4):
        y = y + consts[m]/2 *(D(f-df*m,N)+D(f+df*m,N)); # sum Dirichlet kernels
    y = y/N/consts[0];                                    # normalize

    return y
