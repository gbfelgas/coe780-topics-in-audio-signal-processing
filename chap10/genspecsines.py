import numpy as np
from genbh92lobe import genbh92lobe

def genspecsines(ploc, pmag, pphase, N):
    '''
    Authors: J. Bonada, X. Serra, X. Amatriain, A. Loscos
    Compute a spectrum from a series of sine values
    ---- Inputs ----
    iploc, ipmag, ipphase: sine locations, magnitudes and phases
    N: size of complex spectrum (even)
    ---- Output ----
    Y: generated complex spectrum of sines
    '''
    Y = np.zeros(N,dtype='complex128'); # initialize output spectrum
    hN = N//2;                        # size of positive freq. spectrum
    for i in range(len(ploc)):          # generate all sine spectral lobes
        loc = ploc[i];                  # location of peak (zero-based indexing)
                                        # it should be in range ]0,hN[
        if (loc<=0 or loc>=hN): continue; # avoid frequencies out of range
        binremainder = round(loc)-loc;
        lb = np.arange(binremainder-4,binremainder+5,1) # main lobe (real value) bins to read
        lmag = genbh92lobe(lb,N//2)*10**(pmag[i]/20); # lobe magnitudes of the 
                                                       # complex exponential
        b = np.arange(round(loc)-4,round(loc)+5,1,dtype='int64') # spectrum bins to fill
                                                         # (0-based indexing)
        for m in range(9):
            if (b[m]<0):                       # peak lobe croses DC bin
                Y[-b[m]] = Y[-b[m]] + lmag[m]*np.exp(-1j*pphase[i]);
            elif (b[m]>hN):                  # peak lobe croses Nyquist bin
                Y[2*hN-b[m]] = Y[2*hN-b[m]] + lmag[m]*np.exp(-1j*pphase[i]);
            else:                              # peak lobe in positive freq. range
                Y[b[m]] = Y[b[m]] + lmag[m]*np.exp(1j*pphase[i]) + lmag[m]*np.exp(-1j*pphase[i])*int(b[m]==0 or b[m]==hN)
    Y[hN:] = np.conjugate(Y[hN:0:-1]);   # fill the rest of the spectrum
    return Y
