import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
from peakinterp import *
from f0detectionyin import *
from genspecsines import *

def hpsmodelparams(x, fs, w, N, t, nH, minf0, maxf0, f0et, maxhd, stocf, timemapping = None, timbremapping = None):
    # Authors: J. Bonada, X. Serra, X. Amatriain, A. Loscos
    #=> analysis/synthesis of a sound using the sinusoidal harmonic model
    # x: input sound, fs: sampling rate, w: analysis window (odd size), 
    # N: FFT size (minimum 512), t: threshold in negative dB, 
    # nH: maximum number of harmonics, minf0: minimum f0 frequency in Hz, 
    # maxf0: maximim f0 frequency in Hz, 
    # f0et: error threshold in the f0 detection (ex: 5),
    # maxhd: max. relative deviation in harmonic detection (ex: .2)
    # stocf: decimation factor of mag spectrum for stochastic analysis
    # timemapping: mapping between input and output time (sec)
    # y: output sound, yh: harmonic component, ys: stochastic component
    #
    #--------------------------------------------------------------------------
    # This source code is provided without any warranties as published in 
    # DAFX book 2nd edition, copyright Wiley & Sons 2011, available at 
    # http://www.dafx.de. It may be used for educational purposes and not 
    # for commercial applications without further permission.
    #--------------------------------------------------------------------------

    if timemapping == None:                   # argument not specified
        timemapping = np.array([[0, len(x)/fs],[0, len(x)/fs]])
    M = len(w)                                # analysis window size
    Ns = 1024                                 # FFT size for synthesis
    H = 256                                   # hop size for analysis ans synthesis
    N2 = N//2+1                               # half-size of spectrum
    hNs = Ns//2                               # half synthesis window size
    hM = (M-1)//2                             # half analysis window size
    fftbuffer = np.zeros(N)                   # initialize buffer for FFT
    outsoundlength = 1+round(timemapping[1,-1]*fs)  # length of output sound
    yh = np.zeros(outsoundlength+Ns//2)       # output sine component
    ys = np.zeros(outsoundlength+Ns//2)       # output residual component
    w = w/w.sum()                             # normalize analysis window
    sw = np.zeros(Ns)  
    ow = sig.windows.triang(2*H-1)            # overlapping window
    ovidx = np.arange(Ns//2+1-H,Ns//2+H,1,dtype='int64') # overlap indexes
    sw[ovidx] = ow[:2*H-1]  
    bh = sig.windows.blackmanharris(Ns)       # synthesis window
    bh = bh / bh.sum()                        # normalize synthesis window
    wr = bh.copy()                            # window for residual 
    sw[ovidx] = sw[ovidx] / bh[ovidx]
    sws = H*sig.windows.hann(Ns+2)/2          # synthesis window for stochastic
    sws = sws[1:-1]                           # used hanning(Ns) in Matlab, which cuts zeros 
    lastyhloc = np.zeros(nH)                  # initialize synthesis harmonic locations
    yhphase = 2*np.pi*np.random.rand(nH)         # initialize systhesis harmonic phases
    poutend = outsoundlength - max(hM,H)      # last sample to start a frame
    pout = max(H+1,1+hM)                      # initialize sound pointer to middle of analysis window
    minpin = max(H+1,1+hM)
    maxpin = len(x)-max(hM,hNs)-1

    while pout < poutend:
        pinaux = np.interp(pout/fs, timemapping[1,:], timemapping[0,:])
        pin = np.round(pinaux*fs)
        pin = max(minpin,pin)
        pin = min(maxpin,pin)
        pin = int(pin)
        #-----analysis-----#
        xw = x[pin-hM:pin+hM+1]*w                               # window the input sound
        fftbuffer = fftbuffer*0                                 # reset buffer  
        fftbuffer[:(M+1)//2] = xw[(M+1)//2-1:M]                 # zero-phase window in fftbuffer
        fftbuffer[N-(M-1)//2:] = xw[:(M-1)//2]  
        X = np.fft.fft(fftbuffer)                               # compute the FFT
        mX = 20*np.log10(abs(X[:N2]))                           # magnitude spectrum 
        pX = np.unwrap(np.angle(X[:N2]))                        # unwrapped phase spectrum 
        auxploc = np.where(mX[1:-1]>t,1,0) * np.where(mX[1:-1]>mX[2:],1,0)*np.where(mX[1:-1]>mX[:-2],1,0)
        ploc = 1 + np.where(auxploc>0)[0]                       # find peaks
        ploc,pmag,pphase = peakinterp(mX,pX,ploc)               # refine peak values
        yinws = round(fs*0.0125)                                # using approx. a 12.5 ms window for yin
        yinws = yinws + yinws % 2                               # make it even
        yb = pin - yinws//2
        ye = pin + yinws//2 + yinws
        if yb < 1 or ye > len(x):                               # out of boundaries
            f0 = 0
        else:
            f0 = f0detectionyin(x[yb:ye],fs,yinws,minf0,maxf0)  # compute f0
        hloc = np.zeros(nH)                                     # initialize harmonic locations
        hmag = np.zeros(nH)-100                                 # initialize harmonic magnitudes
        hphase = np.zeros(nH)                                   # initialize harmonic phases
        hf = f0*np.arange(1,nH+1,1)                             # initialize harmonic frequencies
        hi = 0                                                  # initialize harmonic index
        npeaks = len(ploc)                                      # number of peaks found
        while f0>0 and hi < nH and hf[hi]<fs/2 and npeaks>0:    # find harmonic peaks
            dev = np.amin(np.abs(ploc[:npeaks]-1)/N*fs-hf[hi])  # closest peak
            pei = np.argmin(np.abs(ploc[:npeaks]-1)/N*fs-hf[hi])
            if hi == 1 and not np.any(hloc[:hi-1]==ploc[pei]) and dev < maxhd*hf[hi]:
                hloc[hi] = ploc[pei]                            # harmonic locations
                hmag[hi] = pmag[pei]                            # harmonic magnitudes
                hphase[hi] = pphase[pei]                        # harmonic phases
            hi = hi + 1                                         # increase harmonic index
        hloc[:hi-1] = (hloc[:hi-1] != 0)*((hloc[:hi-1]-1)*Ns/N) # synth. locs
        ri = pin - hNs                                          # input pointer for residual analysis
        xr = x[ri:ri+Ns]*wr[:Ns]                                # window the input sound
        Xr = np.fft.fft(np.fft.fftshift(xr))                    # compute fft for residual analysis
        Xh = genspecsines(hloc[:hi-1], hmag, hphase, Ns)        # generate sines
        Xr = Xr-Xh                                              # get the residual complex spectrum
        mXr = 20*np.log10(np.abs(Xr[:Ns//2 + 1]))               # magnitude spectrum of residual
        mXsenv = sig.decimate(np.maximum(np.zeros(mXr.shape)-200,mXr),stocf)    # decimate the magnitude spectrum
                                                                # and avoid -Inf
        #-----synthesis data-----#
        yhloc = hloc                                            # synthesis harmonic locs
        yhmag = hmag                                            # synthesis harmonic amplitudes
        mYsenv = mXsenv                                         # synthesis residual envelope
        yf0 = f0                                                # synthesis f0
        #-----transformations-----#
        #-----pitch transposition and timbre scaling-----#
        fscale = 2 
        yhloc = yhloc*fscale                # scale harmonic frequencies
        yf0 = f0*fscale                     # synthesis fundamental frequency
        if np.all(timbremapping) == None:
            timbremapping = np.array([[0, 4000, fs/2], [0, 4000, fs/2]])
        # harmonics
        if f0>0:
            thloc = np.interp(yhloc/Ns*fs, timbremapping[1,:], timbremapping[0,:]) / fs*Ns  # mapped harmonic freqs.
            auxidx = np.where(hloc>0,1,0) * np.where(hloc<Ns*.5,1,0)
            idx = np.where(auxidx>0)[0]        # harmonic indexes in frequency range
            aux1 = np.append(np.append(0,hloc[idx]),Ns)
            aux2 = np.append(np.append(hmag[0],hmag[idx]),hmag[-1])
            yhmag = np.interp(thloc, aux1, aux2)                # interpolated envelope
        # residual
        # frequency (Hz) of the last  coefficient
        frescoef = fs/2*len(mYsenv)*stocf/len(mXr); 
        # mapped coef. indexes
        aux = np.minimum(np.zeros(len(mYsenv))+fs//2,np.arange(0,len(mYsenv),1)/(len(mYsenv))*frescoef)
        trescoef = np.interp(aux, timbremapping[1,:], timbremapping[0,:])
        # interpolated envelope                                   
        mYsenv = np.interp(trescoef/frescoef*(len(mYsenv)-1),np.arange(0,len(mYsenv),1),mYsenv)
        #-----synthesis-----#
        yhphase = yhphase + 2*np.pi*(lastyhloc+yhloc)/2/Ns*H    # propagate phases
        lastyhloc = yhloc
        Yh = genspecsines(yhloc, yhmag, yhphase, Ns)            # generate sines
        mYs = sig.resample_poly(mYsenv,stocf,1)                 # interpolate to original size
        roffset = int(np.ceil(stocf/2) - 1)                     # interpolated array offset
        mYs = np.append(mYs[0]*np.ones(roffset), mYs[:Ns//2+1-roffset])
        mYs = 10**(mYs/20)                                      # dB to linear magnitude
        if f0 > 0:
            mYs = mYs*np.cos(np.pi*np.arange(0,Ns//2+1,1)/Ns*fs/yf0)**2  # filter residual
        fc = 1 + round(500/fs*Ns)                               # 500Hz
        mYs[:fc] = mYs[:fc]*np.arange(0,fc,1)/(fc-1)**2         # HPF
        pYs = 2*np.pi*np.random.rand(Ns//2+1,1)                 # generate phase spectrum with random values
        mYs1 = np.append(mYs[:Ns//2+1], mYs[Ns//2:1:-1])        # create complete magnitude spectrum
        pYs1 = np.append(pYs[:Ns//2+1], -1*pYs[Ns//2:1:-1])     # create complex phase spectrum
        Ys = mYs1*np.cos(pYs1)+1j*mYs1*np.sin(pYs1)             # compute complex spectrum
        yhw = np.fft.fftshift(np.real(np.fft.ifft(Yh)))         # sines in time domain using IFFT
        ysw = np.fft.fftshift(np.real(np.fft.ifft(Ys)))         # stochastic in time domain using IFFT
        ro = pout - hNs                                         # output sound pointer for overlap
        yh[ro:ro+Ns] += yhw[:Ns]*sw                             # overlap-add for sines
        ys[ro:ro+Ns] += ysw[:Ns]*sws                            # overlap-add for stochastic
        pout += H                                               # advance the sound pointer
    y = yh + ys                               # sum sines and stochastic

    return y, yh, ys