import numpy as np
import scipy
import scipy.signal as sig
from numpy.fft import fft, fftshift,fftfreq,ifft
from peakinterp import peakinterp
from genspecsines import genspecsines
from f0detectiontwm import f0detectiontwm


from scipy.interpolate import interp1d

def interp(ys, mul):
    # linear extrapolation for last (mul - 1) points
    ys = list(ys)
    ys.append(2*ys[-1] - ys[-2])
    # make interpolation function
    xs = np.arange(len(ys))
    fn = interp1d(xs, ys, kind="cubic")
    # call it on desired data points
    new_xs = np.arange(len(ys) - 1, step=1./mul)
    return fn(new_xs)

def hprmodel(x,fs,w,N,t,nH,minf0,maxf0,f0et,maxhd):


    M = len(w);   # analysis window size
    Ns = 1024;                               # FFT size for synthesis
    H = 256;                                 # hop size for analysis and synthesis
    N2 = N//2+1;                              # half-size of spectrum
    soundlength = len(x);                 # length of input sound array
    hNs = Ns//2;                              # half synthesis window size
    hM = (M-1)//2;                            # half analysis window size
    pin = max(H,hM);   # initialize sound pointer to middle of analysis window
    pend = soundlength-max(hM,H);            # last sample to start a frame
    fftbuffer = np.zeros(N);                  # initialize buffer for FFT
    yh = np.zeros(soundlength+Ns//2);          # output sine component
    yr = np.zeros(soundlength+Ns//2);          # output residual component
    w = w/np.sum(w);                            # normalize analysis window
    sw = np.zeros(Ns);
    ow = sig.triang(2*H-1);                      # overlapping window
               # overlap indexes
    sw[Ns//2+1-H:Ns//2+H] = ow[:2*H-1]
    bh = sig.blackmanharris(Ns);                 # synthesis window
    bh = bh / np.sum(bh);                      # normalize synthesis window
    wr = bh;                                 # window for residual 
    sw[Ns//2+1-H:Ns//2+H] = sw[Ns//2+1-H:Ns//2+H] / bh[Ns//2+1-H:Ns//2+H]
    sws = H*sig.hanning(Ns)/2;                   # synthesis window for stochastic
    f0s = []
    while pin<pend:
    #-----analysis-----#
        xw = x[pin-hM:pin+hM+1]*w;         # window the input sound
        fftbuffer[:] = 0;                      # reset buffer
        fftbuffer[:(M+1)//2] = xw[(M+1)//2-1:M];  # zero-phase window in fftbuffer
        fftbuffer[N-(M-1)//2:N] = xw[:(M-1)//2];
        X = fft(fftbuffer);                    # compute the FFT
        mX = 20*np.log10(abs(X[:N2]));           # magnitude spectrum 
        pX = np.unwrap(np.angle(X[:N//2+1]));        # unwrapped phase spectrum 
        ploc = 1 + np.argwhere( (mX[1:N2-1]>t) * (mX[1:N2-1]>mX[2:N2]) *(mX[1:N2-1]>mX[:N2-2]));          # find peaks
        
        [ploc,pmag,pphase] = peakinterp(mX,pX,ploc);          # refine peak values
        f0 = f0detectiontwm(mX,fs,ploc,pmag,f0et,minf0,maxf0);   # find f0
        f0s.append(f0)
        hloc = np.zeros(nH);                    # initialize harmonic locations
        hmag = np.zeros(nH)-100;                # initialize harmonic magnitudes
        hphase = np.zeros(nH);                  # initialize harmonic phases
        hf = (f0>0)*(f0*(np.arange(1,nH+1)));             # initialize harmonic frequencies
        hi = 1;                                # initialize harmonic index
        npeaks = len(ploc);                 # number of peaks found
        while (f0>0 and hi<nH and hf[hi]<fs/2):  # find harmonic peaks
            [dev,pei] = np.min(np.abs((ploc[:npeaks]-1)/N*fs-hf[hi])),np.argmin(np.abs((ploc[:npeaks]-1)/N*fs-hf[hi]));    # closest peak
            if (hi==1 or not ((hloc[1:hi-1]==ploc[pei]).any()) or dev<maxhd*hf[hi]):
                hloc[hi] = ploc[pei];              # harmonic locations
                hmag[hi] = pmag[pei];              # harmonic magnitudes
                hphase[hi] = pphase[pei];          # harmonic phases
            hi = hi+1;                           # increase harmonic index
        hloc[:hi-1] = (hloc[:hi-1]!=0)*((hloc[:hi-1]-1)*Ns/N);  # synth. locs
        ri= pin-hNs;                     # input sound pointer for residual analysis
        xr = x[ri:ri+Ns]*wr[:Ns];          # window the input sound
        Xr = fft(fftshift(xr));                # compute FFT for residual analysis
        Yh = genspecsines(hloc[:hi-1],hmag,hphase,Ns);             # generate sines
        Yr = Xr-Yh;                            # get the residual complex spectrum

        #-----synthesis-----#
        yhw = fftshift(np.real(ifft(Yh)));            # sines in time domain using IFFT
        yrw = fftshift(np.real(ifft(Yr)));            # residual in time domain using inverse FFT
        yh[ri:ri+Ns] = yh[ri:ri+Ns]+yhw[:Ns]*sw;   # overlap-add for sines
        yr[ri:ri+Ns] = yr[ri:ri+Ns]+yrw[:Ns]*sws;  # overlap-add for stoch.
        pin = pin+H;                                     # advance the sound pointer
    y= yh+yr; # sum sines and stochastic
    return y, yh,yr
