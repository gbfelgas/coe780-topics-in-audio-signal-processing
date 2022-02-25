import numpy as np
import scipy.signal as sig
from peakinterp import peakinterp
from genspecsines import genspecsines
from f0detectionyin import f0detectionyin
import scipy.interpolate

def interpolate_1d_vector(vector, factor):
    """
    Interpolate, i.e. upsample, a given 1D vector by a specific interpolation factor.
    :param vector: 1D data vector
    :param factor: factor for interpolation (must be integer)
    :return: interpolated 1D vector by a given factor
    """
    x = np.arange(np.size(vector))
    y = vector
    f = scipy.interpolate.interp1d(x, y)

    x_extended_by_factor = np.linspace(x[0], x[-1], np.size(x) * factor)
    y_interpolated = np.zeros(np.size(x_extended_by_factor))

    i = 0
    for x in x_extended_by_factor:
        y_interpolated[i] = f(x)
        i += 1

    return y_interpolated

def hpsmodeltranspandtimbrescaling(x,fs,w,N,t,nH,minf0,maxf0,f0et,maxhd,stocf,fscale,timbremapping,timemapping=None,Ns=1024,H=256):
    '''
    Authors: J. Bonada, X. Serra, X. Amatriain, A. Loscos
    => analysis/synthesis of a sound using the sinusoidal harmonic model
    ---Inputs---
    x: input sounds
    fs: sampling rate
    w: analysis window (odd size) 
    N: FFT size (minimum 512)
    t: threshold in negative dB 
    nH: maximum number of harmonics
    minf0: minimum f0 frequency in Hz
    maxf0: maximim f0 frequency in Hz
    f0et: error threshold in the f0 detection (ex: 5)
    maxhd: max. relative deviation in harmonic detection (ex: .2)
    stocf: decimation factor of mag spectrum for stochastic analysis
    timemapping: mapping between input and output time (sec): array([[input time],[output time]])
    Ns = 1024: FFT size for synthesis
    H = 256: hop size for analysis and synthesis


    ---Outputs---
    y: output sound
    yh: harmonic component
    ys: stochastic component
    '''
    
    if timemapping is None:                 # argument not specified
        timemapping = np.array([[0,(len(x)-1)/fs],[0,(len(x)-1)/fs]]) #[[input time],[output time]]

    M = len(w);                           # analysis window size (odd)
    N2 = N//2+1;                          # half-size of spectrum
    hNs = Ns//2;                          # half synthesis window size
    hM = (M-1)//2;                        # half analysis window size
    fftbuffer = np.zeros(N);              # initialize buffer for FFT
    outsoundlength = int(round(timemapping[1,-1]*fs)) # length of output sound
    yh = np.zeros(outsoundlength+Ns//2);  # output sine component
    ys = np.zeros(outsoundlength+Ns//2);  # output residual component
    w = w/w.sum();                        # normalize analysis window
    sw = np.zeros(Ns);
    ow = sig.windows.triang(2*H-1);       # overlapping window
    ovidx = np.arange(Ns//2+1-H,Ns//2+H,1,dtype='int64'); # overlap indexes
    sw[ovidx] = ow[:2*H-1];
    bh = sig.windows.blackmanharris(Ns);  # synthesis window
    bh = bh / bh.sum();                   # normalize synthesis window
    wr = bh.copy();                       # window for residual 
    sw[ovidx] = sw[ovidx] / bh[ovidx];
    sws = H*sig.windows.hann(Ns+2)/2;     # synthesis window for stochastic
    sws = sws[1:-1]                       # used hanning(Ns) in Matlab, which cuts zeros 
    lastyhloc = np.zeros(nH);             # initialize synthesis harmonic locs.
    yhphase = 2*np.pi*np.random.uniform(0,1,nH); # initialize synthesis harmonic phases
    poutend = outsoundlength-max(hM,H);   # last sample to start a frame
    pout = max(H,hM);  # initialize sound pointer to middle of analysis window
    minpin = max(H,hM);
    maxpin = len(x)-hM-2;

    while pout<poutend:
        pin = round(np.interp((pout)/fs,timemapping[1,:],timemapping[0,:]) * fs );
        pin = max(minpin,pin);
        pin = min(maxpin,pin);
        #-----analysis-----%
        xw = x[int(pin-hM):int(pin+hM+1)]*w;                  # window the input sound
        fftbuffer = np.zeros(N);                    # initialize buffer for FFT
        fftbuffer[:(M+1)//2] = xw[(M+1)//2-1:];     # zero-phase window in fftbuffer
        fftbuffer[N-(M-1)//2:] = xw[:(M-1)//2];
        X = np.fft.fft(fftbuffer);                  # compute the FFT
        mX = 20*np.log10(abs(X[:N2]));              # magnitude spectrum 
        pX = np.unwrap(np.angle(X[:N//2+1]));       # unwrapped phase spectrum 
        auxploc = np.where(mX[1:-1]>t,1,0) * np.where(mX[1:-1]>mX[2:],1,0) * np.where(mX[1:-1]>mX[:-2],1,0)
        ploc = 1 + np.where(auxploc>0)[0]      # find peaks
        #ploc = 1 + find((mX(2:N2-1)>t) .* (mX(2:N2-1)>mX(3:N2)) ...
        #                .* (mX(2:N2-1)>mX(1:N2-2)));    % find peaks
        ploc,pmag,pphase = peakinterp(mX,pX,ploc);    # refine peak values

        yinws = round(fs*0.0125);          # using approx. a 12.5 ms window for yin
        yinws = yinws+(yinws%2);           # make it even
        yb = pin-yinws//2;
        ye = pin+yinws//2+yinws+1;
        if (yb<0 or ye>len(x)): # out of boundaries
            f0 = 0;
        else:
            f0 = f0detectionyin(x[int(yb):int(ye)],fs,yinws,minf0,maxf0);

        hloc = np.zeros(nH);                       # initialize harmonic locations
        hmag = np.zeros(nH)-100;                   # initialize harmonic magnitudes
        hphase = np.zeros(nH);                     # initialize harmonic phases
        hf = np.where(f0>0,1,0)*(f0*np.arange(1,nH+1,1));# initialize harmonic frequencies
        hi = 0;                                    # initialize harmonic index
        npeaks = len(ploc);                        # number of peaks found
        while (f0>0 and hi<nH and hf[hi]<fs/2):    # find harmonic peaks
            #[dev,pei] = min(abs((ploc(1:npeaks)-1)/N*fs-hf(hi)));  % closest peak
            auxdev = abs((ploc-1)/N*fs-hf[hi])
            dev = auxdev.min()
            pei = np.where(auxdev==dev)[0][0]      # closest peak
            if ((hi==0 or not(np.any(hloc[:hi]==ploc[pei]))) and dev<maxhd*hf[hi]):
                hloc[hi] = ploc[pei];              # harmonic locations
                hmag[hi] = pmag[pei];              # harmonic magnitudes
                hphase[hi] = pphase[pei];          # harmonic phases

            hi = hi+1;                             #increase harmonic index
        hloc[:hi] = np.where(hloc[:hi]!=0,1,0)*(hloc[:hi]*Ns/N); # synth. locs
        ri = pin-hNs;                              # input sound pointer for residual analysis
        xr = x[int(ri):int(ri+Ns)]*wr;                       # window the input sound
        Xr = np.fft.fft(np.fft.fftshift(xr));      # compute FFT for residual analysis
        Xh = genspecsines(hloc[:hi],hmag,hphase,Ns)# generate sines
        Xr = Xr-Xh;                                # get the residual complex spectrum
        mXr = 20*np.log10(abs(Xr[:Ns//2+1]));      # magnitude spectrum of residual
        mXsenv = sig.decimate(np.where(mXr<-200,-200,mXr),stocf);# decimate the magnitude spectrum and avoid -Inf

        #-----synthesis data-----%
        yhloc = hloc.copy();                    # synthesis harmonics locs
        yhmag = hmag.copy();                    # synthesis harmonic amplitudes
        mYsenv = mXsenv.copy();                 # synthesis residual envelope
        yf0 = f0;                               # synthesis f0

        #-----transformations-----#

        # 10.31 code
        # Authors: J. Bonada, X. Serra, X. Amatriain, A. Loscos
        #-----pitch transposition and timbre scaling-----
        yhloc = yhloc*fscale        # scale harmonic frequencies
        yf0 = f0*fscale             # synthesis fundamental frequency
        # harmonics
        if (f0>0):
            thloc = np.interp(yhloc/Ns*fs,timbremapping[1,:],timbremapping[0,:])/fs*Ns; # mapped harmonic freqs.
            idx = np.where((hloc>0) & (hloc<Ns*5))         #harmonic indexes in frequency range
            aux_hloc = np.insert(hloc[idx], 0, 0)
            aux_hloc = np.insert(aux_hloc, len(hloc[idx])+1, Ns)
            aux_hmag = np.insert(hmag[idx], 0, hmag[0])
            aux_hmag = np.insert(aux_hmag, len(hmag[idx])+1, hmag[-1])
            yhmag = np.interp(thloc,aux_hloc, aux_hmag)   #interpolated envelope

        # residual
        # frequency (Hz) of the last coefficient
        frescoef = fs/2*len(mYsenv)*stocf/len(mXr)
        # mapped coef. indexes
        mYsenv_aux0 = np.arange(0,len(mYsenv)-1,1);
        mYsenv_aux1 = np.transpose(mYsenv_aux0)
        aux_min = np.minimum(fs/2,mYsenv_aux1/(len(mYsenv)-1)*frescoef)
        trescoef = np.interp(aux_min,timbremapping[1,:],timbremapping[0,:])
        mYsenv_aux = np.arange(0,len(mYsenv),1);
        # interpolated envelope
        mYsenv = np.interp(trescoef/frescoef*(len(mYsenv)-1),mYsenv_aux, mYsenv)

        #-----synthesis-----%

        yhphase = yhphase + 2*np.pi*(lastyhloc+yhloc)/2/Ns*H; # propagate phases
        lastyhloc = yhloc.copy();
        Yh = genspecsines(yhloc,yhmag,yhphase,Ns);    # generate sines
        mYs = interpolate_1d_vector(mYsenv,stocf);    # interpolate to original size
        roffset = int(np.ceil(stocf/2))-1;            # interpolated array offset
        mYs = np.hstack((mYs[0]*np.ones(roffset), mYs[:Ns//2+1-roffset]));
        mYs = 10**(mYs/20);                           # dB to linear magnitude
        if (f0>0):
            mYs = mYs * (np.cos(np.pi*np.arange(0,Ns//2+1,1)/Ns*fs/yf0)**2)  # filter residual
        fc = 1 + round(500/fs*Ns);                               # 500 Hz
        mYs[:fc] = mYs[:fc] * (np.arange(fc)/(fc-1)**2);    # HPF
        pYs = 2*np.pi*np.random.uniform(0,1,(Ns//2)+1)# generate phase spectrum with random values
        mYs1 = np.hstack((mYs[:Ns//2+1], mYs[Ns//2-1:0:-1]));   # create complete magnitude spectrum
        pYs1 = np.hstack((pYs[:Ns//2+1],-1*pYs[Ns//2-1:0:-1])); # create complete phase spectrum
        Ys = mYs1*np.cos(pYs1)+1j*mYs1*np.sin(pYs1);  # compute complex spectrum
        yhw = np.fft.fftshift(np.real(np.fft.ifft(Yh)));# sines in time domain using IFFT
        ysw = np.fft.fftshift(np.real(np.fft.ifft(Ys)));# stochastic in time domain using IFFT
        ro = pout-hNs;                                  # output sound pointer for overlap
        yh[ro:ro+Ns] += yhw * sw;                       # overlap-add for sines
        ys[ro:ro+Ns] += ysw * sws;                      # overlap-add for stochastic
        pout = pout+H;                                  # advance the sound pointer
    
    y= yh+ys;                                         # sum sines and stochastic
    return y,yh,ys

if __name__=='__main__':

    import audiofile as af
    x, fs = af.read('basket.wav');
    w = sig.windows.hann(1025,False);
    N = 1024;
    t = -100;
    nH = 50;
    minf0 = 100; #minimum f0 frequency in Hz, 
    maxf0 = 400; #maximim f0 frequency in Hz, 
    f0et = 1500;
    maxhd = 1.5;
    stocf = 10;
    #tmap = 0;
    # 10.32 and 10.33 codes
    fscale = 2
    timbremapping_inp = np.array([0, 4000, fs/2]);     # input frequency                    # input time (sec)
    timebremapping_out = np.array([0, 5000, fs/2]);    # output frequency
    timbremapping = np.vstack((timbremapping_inp,timebremapping_out))
    y, yh, ys= hpsmodeltranspandtimbrescaling(x,fs,w,N,t,nH,minf0,maxf0,f0et,maxhd,stocf,fscale, timbremapping);
    af.write('basket_female.wav',y,fs)
    
