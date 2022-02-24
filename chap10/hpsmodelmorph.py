import numpy as np
import scipy.signal as sig
from peakinterp import peakinterp
from genspecsines import genspecsines
from f0detectiontwm import f0detectiontwm
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


def hpsmodelmorph(x,x2,fs,w,N,t,nH,minf0,maxf0,f0et,maxhd,stocf,f0intp,htintp,rintp,Ns=1024,H=256):
    '''
    Authors: J. Bonada, X. Serra, X. Amatriain, A. Loscos
    => morph between two sounds using the harmonic plus stochastic model

    ---Inputs---
    x,x2: input sounds
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
    f0intp: f0 interpolation factor
    htintp: harmonic timbre interpolation factor
    rintp: residual interpolation factor
    Ns = 1024: FFT size for synthesis
    H = 256: hop size for analysis and synthesis


    ---Outputs---
    y: output sound
    yh: harmonic component
    ys: stochastic component
    '''

    if type(f0intp)==int:
        f0intp = np.array([[0,(len(x)-1)/fs],[f0intp, f0intp]]) #[[input time],[control value]]
        
    if type(htintp)==int:
        htintp = np.array([[0,(len(x)-1)/fs],[htintp, htintp]])  #[[input time],[control value]]
    
    if type(rintp)==int:
        rintp = np.array([[0,(len(x)-1)/fs],[rintp, rintp]])  #[[input time],[control value]]

    M = len(w);                           # analysis window size (odd)
    N2 = N//2+1;                          # half-size of spectrum
    soundlength = len(x);                 # length of input sound array
    hNs = Ns//2;                          # half synthesis window size
    hM = (M-1)//2;                        # half analysis window size
    pin = max(H,hM);   #initialize sound pointer to middle of analysis window
    pend = soundlength-max(hM,H)-1;       # last sample to start a frame
    fftbuffer = np.zeros(N);              # initialize buffer for FFT
    yh = np.zeros(soundlength+Ns//2);     # output sine component
    ys = np.zeros(soundlength+Ns//2);     # output residual component
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
    minpin2 = max(H,hM);                  # minimum sample value for x2
    maxpin2 = (len(x2)-hM-2);          # maximum sample value for x2

    while pin<pend:
        #-----first sound analysis-----%
        f0,hloc,hmag,mXsenv = hpsanalysis(x,fs,w,wr,pin,M,hM,N,N2,Ns,hNs,nH,t,f0et,minf0,maxf0,maxhd,stocf);
        #-----second sound analysis-----%
        pin2 = round((pin+1)/len(x)*len(x2))-1; # linear time mapping between inputs
        pin2 = min(maxpin2,max(minpin2,pin2));
        f02,hloc2,hmag2,mXsenv2 = hpsanalysis(x2,fs,w,wr,pin2,M,hM,N,N2,Ns,hNs,nH,t,f0et,minf0,maxf0,maxhd,stocf);
        #-----morph-----#
        cf0intp = np.interp(pin/fs,f0intp[0,:],f0intp[1,:]); # get control value 
        chtintp = np.interp(pin/fs,htintp[0,:],htintp[1,:]); # get control value
        crintp = np.interp(pin/fs,rintp[0,:],rintp[1,:]);    # get control value
        if (f0>0 and f02>0):
            outf0 = f0*(1-cf0intp) + f02*cf0intp;            # both inputs are harmonic 
            yhloc = np.arange(nH)*outf0/fs*Ns;               # generate synthesis harmonic serie
            #idx = find(hloc>0 & hloc<Ns*.5);
            idx = np.where((hloc>0) * (hloc<Ns*0.5))[0]
            # interpolated envelops
            #yhmag = interp1([0;hloc(idx);Ns], [hmag(1);hmag(idx);hmag(end)],yhloc);
            yhmag = np.interp(yhloc,np.insert(hloc[idx],[0,len(idx)],[0,Ns]), np.insert(hmag[idx],[0,len(idx)],[hmag[0],hmag[-1]]));

            idx2 = np.where((hloc2>0) * (hloc2<Ns*0.5))[0];
            #yhmag2 = interp1([0; hloc2(idx2); Ns],[hmag2(1);hmag2(idx2);hmag2(end)],yhloc); % interpolated envelope
            yhmag2 = np.interp(yhloc,np.insert(hloc2[idx2],[0,len(idx2)],[0,Ns]), np.insert(hmag2[idx2],[0,len(idx2)],[hmag2[0],hmag2[-1]]));
            yhmag = yhmag*(1-chtintp) + yhmag2*chtintp;      # timbre morphing
        else:
            outf0 = 0;                                       # remove harmonic content
            yhloc = hloc*0;
            yhmag = hmag*0;
        
        mYsenv = mXsenv*(1-crintp) + mXsenv2*crintp; 
        #-----synthesis-----%
        yhphase = yhphase + 2*np.pi*(lastyhloc+yhloc)/2/Ns*H; # propagate phases
        lastyhloc = yhloc.copy();
        Yh = genspecsines(yhloc,yhmag,yhphase,Ns);    # generate sines
        mYs = interpolate_1d_vector(mYsenv,stocf);    # interpolate to original size
        roffset = int(np.ceil(stocf/2))-1;            # interpolated array offset
        mYs = np.hstack((mYs[0]*np.ones(roffset), mYs[:Ns//2+1-roffset]));
        mYs = 10**(mYs/20);                           # dB to linear magnitude
        pYs = 2*np.pi*np.random.uniform(0,1,(Ns//2)+1)# generate phase spectrum with random values
        mYs1 = np.hstack((mYs[:Ns//2+1], mYs[Ns//2-1:0:-1]));   # create complete magnitude spectrum
        pYs1 = np.hstack((pYs[:Ns//2+1],-1*pYs[Ns//2-1:0:-1])); # create complete phase spectrum
        Ys = mYs1*np.cos(pYs1)+1j*mYs1*np.sin(pYs1);  # compute complex spectrum
        yhw = np.fft.fftshift(np.real(np.fft.ifft(Yh)));# sines in time domain using IFFT
        ysw = np.fft.fftshift(np.real(np.fft.ifft(Ys)));# stochastic in time domain using IFFT
        ro = pin-hNs;                                   # output sound pointer for overlap
        yh[ro:ro+Ns] += yhw * sw;                       # overlap-add for sines
        ys[ro:ro+Ns] += ysw * sws;                      # overlap-add for stochastic
        pin = pin+H;                                    # advance the sound pointer
        
    y= yh+ys;                                           # sum sines and stochastic
    return y,yh,ys

def hpsanalysis(x,fs,w,wr,pin,M,hM,N,N2,Ns,hNs,nH,t,f0et,minf0,maxf0,maxhd,stocf):
    xw = x[pin-hM:pin+hM+1]*w;                  # window the input sound
    fftbuffer = np.zeros(N);                    # initialize buffer for FFT
    fftbuffer[:(M+1)//2] = xw[(M+1)//2-1:];     # zero-phase window in fftbuffer
    fftbuffer[N-(M-1)//2:] = xw[:(M-1)//2];
    X = np.fft.fft(fftbuffer);                  # compute the FFT
    mX = 20*np.log10(abs(X[:N2]));              # magnitude spectrum 
    pX = np.unwrap(np.angle(X[:N//2+1]));        # unwrapped phase spectrum 
    auxploc = np.where(mX[1:-1]>t,1,0) * np.where(mX[1:-1]>mX[2:],1,0) * np.where(mX[1:-1]>mX[:-2],1,0)
    ploc = 1 + np.where(auxploc>0)[0]      # find peaks
    #ploc = 1 + find((mX(2:N2-1)>t) .* (mX(2:N2-1)>mX(3:N2)) ...
    #                .* (mX(2:N2-1)>mX(1:N2-2)));    % find peaks
    ploc,pmag,pphase = peakinterp(mX,pX,ploc);    # refine peak values
    f0 = f0detectiontwm(mX,fs,ploc,pmag,f0et,minf0,maxf0);    # find f0
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
    xr = x[ri:ri+Ns]*wr;                       # window the input sound
    Xr = np.fft.fft(np.fft.fftshift(xr));      # compute FFT for residual analysis
    Xh = genspecsines(hloc[:hi],hmag,hphase,Ns)# generate sines
    Xr = Xr-Xh;                                # get the residual complex spectrum
    mXr = 20*np.log10(abs(Xr[:Ns//2+1]));      # magnitude spectrum of residual
    mXsenv = sig.decimate(np.where(mXr<-200,-200,mXr),stocf);# decimate the magnitude spectrum

    return f0,hloc,hmag,mXsenv

if __name__=='__main__':
    import audiofile as af
    
    x,fs  = af.read('audios2\\soprano-E4.wav');#('soprano-E4.wav');
    x2,fs = af.read('audios2\\violin-B3.wav');
    w=sig.windows.blackmanharris(1024)
    w = np.insert(w,len(w),0)
    f0intp = 0;
    htintp = 1;
    rintp = 0;
    y,yh,ys = hpsmodelmorph(x,x2,fs,w,2048,-150,200,100,400,1500,1.5,10,f0intp,htintp,rintp);
    af.write('audios2\\soprano_violin.wav',y,fs)

    x,fs  = af.read('audios2\\vocalize.wav');
    y,yh,ys = hpsmodelmorph(x[0,:],x2,fs,w,2048,-150,200,100,400,1500,1.5,10,f0intp,htintp,rintp);
    af.write('audios2\\vocalize_violin.wav',y,fs)

    x2,fs = af.read('audios2\\flute2.wav');
    f0intp = 0;
    htintp = 1;
    rintp = 1;
    y,yh,ys = hpsmodelmorph(x[0,:],x2,fs,w,2048,-150,200,100,440,1500,1.5,10,f0intp,htintp,rintp);
    af.write('audios2\\vocalize_flute.wav',y,fs)

    
    x,fs  = af.read('audios2\\soprano-E4.wav');#('soprano-E4.wav');
    dur = (len(x)-1)/fs;
    f0intp = np.array([[0,dur],[0, 1]])
    htintp = np.array([[0,dur],[0, 1]])
    rintp = np.array([[0,dur],[0, 1]])
    y,yh,ys = hpsmodelmorph(x,x2,fs,w,2048,-150,200,100,400,1500,1.5,10,f0intp,htintp,rintp);
    af.write('audios2\\soprano_violin2.wav',y,fs)

    x,fs  = af.read('audios2\\vocalize.wav');
    dur = (x.shape[1]-1)/fs;
    f0intp = 0
    htintp = np.array([[0,dur/3,2*dur/3,dur],[0, 1, 1, 0]])
    rintp = np.array([[0,dur/3,2*dur/3,dur],[0, 1, 1, 0]])
    y,yh,ys = hpsmodelmorph(x[0,:],x2,fs,w,2048,-150,200,100,400,1500,1.5,10,f0intp,htintp,rintp);
    af.write('audios2\\vocalize_violin2.wav',y,fs)
