import numpy as np
import scipy.signal as sig
from peakinterp import peakinterp
from genspecsines import genspecsines
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


def spsmodel(x,fs,w,N,t,maxnS,stocf,Ns=1024,H=256,effect=None,Filter=None,fshift=None,fstretch=None,fscale=None):
    '''
    Authors: J. Bonada, X. Serra, X. Amatriain, A. Loscos
    => analysis/synthesis of a sound using the sinusoidal harmonic model

    ---- Input ----
    x: input sound (one dimension)
    fs: sampling rate
    w: analysis window (odd size)
    N: FFT size (minimum 512, even)
    t: threshold in dB (negative) 
    maxnS: maximum number of sinusoids
    stocf: decimation factor of mag spectrum for stochastic analysis
    Ns = 1024: FFT size for synthesis (even)
    H = 256: hop size for analysis and synthesis
    effect: {'filtering','freqshift','freqstretch','freqscale'}.
            It will requires additional parameter according to effect

    'filtering': Apply an arbitrary filter to spectrum
        Filter: array (2,nf), with frequencies in Hz in first line  and
                    magnitude in dB in second line
            Example:
            filterFreq = np.array([   0, 2099, 2100, 3000, 3001, fs/2]) # Hz
            filterMag  = np.array([-200, -200,    0,    0, -200, -200]) # db
            Filter = np.vstack((filterFreq,filterMag))

    'freqshift': Frequency shifting
        fshift: frequency shift factor, in Hz

    'freqstretch': Frequency stretching
        fstretch: stretching factor to the frequency to each partial
                  f[i] = f[i] * (fstretch)^(i−1)

    'freqscale': Frequency scaling (a pitch shifter without timbre preservation)
        fscale: scaling factor to all partials 
    
   
    ---- Output ----
    y: output sound
    yh: harmonic component
    ys: stochastic component
    '''

    M = len(w);                              # analysis window size
    N = max(N,512)
    N2 = N//2+1;                             # half-size of spectrum
    soundlength = len(x);                    # length of input sound array
    hNs = Ns//2;                             # half synthesis window size
    hM = (M-1)//2;                           # half analysis window size
    pin = max(H,hM); # initialize sound pointer to middle of analysis window
    pend = soundlength-max(hM,H)-1;          # last sample to start a frame
    fftbuffer = np.zeros(N);                 # initialize buffer for FFT
    yh = np.zeros(soundlength+Ns//2);         # output sine component
    ys = np.zeros(soundlength+Ns//2);         # output residual component
    w = w/w.sum();                           # normalize analysis window
    sw = np.zeros(Ns);
    ow = sig.windows.triang(2*H-1);   # overlapping window
    ovidx = np.arange(Ns//2+1-H,Ns//2+H,1,dtype='int64') # overlap indexes
    sw[ovidx] = ow[:2*H-1];
    bh = sig.windows.blackmanharris(Ns);     # synthesis window
    bh = bh / bh.sum();                      # normalize synthesis window
    wr = bh.copy()                           # window for residual 
    sw[ovidx] = sw[ovidx] / bh[ovidx];
    sws = H*sig.windows.hann(Ns+2)/2;        # synthesis window for stochastic
    sws = sws[1:-1]       # used hanning(Ns) in Matlab, which cuts zeros 
    lastysloc = np.zeros(maxnS);             # initialize synthesis harmonic locations
    ysphase = 2*np.pi*np.random.uniform(0,1,maxnS); # initialize synthesis harmonic phases
    fridx = 0;

    while pin<pend:
        #-----analysis-----
        xw = x[pin-hM:pin+hM+1]*w;             # window the input sound
        fftbuffer = fftbuffer*0;               # reset buffer;
        fftbuffer[:(M+1)//2] = xw[(M+1)//2-1:M]# zero-phase window in fftbuffer
        fftbuffer[N-(M-1)//2:] = xw[:(M-1)//2];
        X = np.fft.fft(fftbuffer);             # compute the FFT
        mX = 20*np.log10(abs(X[:N2]));         # magnitude spectrum 
        pX = np.unwrap(np.angle(X[:N2]));      # unwrapped phase spectrum 
        auxploc = np.where(mX[1:-1]>t,1,0) * np.where(mX[1:-1]>mX[2:],1,0) * np.where(mX[1:-1]>mX[:-2],1,0)
        ploc = 1 + np.where(auxploc>0)[0]      # find peaks
        #ploc = 1 + find((mX(2:N2-1)>t) .* (mX(2:N2-1)>mX(3:N2)) ...
        #              .* (mX(2:N2-1)>mX(1:N2-2)));          % find peaks
        ploc,pmag,pphase = peakinterp(mX,pX,ploc);  # refine peak values  
        if type(ploc)!=np.ndarray:
            ploc = np.array([ploc])
            print('ploc')
        if type(pmag)!=np.ndarray:
            pmag = np.array([pmag])
            print('pmag')
        if type(pphase)!=np.ndarray:
            pphase = np.array([pphase])
            print('phase')
        # sort by magnitude
        #[smag,I] = sort(pmag(:),1,'descend');
        smag = np.sort(pmag)
        smag = smag[::-1]
        I = np.argsort(pmag)
        I = I[::-1]

        nS = min(maxnS,len(np.where(smag>t)[0]));
        sloc = ploc[I[:nS]];
        sphase = pphase[I[:nS]];  
      
        testZero=0; #added to avoid error where there is a frame with no peak
        if (fridx==0):
            # update last frame data for first frame
            lastnS = nS;
            lastsloc = sloc.copy();
            lastsmag = smag.copy();
            lastsphase = sphase.copy();
        elif(lastnS==0): #added to avoid error where there is a frame with no peak
            lastnS = nS;
            lastsloc = sloc.copy();
            lastsmag = smag.copy()
            lastsphase = sphase.copy();
            testZero = 1
        # connect sinusoids to last frame lnS (lastsloc,lastsphase,lastsmag)
        sloc[:nS] = np.where(sloc!=0,1,0)*(sloc*Ns/N);  # synth. locs
        lastidx = np.zeros(nS,dtype='int64');
        for i in range(nS):
            #[dev,idx] = min(abs(sloc(i) - lastsloc(1:lastnS)));
            auxdev = abs(sloc[i]-lastsloc)
            dev = auxdev.min()
            idx = np.where(auxdev==dev)[0][0]
            lastidx[i] = idx;

        ri= pin-hNs;                     # input sound pointer for residual analysis
        xr = x[ri:ri+Ns]*wr;             # window the input sound
        Xr = np.fft.fft(np.fft.fftshift(xr));    # compute FFT for residual analysis
        Xh = genspecsines(sloc,smag,sphase,Ns);  # generate sines
        Xr = Xr-Xh;                              # get the residual complex spectrum
        mXr = 20*np.log10(abs(Xr[:Ns//2+1]));    # magnitude spectrum of residual
        mXsenv = sig.decimate(np.where(mXr<-200,-200,mXr),stocf);  # decimate the magnitude spectrum
                                                                   # and avoid -Inf
        #-----synthesis data-----
        ysloc = sloc.copy();                     # synthesis locations
        ysmag = smag[:nS];                       # synthesis magnitudes
        mYsenv = mXsenv.copy()                   # synthesis residual envelope


        #-----transformations-----
        if effect=='filtering': #Filtering
            if filter is None:
                filterFreq= np.array([   0, 2099, 2100, 3000, 3001, fs/2]) # Hz
                filterMag  = np.array([-200, -200,    0,    0, -200, -200]) # db
                Filter = np.vstack((filterFreq,filterMag))
            ysmag = ysmag+np.interp(ysloc/Ns*fs, Filter[0,:],Filter[1,:]);

        elif effect=='freqshift': #Frequency shifting
            if fshift is None:
                fshift = 100;
            ysloc = np.where(ysloc>0,1,0)*(ysloc + fshift/fs*Ns); # frequency shift in Hz

        elif effect=='freqstretch': #Frequency stretching
            if fstretch is None:
                fstretch = 1.1
            ysloc = ysloc * (fstretch ** np.arange(len(ysloc)))

        elif effect == 'freqscale': #Frequency scaling
            if fscale is None:
                fscale = 1.2
            ysloc = ysloc*fscale;



        #-----synthesis-----
        if (fridx==0):
            lastysphase = ysphase.copy();
        elif (testZero==1):
            lastysloc = np.zeros(maxnS);  #initial lastysloc
            lastysphase = 2*np.pi*np.random.uniform(0,1,maxnS); # initial yphase
    
        if (nS>lastnS):
            lastysphase = np.hstack((lastysphase, np.zeros(nS-lastnS)));
            lastysloc = np.hstack((lastysloc,np.zeros(nS-lastnS)));
        ysphase = lastysphase[lastidx] + 2*np.pi*(lastysloc[lastidx]+ysloc)/2/Ns*H; # propagate phases
        lastysloc = ysloc.copy();
        lastysphase = ysphase.copy();  
        lastnS = nS;                             # update last frame data
        lastsloc = sloc;                           # update last frame data
        lastsmag = smag;                           # update last frame data
        lastsphase = sphase;                       # update last frame data
        Yh = genspecsines(ysloc,ysmag,ysphase,Ns); # generate sines
        mYs = interpolate_1d_vector(mYsenv,stocf); # interpolate to original size
        roffset = int(np.ceil(stocf/2))-1;         # interpolated array offset
        mYs = np.hstack((mYs[0]*np.ones(roffset), mYs[:Ns//2+1-roffset]));
        mYs = 10**(mYs/20);                        # dB to linear magnitude
        pYs = 2*np.pi*np.random.uniform(0,1,(Ns//2)+1); # generate phase spectrum with random values
        mYs1 = np.hstack((mYs[:Ns//2+1], mYs[Ns//2-1:0:-1])); # create complete magnitude spectrum
        pYs1 = np.hstack((pYs[:Ns//2+1],-1*pYs[Ns//2-1:0:-1])); # create complete phase spectrum
        Ys = mYs1*np.cos(pYs1)+1j*mYs1*np.sin(pYs1);   # compute complex spectrum
        yhw = np.fft.fftshift(np.real(np.fft.ifft(Yh))); # sines in time domain using inverse FFT
        ysw = np.fft.fftshift(np.real(np.fft.ifft(Ys))); # stochastic in time domain using IFFT
        yh[ri:ri+Ns] += yhw * sw;                        # overlap-add for sines
        ys[ri:ri+Ns] += ysw * sws;                       # overlap-add for stochastic
        pin = pin+H;                                     # advance the sound pointer
        fridx += 1;

    y= yh+ys;                                     # sum sines and stochastic
    return y, yh, ys

if __name__=='__main__':
    import audiofile as af
    import matplotlib.pyplot as plt

    #inputFile='tedeum2.wav'
    inputFile='soprano-E4.wav'
    x, fs = af.read(inputFile);
    w = sig.windows.hann(1025,sym=False);
    N = 1024;
    t = -60;
    maxnS = 10;
    stocf = 4;

    #Without effect
    
    y, yh, ys = spsmodel(x,fs,w,N,t,maxnS,stocf)
    auxName = inputFile.split('.wav')[0]
    auxName = auxName+'_t'+str(t)+'_nS'+str(maxnS)
    af.write(auxName+'_hps.wav',y,fs)
    af.write(auxName+'_h.wav',yh,fs)
    af.write(auxName+'_s.wav',ys,fs)
    
    #filtering
    inputFile='soprano-E4.wav'
    x, fs = af.read(inputFile);
    # D4# =  311.13 Hz
    # E4 = 329.63 Hz
    # F4 = 349.23 Hz
    filterFreq= np.array([   0,   310,  311,  350,  351, fs/2]) # Hz
    filterMag  = np.array([-200, -200,    0,    0, -200, -200]) # db
    Filter = np.vstack((filterFreq,filterMag))

    y, yh, ys = spsmodel(x,fs,w,N,t,maxnS,stocf,effect='filtering', Filter=Filter)

    auxName = inputFile.split('.wav')[0]
    auxName = auxName+'_filter_t'+str(t)+'_nS'+str(maxnS)
    af.write(auxName+'_hps.wav',y,fs)
    af.write(auxName+'_h.wav',yh,fs)
    af.write(auxName+'_s.wav',ys,fs)
    
    #frequency shifting
    # D4# =  311.13 Hz
    # E4 = 329.63 Hz
    # F4 = 349.23 Hz
    # G4 = 392.00 Hz
    fshift = (392-329.63)
    y, yh, ys = spsmodel(x,fs,w,N,t,maxnS,stocf,effect='freqshift', fshift=fshift)

    auxName = inputFile.split('.wav')[0]
    auxName = auxName+'_fshift_'+str(round(fshift))+'_t'+str(t)+'_nS'+str(maxnS)
    af.write(auxName+'_hps.wav',y,fs)
    af.write(auxName+'_h.wav',yh,fs)
    af.write(auxName+'_s.wav',ys,fs)
    
    #frequency stretching
    # D4# =  311.13 Hz
    # E4 = 329.63 Hz
    # F4 = 349.23 Hz
    # G4 = 392.00 Hz
    
    fstretch = 0.9# 2**(2/12)
    y, yh, ys = spsmodel(x,fs,w,N,t,maxnS,stocf,effect='freqstretch', fstretch=fstretch)

    auxName = inputFile.split('.wav')[0]
    auxName = auxName+'_fstrech_'+str(round(fstretch,2))+'_t'+str(t)+'_nS'+str(maxnS)
    af.write(auxName+'_hps.wav',y,fs)
    af.write(auxName+'_h.wav',yh,fs)
    af.write(auxName+'_s.wav',ys,fs)
    
    #frequency scaling
    fscale = 2**(2/12)
    y, yh, ys = spsmodel(x,fs,w,N,t,maxnS,stocf,effect='freqscale', fscale=fscale)

    auxName = inputFile.split('.wav')[0]
    auxName = auxName+'_fscale_'+str(round(fscale,2))+'_t'+str(t)+'_nS'+str(maxnS)
    af.write(auxName+'_hps.wav',y,fs)
    af.write(auxName+'_h.wav',yh,fs)
    af.write(auxName+'_s.wav',ys,fs)
