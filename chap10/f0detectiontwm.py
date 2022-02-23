import numpy as np

def f0detectiontwm(mX, fs, ploc, pmag, ef0max, minf0, maxf0):
    '''
    Authors: J. Bonada, X. Serra, X. Amatriain, A. Loscos
    Fundamental frequency detection function
    ---Inputs---
    mX: magnitude spectrum
    fs: sampling rate
    ploc, pmag: peak loc and mag
    ef0max: maximim error allowed
    minf0: minimum f0 (Hz)
    maxf0: maximum f0 (Hz)
    ---Output---
    f0: fundamental frequency detected in Hz
    '''
    N = len(mX)*2;                     # size of complex spectrum
    nPeaks = len(ploc);                # number of peaks
    f0 = 0;                            # initialize output
    if(nPeaks>3):      # at least 3 peaks in spectrum for trying to find f0
        nf0peaks = min(50,nPeaks);     # use a maximum of 50 peaks
        f0,f0error = TWM(ploc[:nf0peaks],pmag[:nf0peaks],N,fs,minf0,maxf0);
        if (f0>0 and f0error>ef0max):  # limit the possible error by ethreshold
            f0 = 0; 
    return f0

def TWM (ploc, pmag, N, fs, minf0, maxf0):
    '''
    ---Inputs---
    Two-way mismatch algorithm (by Beauchamp&Maher)
    ploc, pmag: peak locations and magnitudes,
    N: size of complex spectrum
    fs: sampling rate of sound
    minf0: minimum f0 (Hz)
    maxf0: maximum f0 (Hz)
    ---Outputs---
    f0: fundamental frequency detected (Hz)
    f0error: error measure 
    '''
    pfreq = (ploc)/N*fs;                      # frequency in Hertz of peaks
    #[zvalue,zindex] = min(pfreq);
    zvalue = pfreq.min()
    zindex = np.where(pfreq==zvalue)[0]
    if (zvalue==0):                           # avoid zero frequency peak
        pfreq[zindex] = 1;
        pmag[zindex] = -100;
    
    ival2 = pmag.copy();
    #[Mmag1,Mloc1] = max(ival2);              # find peak with maximum magnitude
    Mmag1 = ival2.max()
    Mloc1 = np.where(ival2==Mmag1)[0][0]
    ival2[Mloc1] = -100;                      # clear max peak
    #[Mmag2,Mloc2]= max(ival2);               # find second maximum magnitude peak
    Mmag2 = ival2.max()
    Mloc2 = np.where(ival2==Mmag2)[0][0]
    ival2[Mloc2] = -100;                      # clear second max peak
    #[Mmag3,Mloc3]= max(ival2);               # find third maximum magnitude peak
    Mmag3 = ival2.max()
    Mloc3 = np.where(ival2==Mmag3)[0][0]
    nCand = 3;                # number of possible f0 candidates for each max peak
    f0c = np.zeros(3*nCand);                  # initialize array of candidates
    f0c[:nCand]=(pfreq[Mloc1]*np.ones(nCand))/((nCand-np.arange(nCand))); # candidates 
    f0c[nCand:nCand*2]=(pfreq[Mloc2]*np.ones(nCand))/((nCand-np.arange(nCand))); 
    f0c[nCand*2:]=(pfreq[Mloc3]*np.ones(nCand))/((nCand-np.arange(nCand))); 
    f0c = f0c[np.where((f0c<maxf0)*(f0c>minf0))[0]];# candidates within boundaries
    if (len(f0c)==0):                         # if no candidates exit
        f0 = 0;
        f0error=100;
        return f0, f0error
    harmonic = f0c.copy();
    ErrorPM = np.zeros(harmonic.shape);       # initialize PM errors
    MaxNPM = min(10,len(ploc));
    for i in range(MaxNPM):    #predicted to measured mismatch error
        difmatrixPM = harmonic.reshape((-1,1))@ np.ones((1,len(pfreq)));
        difmatrixPM = abs(difmatrixPM - np.ones((len(harmonic),1)) @ pfreq.reshape((1,-1)));
        #[FreqDistance,peakloc] = min(difmatrixPM,[],2);
        FreqDistance = difmatrixPM.min(axis=1)
        peakloc = np.zeros(FreqDistance.shape,dtype='int64')
        for j in range(len(FreqDistance)):
            peakloc[j] = np.where(difmatrixPM[j,:]==FreqDistance[j])[0][0]
        Ponddif = FreqDistance * (harmonic**(-0.5));
        PeakMag = pmag[peakloc];
        MagFactor = 10**((PeakMag-Mmag1)/20);
        ErrorPM = ErrorPM + (Ponddif + MagFactor*(1.4*Ponddif-0.5));
        harmonic = harmonic+f0c;

    ErrorMP = np.zeros(harmonic.shape);         # initialize MP errors
    MaxNMP = min(10,len(pfreq));
    for i in range(len(f0c)):                   # measured to predicted mismatch error
        nharm = np.round(pfreq[:MaxNMP]/f0c[i]);
        nharm = np.where(nharm>=1,1,0)*nharm + np.where(nharm<1,1,0);
        FreqDistance = abs(pfreq[:MaxNMP] - nharm*f0c[i]);
        Ponddif = FreqDistance * (pfreq[:MaxNMP]**(-0.5));
        PeakMag = pmag[:MaxNMP];
        MagFactor = 10**((PeakMag-Mmag1)/20);
        ErrorMP[i] = sum(MagFactor*(Ponddif+MagFactor*(1.4*Ponddif-0.5)));
    
    Error = (ErrorPM/MaxNPM) + (0.3*ErrorMP/MaxNMP);  # total errors
    f0error = Error.min();                  # get the smallest error
    f0index = np.where(Error==f0error)[0][0];
    f0 = f0c[f0index];                      # f0 with the smallest error
    return f0, f0error

    
if __name__=='__main__':
    import audiofile as af
    from peakinterp import peakinterp
    import matplotlib.pyplot as plt
    
    x,fs = af.read('audios2\\violin-B3.wav')
    w = np.hanning(1026)
    w = w[:-1].copy()
    M = len(w);                           # analysis window size (odd)
    Ns = 1024;                            # FFT size for synthesis
    H = 256;                              # hop size for analysis and synthesis
    N = 2048
    N2 = N//2+1;                          # half-size of spectrum
    hNs = Ns//2;                          # half synthesis window size
    hM = (M-1)//2;                        # half analysis window size
    pin = max(H,hM);   #initialize sound pointer to middle of analysis window
    pend = len(x)-max(hM,H)-1;            # last sample to start a frame
    fftbuffer = np.zeros(N);              # initialize buffer for FFT
    w = w/w.sum();                        # normalize analysis window
    t=-150
    f0et = 1500
    minf0 = 100
    maxf0 = 400
    saida = []
    xpin = []
    while pin<pend:
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
        saida.append(f0)
        xpin.append(pin)
        pin+=H

    f0B3 = 246.94
    plt.figure()
    plt.plot(xpin,saida,label=r'$F_0$ (measured)')
    plt.plot([xpin[0],xpin[-1]],[f0B3,f0B3],'k--',label='B3')
    plt.title('$F_0$ detection - TWM ("violin-B3.wav")')
    plt.xlabel(r'n $\rightarrow$')
    plt.ylabel(r'Frequency [HZ]')
    plt.legend()
    plt.show()
    
