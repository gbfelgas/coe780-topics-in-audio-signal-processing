import numpy as np

def FIRChirpLinear(n=300,fs=44100,f0=50,f1=4000, returnFreq=False):
    '''
    '''
    
    x = (np.arange(n)+1)/n
    freq = 2*np.pi * (f0+(f1-f0)*x) / fs;
    fir = np.sin(freq.cumsum())

    if returnFreq:
        return fir, freq
    else:
        return fir

def FIRChirpExponential(n=300,fs=44100,f0=50, f1=4000, returnFreq=False):
    '''
    '''
    x = (np.arange(n)+1)/n
    rap = f1/f0;
    freq = (2*np.pi*f0/fs) * (rap**x);
    fir = np.sin(freq.cumsum())

    if returnFreq:
        return fir, freq
    else:
        return fir


def FIRDesign3(M = 300,WLen = 1024):
    '''
    '''
    mask = np.concatenate((np.array([1]), 2*np.ones(WLen//2-1),np.array([1]), np.zeros(WLen//2-1)));
    fs = M* np.arange(WLen//2+1)/ WLen; # linear increasing delay
    teta = np.concatenate(((-2*np.pi*fs)*np.arange(WLen//2+1)/WLen, np.zeros(WLen//2-1)));
    f2 = np.exp(1j*teta);
    fir = np.fft.fftshift(np.real(np.fft.ifft(f2*mask)));
    
    return fir


if __name__=='__main__':
    import matplotlib.pyplot as plt
    import audiofile as af
    import scipy.signal as sig
    
    firLin, freqLin = FIRChirpLinear(returnFreq=True)
    firExp, freqExp = FIRChirpExponential(returnFreq=True)
    fs=44100
    n=300
    freqLin = freqLin/(2*np.pi)*fs
    freqExp = freqExp/(2*np.pi)*fs
    
    plt.figure(figsize=(10,6));
    plt.subplot(221);
    plt.plot(freqLin,'k--',lw=0.5, label='Linear');
    plt.plot(freqExp,'k-',lw=0.5, label='Exponential');
    plt.legend()
    plt.xlabel('n')
    plt.ylabel('f')

    plt.subplot(222);
    n1 = (np.arange(n))
    n2 = (np.arange(2*n))
    freqLin2 = np.concatenate((np.tile(freqLin[0],n),freqLin))
    freqExp2 = np.concatenate((np.tile(freqExp[0],n),freqExp))
    plt.plot(n1,freqLin,'k--',lw=0.5, label='Linear');
    plt.plot(n2[-n:],freqLin2[-n:],'k--',lw=0.5);
    plt.plot(n1, freqExp,'k-',lw=0.5, label='Exponential');
    plt.plot(n2, freqExp2,'k-',lw=0.5);
    plt.plot([n,2*n],[freqExp[-1],freqExp2[-1]],'k-',lw=0.5)
    plt.legend()
    plt.xlabel('n')
    plt.ylabel('f')

    plt.subplot(223);
    plt.plot(firLin,'k--', lw=0.5, label='Linear');
    plt.plot(firExp,'k-',lw=0.5, label='Exponential');
    plt.legend();
    plt.xlabel('n')
    plt.ylabel('s(n)')
    
    fig = plt.gcf()
    fig.suptitle('Dispersion - Chirp')
    plt.tight_layout()
    plt.show()

    fird3 = FIRDesign3()
    w3,H3 = sig.freqz(fird3)
    
    plt.figure(figsize=(14,4))
    plt.subplot(131)
    plt.plot(fird3,'k-',lw=0.5)
    plt.xlabel(r'n $\rightarrow$')
    plt.ylabel(r'h(n) $\rightarrow$')
    plt.subplot(132)
    plt.plot(w3/np.pi,abs(H3),'k-',lw=0.5)
    plt.xlabel(r'$\omega/\pi \rightarrow$')
    plt.ylabel(r'$|H(e^{j\omega})| \rightarrow$')
    plt.axis(ymin=0, ymax=1.2)
    plt.subplot(133)
    plt.plot(w3/np.pi,np.angle(H3),'k-',lw=0.5)
    plt.xlabel(r'$\omega/\pi \rightarrow$')
    plt.ylabel(r'$\angle H(e^{j\omega}) \rightarrow$')
    

    fig = plt.gcf()
    fig.suptitle('Dispersion - "Design 3"')
    plt.tight_layout()
    plt.show()

    
    fs=44100
    firLin = FIRChirpLinear(n = fs*4)
    firExp = FIRChirpExponential(n = fs*4)
    af.write('chirp_linear.wav',0.7*firLin,fs)
    af.write('chirp_exponential.wav',0.7*firExp,fs)
