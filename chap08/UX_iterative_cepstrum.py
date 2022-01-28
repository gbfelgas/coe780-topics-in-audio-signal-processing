#UX_iterative_cepstrum.m

import numpy as np
import matplotlib.pyplot as plt

def iterative_cepstrum(x,order,eps,niter,Delta,fs,NF=None,figIndex=0,plotIt=[]):
    '''
    [DAFXbook, 2nd ed., chapter 8]
    This function computes the spectral enveloppe using the iterative 
    cepstrum method

    ----Inputs----
    - x [real, 1-dim array]    signal (windowed)
    - order [int]              cepstrum truncation order (N_1 <= NF/2)
    - eps [float]              bias
    - niter [int]              maximum number of iterations
    - Delta [float]            spectral envelope difference threshold
    - fs [int/float]           sampling frequency
    - NF [int]                 Number of FFT frequency bins
    - figIndex [int]           figure index (>0 to plot spectrum and
                               envelope, otherwise they'll not be plotted)
    - plotIt [int array-like]  list of iteration (1 to niter) to plot
                               the envelope. The last iteration
                               will always be plotted.

    ----Outputs----
    - env [float (NF,)]        magnitude of spectral enveloppe
    - source [complex (NF,)]   complex source
    
    '''
    if NF is None: NF = len(x)
    FT = np.fft.fft(x,n=NF)
    
    #---- drawing ----
    if figIndex>0:
        freq = np.arange(NF)/NF*FS
        plt.figure(figIndex, figsize=(16,7))
        plt.clf()
        plt.subplot(221)
        plt.plot(x,lw=0.5)
        plt.title('Windowed signal x(n)')
        plt.xlabel(r'n $\rightarrow$')

        plt.subplot(222)
        plt.plot(freq,20*np.log10(abs(FT)),lw=0.5)
        plt.title('Spectrum X(f) in dB')
        plt.xlabel(r'n $\rightarrow$')
        plt.xlabel(r'f [Hz] $\rightarrow$');
        plt.axis(xmin=0,xmax=fs/2)

        plt.subplot(224);
        plt.plot(freq, 20*np.log10(abs(FT)), 'grey',lw=0.5,label='spectrum')
        plt.title('Original spectrum and its enveloppe per iteration')        
        plt.xlabel(r'f [Hz] $\rightarrow$')
        plt.ylabel(r'$\rho$(f) / d $\rightarrow$')
        plt.axis(xmin=0,xmax=fs/2)
        

    #---- initializing ----
    Ep = FT.copy()

    #---- computing iterative cepstrum ----
    for k in range(niter):
        flog     = np.log(np.maximum(eps,abs(Ep)));
        cep      = np.fft.ifft(flog);    # computes the cepstrum
        cep_cut  = np.concatenate((np.array([cep[0]]),2*cep[1:order], np.zeros(NF-order)));
        if k==0:
            cep0 = np.real(cep)
            cep_cut0 =np.real(cep_cut) 
        flog_cut = np.real(np.fft.fft(cep_cut));
        env      = np.exp(flog_cut);     # extracts the spectral shape
        Ep       = np.maximum(env, Ep);  # get new spectrum for next iteration
        #---- drawing ----
        if figIndex>0:
            if (k+1) in plotIt:
                plt.figure(figIndex);
                plt.subplot(224);
                plt.plot(freq, 20*np.log10(abs(env)), lw=0.5, label='envelope['+str(k+1)+']');
        #---- convergence criterion ----
        if abs(Ep).max() <= Delta: break

    if figIndex>0:
        plt.figure(figIndex);
        plt.subplot(245)
        plt.plot(np.real(cep0),lw=0.5)
        plt.title('Real cepstrum c(n)')
        plt.xlabel(r'n $\rightarrow$')
        plt.axis(ymin=-1,ymax=1,xmin=-len(x)//400,xmax=len(x)//4)
        plt.subplot(246)
        plt.plot(np.real(cep_cut0),lw=0.5)
        plt.title('Windowed cepstrum '+r'$c_{LP}$(n)')
        plt.xlabel(r'n $\rightarrow$')
        plt.axis(ymin=-1,ymax=1,xmin=-len(x)//400,xmax=len(x)//4)

        plt.subplot(224)
        if not((k+1) in plotIt):
            plt.plot(freq, 20*np.log10(abs(env)), lw=0.5, label='envelope['+str(k+1)+']');
        plt.legend(loc='best')

        plt.tight_layout()

    source = FT / env;
    return env, source

if __name__=='__main__':
    import audiofile as af
    import scipy.signal as sig


    #Example 1 - 'La'
    #Parameters
    filePath = 'la.wav'
    lenWin = 2048
    nIter = 50
    N1 = 150
    Delta = 0.1
    epsilon = 0.001

    #signal
    x,FS = af.read(filePath)
    w = sig.windows.hann(lenWin)
    xw = x[2000:2000+lenWin]*w

    #Results
    env,source = iterative_cepstrum(xw,N1,epsilon,nIter,Delta,FS,figIndex=1,plotIt=[1,5,10,25,50])

    plt.figure(figsize=(12,5))
    freq = np.arange(len(env))/len(env)*FS
    X = np.fft.fft(xw)
    plt.plot(freq,20*np.log10(abs(X)),'gray',lw=0.5,label='spectrum')
    plt.plot(freq,20*np.log10(abs(env)),lw=0.5,label='envelope')
    plt.plot(freq,20*np.log10(abs(source)),lw=0.5,label='source')
    plt.xlabel(r'f [Hz] $\rightarrow$')
    plt.ylabel('Magnitude in dB')
    plt.title('Source and envelope separation - ' + filePath)
    plt.legend()
    plt.axis(xmin=0,xmax=FS/2)
    plt.show()
    
    #Example 2 - 'Guitar'
    #Parameters
    filePath = 'moore_guitar.wav'
    lenWin = 2048
    nIter = 1000
    N1 = 300
    Delta = 0.1
    epsilon = 0.001

    #signal
    x,FS = af.read(filePath)
    w = sig.windows.hann(lenWin)
    xw = x[40000:40000+lenWin]*w
    X = np.fft.fft(xw)
    freq = np.arange(len(X))/len(X)*FS
    
    #Results
    env,source = iterative_cepstrum(xw,N1,epsilon,nIter,Delta,FS,figIndex=1,plotIt=[1,5,10,25,50])
    
    
    plt.figure(3,figsize=(12,8))
    plt.subplot(211)
    plt.plot(freq,20*np.log10(abs(X)),'gray',lw=0.5,label='spectrum')
    plt.plot(freq,20*np.log10(abs(env)),lw=0.5,label='envelope')
    plt.plot(freq,20*np.log10(abs(source)),lw=0.5,label='source')
    plt.xlabel(r'f [Hz] $\rightarrow$')
    plt.ylabel('Magnitude in dB')
    plt.title('Source and envelope separation - ' + filePath + r' - $N_1$ = '+ str(N1))
    plt.legend()
    plt.axis(xmin=0,xmax=FS/2)

    #Parameter change   
    N1 = 150
    
    #Results 2
    env2,source2= iterative_cepstrum(xw,N1,epsilon,nIter,Delta,FS,figIndex=2,plotIt=[1,5,10,25,50])
    
    plt.figure(3)
    plt.subplot(212)
    plt.plot(freq,20*np.log10(abs(X)),'gray',lw=0.5,label='spectrum')
    plt.plot(freq,20*np.log10(abs(env2)),lw=0.5,label='envelope')
    plt.plot(freq,20*np.log10(abs(source2)),lw=0.5,label='source')
    plt.xlabel(r'f [Hz] $\rightarrow$')
    plt.ylabel('Magnitude in dB')
    plt.title('Source and envelope separation - ' + filePath + r' - $N_1$ = '+ str(N1))
    plt.legend()
    plt.axis(xmin=0,xmax=FS/2)
    plt.show()
