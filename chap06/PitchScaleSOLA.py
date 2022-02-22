# PitchScaleSOLA.m
from TimeScaleSOLA import TimeScaleSOLA
import numpy as np
import scipy.signal as sig

def PitchScaleSOLA(signal, fs, Sa=256, N=2048, n1=None, n2=None, semitones=None, normToSignal = False):
    '''
    Pitch change applyinh time scaling with Synchronized Overlap and Add

    INPUTS:
    -------------------------------------------------------
    signal          signal mono or multi-channel               
    fs              sampling frequency
    Sa              analysis hop size    (Sa = 256 (default parameter))
    N               block length         (N  = 2048 (default parameter))
    n1, n2          pitch scaling n1/n2
                    (expands the input signal from length N1 to length N2,
                     then resamples to reduce from length N2 back to length N1)
    semitones       define semitones pitch (if used, n1 and n2 will be ignored)
    normToSignal    normalize to original signal peak.
    
    OUTPUT:
    -------------------------------------------------------
    DAFx_out        pitch scaled signal with same time length and sampling frequency

    '''    
    
    #Adapting x shape to (sample, channel)
    if signal.ndim == 1:
        x_adj = signal.reshape((signal.shape[0],1))
    elif signal.ndim == 2:
        if signal.shape[0]>signal.shape[1]:
            x_adj = signal.copy()
        else:
            x_adj = signal.T.copy()
    else:
        raise TypeError('unknown audio data format !!!')
        return

    nChan = x_adj.shape[1]
    lenX = x_adj.shape[0]

    #Other parameters
    if not(semitones is None):
        n2 = Sa
        n1 = round(n2 * (2**(semitones/12)))

    alpha = n1/n2
    L = round(Sa*alpha/2)
    # Segmentation into blocks of length N every Sa samples leads to M segments
    M = int(np.ceil(lenX/Sa))
    len_DAFx_in = M*Sa+N
    
    # ****** Time Stretching with alpha=n2/n1******
    Overlap = TimeScaleSOLA(x_adj, fs, alpha, Sa, N, L)
    
    # ****** Pitch shifting with alpha=n1/n2 ******
    lfen=N                    #Original lfen=2048
    lfen2=round(lfen/2);
    w1= sig.windows.hann(lfen,False) # hanningz(lfen)
    #Adjusting shape to (sample,numChannel)
    w1 = np.tile(w1,nChan).reshape((nChan,len(w1))).T
    w2    = w1.copy()    # output window (not applied)
    #for linear interpolation of a grain of length lx to length lfen
    lx = int(np.floor(lfen*n1/n2))
    x = np.arange(lfen)*lx/lfen;
    ix = np.int64(np.floor(x));
    ix1 = ix+1;
    dx = x-ix;

    #Adjusting shape to (sample,numChannel)
    dx = np.tile(dx,nChan).reshape((nChan,len(dx))).T

    dx1 = 1-dx;

    lmax=max(lfen,lx);
    DAFx_out=np.zeros((len_DAFx_in,nChan));

    pin=0;
    pout=0;
    pend=Overlap.shape[0]-lmax;
    
    #Pitch shifting by resampling a grain of length lx to length lfen
    while pin<pend:
        grain2= (Overlap[pin+ix,:]*dx1 + Overlap[pin+ix1,:]*dx) * w1; 
        DAFx_out[pout:pout+lfen,:] = DAFx_out[pout:pout+lfen,:]+grain2;
        pin=pin+n1
        pout=pout+n2;

    #Normalize to original max value
    DAFx_out = DAFx_out/abs(DAFx_out).max() * abs(signal).max()
    
    #return DAFx_out according to original signal shape
    if signal.ndim == 1:
        return DAFx_out[:,0]
    else:
        if signal.shape[1] == x_adj.shape[1]:
            return DAFx_out
        else:
            return DAFx_out.T

#Test
if __name__=='__main__':
    import audiofile as af
    import matplotlib.pyplot as plt

    #Sample File
    inputFile = 'x2.wav'
    
    x, fs = af.read(inputFile)
    name = inputFile.split('.wav')[0]
    semitones = [-12,-2,-1,1,2,12]

    for st in semitones:
        y = PitchScaleSOLA(x,fs,semitones=st)
        af.write(name+'_pitch_'+str(st)+'.wav',y,fs)

    #Spectrogram, 2*f
    win = sig.get_window('hann', 1024, fftbins=True)
    wCut = 1000
    f, t, Sxx = sig.spectrogram(x, window=win, scaling='spectrum', noverlap=len(win)//2)
    yf, yt, Syy = sig.spectrogram(y, window=win,scaling='spectrum', noverlap=len(win)//2)

    plt.figure(figsize=(14,6))
    plt.subplot(121)
    plt.pcolormesh(t, f*fs/2, Sxx, shading='gouraud',cmap='gray')
    plt.title('Spectrogram - Original')
    plt.ylabel('Frequency [Hz]'+r' $\rightarrow$')
    plt.xlabel(r'n $\rightarrow$')
    plt.axis(ymax=wCut)
    plt.colorbar()
    
    plt.subplot(122)
    plt.pcolormesh(yt, yf*fs/2, Syy, shading='gouraud', cmap='gray')
    plt.ylabel('Frequency [Hz]'+r' $\rightarrow$')
    plt.xlabel(r'n $\rightarrow$')
    plt.axis(ymax=wCut)
    plt.colorbar()

    plt.show()

    #Vocalize artificial
    inputFile = 'voc_B.wav'
    
    x1, fs = af.read(inputFile)
    x2 = PitchScaleSOLA(x1,fs,semitones=2,normToSignal= True)
    x3 = PitchScaleSOLA(x1,fs,semitones=4,normToSignal= True)
    x4 = PitchScaleSOLA(x1,fs,semitones=5,normToSignal= True)
    x5 = PitchScaleSOLA(x1,fs,semitones=7,normToSignal= True)
    x5Longo = TimeScaleSOLA(x5,fs,4.4)
    x1Longo = TimeScaleSOLA(x1,fs,4.4)
    y = np.concatenate((x1,x2,x3,x4,x5Longo,np.zeros(x1.shape),x5,x4,x3,x2,x1Longo),axis=1)
    af.write('vocalize_artificial.wav',y,fs)

