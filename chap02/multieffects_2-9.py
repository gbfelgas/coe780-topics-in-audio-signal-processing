# Author: S. Disch
#
#--------------------------------------------------------------------------
# This source code is provided without any warranties as published in 
# DAFX book 2nd edition, copyright Wiley & Sons 2011, available at 
# http://www.dafx.de. It may be used for educational purposes and not 
# for commercial applications without further permission.
#--------------------------------------------------------------------------
import numpy as np
import scipy.signal as sig

def LowPass1st(x,fs,fc):
    '''
       
    Apply a 1st-order Low Pass filter to a signal

    Parameters
    ----------
    x:  numpy.ndarray
        one (mono audio) or two (multichannel audio) dimensions array with sampled audio
        in case of 2-dim array, the smaller dimension is considered as the channel
    fs: float
        Sample frequency (in Hertz) used for digital audio
    fc: float
        Cut-off frequency (in Hertz)
    
    Return
    ------
    y:  numpy.ndarray
        Filtered audio

    '''  
    
    K = np.tan(np.pi*fc/fs)
    b_lp = [K/(K+1), K/(K+1)]
    a_lp = [1, (K-1)/(K+1)]
    if x.ndim == 1:
        ax = 0
    elif x.ndim == 2:
        if x.shape[0]>x.shape[1]:
            ax = 1
        else:
            ax = 0
    else:
        raise TypeError('unknown audio data format !!!')
        return
    
    y = sig.lfilter(b_lp,a_lp,x,axis = ax)

    return y

def multieffects(x,SAMPLERATE,BL,FF,FB,Delay,Depth,ModType,ModFreq,Interpolation = 'linear'):
    '''
       
    Apply vibrato, flanger, chorus and doubling to a sampled audio
    according to parameters below:

    Effect   |  BL |  FF |  FB |  DELAY  |  DEPTH  |     MOD     |
    ---------|-----|-----|-----|---------|---------|-------------|
    Vibrato  |  0  |  1  |  0  |   0 ms  |  0-3 ms |0.1-5 Hz sine|
    Flanger  | 0.7 | 0.7 | 0.7 |   0 ms  |  0-2 ms |0.1-1 Hz sine|
    Chorus   | 0.7 |  1  |-0.7 | 1-30 ms | 1-30 ms |Lowpass noise|
    Doubling | 0.7 | 0.7 |  0  |10-100 ms|1-100 ms |Lowpass noise|

    Parameters
    ----------
    x:  numpy.ndarray
        one (mono audio) or two (multichannel audio) dimensions array with sampled audio
        in case of 2-dim array, the smaller dimension is considered as the channel
    SAMPLERATE: float
        Sample rate (in Hertz) used for digital audio
    BL: float
        Original signal bypass coeficient
    FF: float
        Feedfoward coeficient for delayed signal
    FB: float
        Feedback coeficient for delayed signal
    Delay: float
        Reference (in seconds)
    ModType: {'sine', 'noise'}
        Type of delay modulation
        'sine'  :  sine oscilatin with ModFreq frequency
        'noise' :  white noise filtered by a Low Pass filter with cut-off
                   frequency 'ModFreq'
    ModFreq: float
        Modulation frequency (in Hertz)
        In case of ModType == 'sine', it corresponds to LFO frequency
        In case of ModType == 'noise', if corresponds to LP cut-off frequency
    Depth:  float
        Delay modulation depth (in seconds). Modulation oscilates in a range
        of length '2 * Depth' (from -Depth to +Depth)
    Interpolation:  {'linear', 'allpass', 'spline3'}
        Interpolation algorythm
        'linear'  :  Linear interpolation
        'allpass' :  allpass interpolation
        'spline3' :  3rd order spline interpolation

    Return
    ------
    y:  numpy.ndarray
        Sampled audio with effect
        
    '''
    DEPTH = round(Depth * SAMPLERATE)   # modulation width in # samples
    DELAY = round(Delay * SAMPLERATE)   # basic delay in # samples
    
    # additional delay due to modulation to keep it causal, in # samples
    if Interpolation.casefold() == 'spline3':
        ADDDELAY = DEPTH + 2            # 3rd order Spline uses x(n-[M-2])
    else:
        ADDDELAY = DEPTH

    MODFREQ = ModFreq/SAMPLERATE        # modulation frequency in sample^-1

    #Adapting x shape to (sample, channel)
    if x.ndim == 1:
        x_adj = x.reshape((x.shape[0],1))
    elif x.ndim == 2:
        if x.shape[0]>x.shape[1]:
            x_adj = x.copy()
        else:
            x_adj = x.T.copy()
    else:
        raise TypeError('unknown audio data format !!!')
        return
    
    nChan = x_adj.shape[1]                         # # of audio channels (mono/muti-channel)
    xh_n_k = np.zeros(nChan)                       # output from mod delay
    LEN = x_adj.shape[0]                           # # of samples in WAV-file
    L = DELAY + ADDDELAY + DEPTH + 2               # length of the entire delay (maximized) 
    Delayline = np.zeros((L, x_adj.shape[1]))      # memory allocation for delay
    y = np.zeros(x_adj.shape, dtype = x_adj.dtype) # memory allocation for output vector

    #Preparing delay modulation signal
    if ModType.casefold() == 'sine':
        aux_n = np.arange(LEN,dtype='float64')
        MOD_TOTAL = np.sin(2 * np.pi * MODFREQ * aux_n)
    elif ModType.casefold() == 'noise':
        np.random.seed(int(MODFREQ))
        whiteNoise = np.random.uniform(-1,1,LEN)
        MOD_TOTAL = LowPass1st(whiteNoise,SAMPLERATE,MODFREQ)
    else:
        raise TypeError('unknown modulation type !!!')

    for n in range(LEN):

        xh_n = x_adj[n,:] + FB * xh_n_k
        Delayline = np.concatenate((xh_n.reshape((1,nChan)), Delayline[:-1,:]))

        #Delay modulation
        MOD = MOD_TOTAL[n]
        TAP = DELAY + ADDDELAY + DEPTH * MOD
        i = int(np.floor(TAP))
        frac = TAP - i
        
        if Interpolation.casefold() == 'linear':
            #---Linear Interpolation---------------------------------
            xh_n_k = Delayline[i+1,:]*frac + Delayline[i,:]*(1-frac)
        elif Interpolation.casefold() == 'allpass':
            #---Allpass Interpolation--------------------------------
            xh_n_k = Delayline[i+1,:] + (1-frac)*Delayline[i,:] - (1-frac)*xh_n_k
        elif Interpolation.casefold() == 'spline3':
            #--3rd-order Spline Interpolation------------------------
            xh_n_k = Delayline[i+1,:]*(frac**3)/6 \
                   + Delayline[i  ,:]*((1+frac)**3 - 4*(frac**3))/6 \
                   + Delayline[i-1,:]*((2-frac)**3 - 4*((1-frac)**3))/6 \
                   + Delayline[i-2,:]*((1-frac)**3)/6
        else:
            raise TypeError('unknown interpolation algorythm !!!')
            return

        y[n,:] = xh_n * BL + xh_n_k * FF
        
    #return y according to original x shape
    if x.ndim == 1:
        return y[:,0]
    else:
        if x.shape[0] == x_adj.shape[0]:
            return y
        else:
            return y.T
