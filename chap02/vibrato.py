def vibrato(x,SAMPLERATE,Modfreq,Width,Interpolation = 'linear'):
    '''
       
    Apply a vibrato to a sampled audio

    Parameters
    ----------
    x:  numpy.ndarray
        One or two dimensions array (mono/multi-channel audio) with sampled audio
        In case of 2-dim array, the smaller dimension is considered as the channel
    SAMPLERATE: float
        Sample rate (in Hertz) used for digital audio
    Modfreq: float
        Modulation frequency of LFO (in Hertz)
    Width:  float
        Modulation width (in seconds).
        The modulation oscilate in a range of 2 * Width (from -Width to +Width)
    Interpolation:  {'linear', 'allpass', 'spline3'}
        Interpolation algorythm
        'linear'  :  Linear interpolation
        'allpass' :  allpass interpolation
        'spline3' :  3rd order spline interpolation

    Return
    ------
    y:  numpy.ndarray
        Sampled audio with vibrato
        
    '''
    import numpy as np
    WIDTH = round(Width * SAMPLERATE)   # modulation width in # samples

    # basic delay in # samples
    if Interpolation.casefold() == 'spline3':
        DELAY = WIDTH + 2               # 3rd order Spline uses x(n-[M-2])
    else:
        DELAY = WIDTH

    MODFREQ = Modfreq/SAMPLERATE        # modulation frequency in sample^-1

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
    
    nChan = x_adj.shape[1]                         # # of audio channels (mono/multi-channel)
    ya_alt = np.zeros(nChan)                       # used with Allpass interp.
    LEN = x_adj.shape[0]                           # # of samples in WAV-file
    L = 2 + DELAY + WIDTH                          # length of the entire delay  
    Delayline = np.zeros((L, x_adj.shape[1]))      # memory allocation for delay
    y = np.zeros(x_adj.shape, dtype = x_adj.dtype) # memory allocation for output vector

    for n in range(LEN):
        MOD = np.sin(MODFREQ * 2 * np.pi * n)
        TAP = DELAY + WIDTH * MOD
        i = int(np.floor(TAP))
        frac = TAP - i
        Delayline = np.concatenate((x_adj[n,:].reshape((1,nChan)), Delayline[:-1,:]))               
        if Interpolation.casefold() == 'linear':
            #---Linear Interpolation---------------------------------
            y[n,:] = Delayline[i+1,:]*frac + Delayline[i,:]*(1-frac)
        elif Interpolation.casefold() == 'allpass':
            #---Allpass Interpolation--------------------------------
            y[n,:] = Delayline[i+1,:] + (1-frac)*Delayline[i,:] - (1-frac)*ya_alt
            ya_alt = y[n,:].copy()
        elif Interpolation.casefold() == 'spline3':
            #--3rd-order Spline Interpolation------------------------
            y[n,:] = Delayline[i+1,:]*(frac**3)/6 \
                   + Delayline[i  ,:]*((1+frac)**3 - 4*(frac**3))/6 \
                   + Delayline[i-1,:]*((2-frac)**3 - 4*((1-frac)**3))/6 \
                   + Delayline[i-2,:]*((1-frac)**3)/6
        else:
            raise TypeError('unknown interpolation algorythm !!!')
            return

    #return y according to original x shape
    if x.ndim == 1:
        return y[:,0]
    else:
        if x.shape[0] == x_adj.shape[0]:
            return y
        else:
            return y.T
