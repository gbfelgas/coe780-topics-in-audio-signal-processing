# TimeScaleSOLA.m
def TimeScaleSOLA(x, fs, alpha, Sa=256, N=2048, L=None):
    '''
    Time Scaling with Synchronized Overlap and Add

    INPUTS:
    -------------------------------------------------------
    x          signal mono or multi-channel               
    fs         sampling frequency
    alpha      time scaling factor  (0.25 <= alpha <= 2)
    Sa         analysis hop size    (Sa = 256 (default parameter))
    N          block length         (N  = 2048 (default parameter))
    L          overlap interval     (L must be chosen to be less than N-Ss.
                                     if None, L  = round(256*alpha/2) )

    OUTPUT:
    -------------------------------------------------------
    y              stretched signal with same sampling frequency

    '''
    import numpy as np
    
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
    
    nChan = x_adj.shape[1]
    lenX = x_adj.shape[0]

    #Other parameters
    Ss = round(Sa*alpha);     
    if L is None: L = round(Sa*alpha/2)

    # Segmentation into blocks of length N every Sa samples
    # leads to M segments
    M = int(np.ceil(lenX/Sa))

    DAFx_in = np.zeros((M*Sa+N,nChan))
    DAFx_in[:lenX,:] = x_adj.copy()
    Overlap = DAFx_in[:N,:]

    # **** Main TimeScaleSOLA loop ****
    for ni in range(M-1):
        grain = DAFx_in[(ni+1)*Sa:N+(ni+1)*Sa,:]
        # calculate correlation for each channel
        xCorrel = np.zeros((2*L-1,nChan))
        for ch in range(nChan):
            xCorrel[:,ch] = np.correlate(grain[:L,ch],Overlap[(ni+1)*Ss:(ni+1)*Ss + L,ch],mode='full')	
        auxMax = np.where(xCorrel==xCorrel.max())
        km = auxMax[0][0] - (L-1)
        startCrossFade = (ni+1)*Ss - km
        lenCrossFade = Overlap.shape[0] - (startCrossFade + 1)
        if lenCrossFade > N: lenCrossFade = N
        fadeout = np.arange(lenCrossFade-1,-1,-1)/(lenCrossFade-1);
        fadeout = np.tile(fadeout,nChan).reshape((nChan,lenCrossFade)).T
        fadein = np.arange(0,lenCrossFade,1)/(lenCrossFade-1);
        fadein = np.tile(fadein,nChan).reshape((nChan,lenCrossFade)).T
        Tail = Overlap[startCrossFade:(startCrossFade+lenCrossFade),:] * fadeout;

        Begin = grain[:lenCrossFade,:] * fadein;
        Add = Tail + Begin;
        Overlap = np.concatenate((Overlap[:startCrossFade,:],Add, grain[lenCrossFade:,:]),axis=0);
    # **** end TimeScaleSOLA loop ****

    #return y according to original x shape
    if x.ndim == 1:
        return Overlap[:,0]
    else:
        if x.shape[0] == x_adj.shape[0]:
            return Overlap
        else:
            return Overlap.T

#Test
if __name__=='__main__':
    import audiofile as af
    import numpy as np

    inputFile = 'medicamento.wav'
    alpha = 1.3
    x, fs = af.read(inputFile)
    name = inputFile.split('.wav')[0]
    y = TimeScaleSOLA(x,fs,alpha)

    af.write(name+'_out_'+str(alpha)+'.wav',y,fs)
