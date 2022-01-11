#% distfx.m

import numpy as np
import scipy.signal as sig

#% Author: V. Pulkki, T. Lokki
#%
#%--------------------------------------------------------------------------
#% This source code is provided without any warranties as published in 
#% DAFX book 2nd edition, copyright Wiley & Sons 2011, available at 
#% http://www.dafx.de. It may be used for educational purposes and not 
#% for commercial applications without further permission.
#%--------------------------------------------------------------------------
    
def distfx(x, Fs, distsrc, distwall, c = 340, lenRevResp = 0.1, offset=100):
    '''
    INPUTS
    ---------
    x:          signal
    Fs:         Sampling frequency
    distsrc:    distance listener-source, in meters
    distwall:   distance listener-wall, in meters
    c:          sound speed, in m/s
    lenRevResp: length of reverb impulse response, in seconds
    offset:     offset to revert, in samples

    OUTPUTS
    -------
    y:         direct signal
    w:         direct signal + echo
    z:         direct signal + delayed reverb
    '''
    
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

    lenh = round(lenRevResp*Fs)
    np.random.seed(0)
    GNoiseExpDecay = np.random.normal(0,1,lenh) * np.exp(-np.arange(lenh)*0.01/distwall) / 100                                                             
    h = sig.lfilter([0.5,0.5],1,GNoiseExpDecay) #reverb impulse response
                    
    
    if distwall<=distsrc:
        raise TypeError('Distance listener-source should be less than distance listener-wall!!!')
        return
    
    del1 = int(np.floor(distsrc/c*Fs)) #delay source-listener
    del2 = int(np.floor((distwall*2 - distsrc)/c*Fs)) #delay source-wall-listener

    lenOut = max(del1+lenX, del2+lenX, del2 + lenX+lenh-1 + offset)
    y = np.zeros((lenOut,nChan))
    w = np.zeros((lenOut,nChan))
    z = np.zeros((lenOut,nChan))
    
    y[del1:del1+lenX,:] = x_adj/(1+distsrc)              # direct signal 
    w[del2:del2+lenX,:] = x_adj/(1+(2*distwall-distsrc)) # echo
    w = y + w                                            # direct signal + echo    
                 
    reverb = np.zeros((lenX + lenh - 1,nChan))
    for chan in range(nChan):
        reverb[:,chan] = np.convolve(x_adj[:,chan],h)/np.sqrt(1+distsrc)

    z[del2+offset:del2 + offset + lenX+lenh-1,:] = reverb   # delayed reverb
    z = y + z                                               # direct signal + delayed reverb


    #return y,w,z according to original x shape
    if x.ndim == 1:
        return y[:,0], w[:,0], z[:,0]
    else:
        if x.shape[0] == x_adj.shape[0]:
            return y, w, z
        else:
            return y.T, w.T, z.T
