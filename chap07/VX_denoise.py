# VX_denoise.m  [DAFXbook, 2nd ed., chapter 7]
import numpy as np
import scipy.signal as sig
from pytictoc import TicToc


def VX_denoise(x, fs, s_win=2048, n1=512, n2=512, coef=0.01, method='sum', normOrigPeak = False):
    '''
    ===== This program makes a denoising of a sound
    
    INPUTS
    ---------------------
    x             signal
    fs            sampling frequency
    s_win         analysis window length [samples]
    n1            analysis step [samples] (s_win/8)
    n2            synthesis step [samples]    
    randomize     {'phase','magnitude'} where the effect will be applied
    coef          denoise coefficient
    method        {'sum','max'} Attenuation method
                   'sum': ft = f*r/(r+coef);
                   'max': ft = f*r/max(r,coef);
    normOrigPeak  normalize according original signal max peak

    OUTPUT
    --------------------
    y             signal with denoising
    '''
    
    #---- Adapting x shape to (sample, channel) ----
    if x.ndim == 1:
        DAFx_in = x.reshape((x.shape[0],1))
    elif x.ndim == 2:
        if x.shape[0]>x.shape[1]:
            DAFx_in = x.copy()
        else:
            DAFx_in = x.T.copy()
    else:
        raise TypeError('unknown audio data format !!!')
        return
    nChan = DAFx_in.shape[1]


    #----- initialize windows, arrays, etc -----
    w1  = sig.windows.hann(s_win, sym=False)   # analysis window
    w1  = np.tile(w1,nChan).reshape((nChan,len(w1))).T
    w2  = w1.copy()                            # synthesis window
    L   = DAFx_in.shape[0]

    # 0-pad & normalize
    DAFx_in = np.vstack((np.zeros((s_win,nChan)),DAFx_in,np.zeros((s_win-(L%n1),nChan))))/abs(DAFx_in).max()
    DAFx_out = np.zeros(DAFx_in.shape)

    hs_win   = s_win//2;

    t = TicToc()
    t.tic() #Start timer
    #UUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU
    pin  = 0;
    pout = 0;
    pend = DAFx_in.shape[0] - s_win;
    while pin<pend:
        grain = DAFx_in[pin:pin+s_win,:] * w1;
        
        #===========================================
        f     = np.fft.fft(grain,axis=0); # FFT
        r     = abs(f)/hs_win;
        if method=='sum':
            ft    = f * r / (r+coef);
        elif method=='max':
            ft    = f * r / np.where(r<coef,coef,r);
        else:
            raise TypeError('method must be equal to "sum" or "max" !!!')
            return
            
        grain = np.real(np.fft.ifft(ft,axis=0))*w2
        # ===========================================
        DAFx_out [pout:pout+s_win,:] = DAFx_out [pout:pout+s_win,:] + grain;
        pin  = pin + n1;
        pout = pout + n2;

    #%UUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU
    t.toc()

    #----- output -----
    #DAFx_in = DAFx_in[s_win:s_win+L,:];
    DAFx_out = DAFx_out[s_win:s_win+L,:] / abs(DAFx_out).max();
    if normOrigPeak: DAFx_out = DAFx_out * abs(x).max()

    #return DAFx_out according to original signal shape
    if x.ndim == 1:
        return DAFx_out[:,0]
    else:
        if x.shape[1] == DAFx_out.shape[1]:
            return DAFx_out
        else:
            return DAFx_out.T

if __name__=='__main__':
    import audiofile as af
    inputFile = 'redwheel.wav'

    x, fs = af.read(inputFile)
    auxName = inputFile.split('.wav')[0]

    y = VX_denoise(x,fs,method='max',normOrigPeak=False)
    af.write(auxName+'_denoise_max.wav',y,fs)
