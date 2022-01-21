# VX_whisper.m   [DAFXbook, 2nd ed., chapter 7]
import numpy as np
import scipy.signal as sig
from pytictoc import TicToc

def VX_whisper(x, s_win=512, n1=64, n2=64, randomize='phase',normOrigPeak = False):
    
    '''
    %===== This program makes the whisperization of a sound, 
    %===== by randomizing the phase (or the amplitude)

    INPUTS
    ---------------------
    x             signal
    s_win         analysis window length [samples]
    n1            analysis step [samples] (s_win/8)
    n2            synthesis step [samples]    
    randomize     {'phase','magnitude'} where the effect will be applied
    normOrigPeak  normalize according original signal max peak

    OUTPUT
    --------------------
    y             signal with whisperization
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
    t = TicToc()
    t.tic() #Start timer
    #UUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU
    pin  = 0;
    pout = 0;
    pend = DAFx_in.shape[0] - s_win;
    while pin<pend:
        grain = DAFx_in[pin:pin+s_win,:] * w1;
        if randomize=='phase':
            #===========================================
            f     = np.fft.fft(np.fft.fftshift(grain,axes=0),axis=0); # FFT
            r     = abs(f);
            phi   = 2*np.pi*np.random.uniform(0,1,(s_win,nChan))
            ft    = (r * np.exp(1j*phi));
            grain = np.fft.fftshift(np.real(np.fft.ifft(ft,axis=0)),axes=0)*w2
            #===========================================

        elif randomize=='magnitude':
            #===========================================
            f     = np.fft.fft(np.fft.fftshift(grain,axes=0),axis=0); # FFT
            r     = abs(f) * np.random.normal(0,1,(s_win,nChan))
            phi   = np.angle(f)
            ft    = (r * np.exp(1j*phi));
            grain = np.fft.fftshift(np.real(np.fft.ifft(ft,axis=0)),axes=0)*w2
            #===========================================
            
        else:
            raise TypeError('randominze must be equal to "phase" or "magnitude" !!!')
            return
        DAFx_out [pout:pout+s_win,:] = DAFx_out [pout:pout+s_win,:] + grain;
        pin   = pin + n1;
        pout  = pout + n2;
    #UUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU
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
