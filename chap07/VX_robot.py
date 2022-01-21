# VX_robot.m   [DAFXbook, 2nd ed., chapter 7]

import numpy as np
import scipy.signal as sig
from pytictoc import TicToc

def VX_robot(x, s_win=1024, n1=441, fs=None, robotFreq=None, normOrigPeak = False):
    '''

    #===== this program performs a robotization of a sound

    INPUTS
    ---------------------
    x             signal
    s_win         analysis window length [samples]
    n1            analysis step [samples]
    fs            sampling frequency (necessary only if robotFreq is informed)
    robotFreq     robot frequency [Hertz]
                  If None robotFreq will be fs/n1, otherwise n1 and n2 are ignored
    normOrigPeak  normalize according original signal max peak

    OUTPUT
    --------------------
    y             robotic signal
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

    if not(robotFreq is None):
        n1 = round(fs/robotFreq)
    n2 = n1                                    # synthesis step [samples]  ( = n1)  

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
        #===========================================
        f     = np.fft.fft(grain,axis=0); # FFT
        r     = abs(f)
        grain = np.real(np.fft.ifft(r,axis=0))*w2
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

    y = VX_robot(x,fs=fs,robotFreq=70,normOrigPeak=False)
    af.write(auxName+'_robot.wav',y,fs)

