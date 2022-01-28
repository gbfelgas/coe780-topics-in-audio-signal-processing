# VX_tstretch_real_pv.m   [DAFXbook, 2nd ed., chapter 7]

import scipy.signal as sig
import numpy as np

def princarg(phase_in):
    '''
    This function puts an arbitrary phase value into ]-pi,pi] [rad]
    -----------------------------------------------
    '''
    phase = (phase_in+np.pi)%(-2*np.pi) + np.pi
    return phase

def VX_tstretch_real_pv(x, n1=200, n2=512, s_win=2048, normOrigPeak = False):
    '''
    This program performs time stretching using the oscillator bank approach
    ----- user data -----
    n1            analysis step [samples]
    n2            synthesis step [samples]
    s_win         window size [samples]
    normOrigPeak  normalize according original signal max peak
    ---- output-----
    y             signal as the sum of weighted cosine
    '''
    #Adapting x shape to (sample, channel)
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
    tstretch_ratio = n2/n1
    w1    = sig.windows.hann(s_win, sym=False) # input window
    w1    = np.tile(w1,nChan).reshape((nChan,len(w1))).T
    w2    = w1.copy()    # output window
    L     = DAFx_in.shape[0]

    # 0-pad & normalize
    DAFx_in = np.vstack((np.zeros((s_win,nChan)),DAFx_in,np.zeros((s_win-(L%n1),nChan))))/abs(DAFx_in).max()
    DAFx_out = np.zeros((s_win+int(np.ceil(DAFx_in.shape[0]*tstretch_ratio)),nChan))

    omega = 2*np.pi*n1*np.arange(s_win)/s_win
    omega = np.tile(omega,nChan).reshape((nChan,len(omega))).T
    phi0  = np.zeros((s_win,nChan))
    psi   = np.zeros((s_win,nChan))

    #UUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU
    pin  = 0
    pout = 0
    pend = DAFx_in.shape[0] - s_win

    while pin < pend:

        grain = DAFx_in[pin:pin+s_win,:] * w1

        #===========================================
        f = np.fft.fft(np.fft.fftshift(grain,axes=0),axis=0) # FFT
        r = np.abs(f) # magnitude
        phi = np.angle(f) # phase

        #---- computing input phase increment ----
        delta_phi = omega + princarg(phi-phi0-omega)
        #---- computing output phase increment ----
        psi   = princarg(psi+delta_phi*tstretch_ratio)
        #---- comouting synthesis Fourier transform & grain ----
        ft = (r * np.exp(1j * psi)) # reconstructed FFT
        grain = np.fft.fftshift(np.real(np.fft.ifft(ft,axis=0)),axes=0) * w2

        # ===========================================
        DAFx_out[pout:pout+s_win,:] = DAFx_out[pout:pout+s_win,:] + grain
        
       #----- for next block -----
        phi0 = phi
        pin  = pin + n1
        pout = pout + n2

    #UUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU

    #-----  output -----
    # DAFx_in = DAFx_in[s_win:s_win+L,:];
    DAFx_out = DAFx_out[s_win:DAFx_out.shape[0],:] / abs(DAFx_out).max()
    if normOrigPeak: DAFx_out = DAFx_out * abs(x).max()
        
    #return DAFx_out according to original signal shape
    if x.ndim == 1:
        return DAFx_out[:,0]
    else:
        if x.shape[1] == DAFx_out.shape[1]:
            return DAFx_out
        else:
            return DAFx_out.T

#Test
if __name__=='__main__':
    import matplotlib.pyplot as plt
    import audiofile as af
    from pytictoc import TicToc

    inputFiles = ['la.wav'] # Input file names in an array
    
    for i in range(len(inputFiles)): # Perform stretching on all files 
        inputFile = inputFiles[i]

        stdName = inputFile.split('.wav')[0]
        x, fs = af.read(inputFile)
        w1 = 256
        w2 = 200
        t = TicToc()
        t.tic() #Start timer
        y = VX_tstretch_real_pv(x,w1,w2, normOrigPeak=True)
        t.toc()
        af.write(stdName+'_tstretch_real_pv'+'_x'+ str(w2/w1) +'.wav',y, fs)
