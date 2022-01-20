# VX_bank_nothing.m   [DAFXbook, 2nd ed., chapter 7]

import scipy.signal as sig
import numpy as np

def princarg(phase_in):
    '''
    % This function puts an arbitrary phase value into ]-pi,pi] [rad]
    %
    %--------------------------------------------------------------------------
    % This source code is provided without any warranties as published in 
    % DAFX book 2nd edition, copyright Wiley & Sons 2011, available at 
    % http://www.dafx.de. It may be used for educational purposes and not 
    % for commercial applications without further permission.
    %--------------------------------------------------------------------------
    '''
    phase = (phase_in+np.pi)%(-2*np.pi) + np.pi
    return phase

def VX_bank_nothing(x, n1=200, n2=200, s_win=2048, normOrigPeak = False):
    '''
    This program performs an FFT analysis and oscillator bank synthesis
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
    w1    = sig.windows.hann(s_win, sym=False) # input window
    w1    = np.tile(w1,nChan).reshape((nChan,len(w1))).T
    w2    = w1.copy()    # output window
    L     = DAFx_in.shape[0]

    # 0-pad & normalize
    DAFx_in = np.vstack((np.zeros((s_win,nChan)),DAFx_in,np.zeros((s_win-(L%n1),nChan))))/abs(DAFx_in).max()

    DAFx_out = np.zeros(DAFx_in.shape)
    ll    = s_win//2;
    omegaRa = 2*np.pi*n1*np.arange(ll)/s_win; #all frequencies of FFT [0,pi[ multiplied by R_a
    omegaRa = np.tile(omegaRa,nChan).reshape((nChan,len(omegaRa))).T
    phi0  = np.zeros((ll,nChan));
    r0    = np.zeros((ll,nChan));
    psi   = np.zeros((ll,nChan));
    grain = np.zeros((s_win,nChan));
    res   = np.zeros((n2,nChan));

    #UUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU
    pin  = 0;
    pout = 0;
    pend = DAFx_in.shape[0] - s_win;

    while pin<pend:
        grain = DAFx_in[pin:pin+s_win,:] * w1;
        #===========================================
        fc  = np.fft.fft(np.fft.fftshift(grain,axes=0),axis=0); # FFT
        f   = fc[:ll,:]            # positive frequency spectrum
        r   = abs(f);             # magnitudes
        phi = np.angle(f);        # phases
        #----- unwrapped phase difference on each bin for a n2 step
        delta_phi = omegaRa + princarg(phi-phi0-omegaRa);
        #----- phase and magnitude increment, for linear
        #      interpolation and reconstruction -----
        delta_r   = (r-r0)/n1;    # magnitude increment
        delta_psi = delta_phi/n1; # phase increment
        for k in range(n2): # compute the sum of weighted cosine
            r0     = r0 + delta_r;
            psi    = psi + delta_psi;
            for ch in range(nChan):
                res[k,ch] = (r0[:,ch]*np.cos(psi[:,ch])).sum()
      
        #----- for next time -----
        phi0 = phi;
        r0   = r;
        psi  = princarg(psi);
        # ===========================================
        DAFx_out[pout:pout+n2,:] = DAFx_out[pout:pout+n2,:].copy() + res;
        pin  = pin + n1;
        pout = pout + n2;

    #UUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU

    #-----  output -----
    # DAFx_in = DAFx_in[s_win:s_win+L,:];
    DAFx_out = DAFx_out[s_win//2+n1:s_win//2+n1+L,:] / abs(DAFx_out).max();
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
    
    inputFile = 'la.wav'

    stdName = inputFile.split('.wav')[0]
    x, fs = af.read(inputFile)
    t = TicToc()
    t.tic() #Start timer
    y = VX_bank_nothing(x, normOrigPeak=True)
    t.toc()
    af.write(stdName+'_bank_nothing.wav',y, fs)

    '''
    X = np.fft.fft(x)
    kx = np.arange(len(x))/len(x)*2
    Y = np.fft.fft(y)
    ky = np.arange(len(y))/len(y)*2
    '''

    win = sig.get_window('hann', 1024, fftbins=True)
    nSpec = 12000
    wCut = 500
    
    fa, ta, Sxxa = sig.spectrogram(x[:nSpec], window=win, noverlap=len(win)-1, mode='angle')
    f, t, Sxx = sig.spectrogram(x[:nSpec], window=win, scaling='spectrum', noverlap=len(win)-1)
    yfa, yta, Syya = sig.spectrogram(y[:nSpec], window=win, noverlap=len(win)-1, mode='angle')
    yf, yt, Syy = sig.spectrogram(y[:nSpec], window=win,scaling='spectrum', noverlap=len(win)-1)

    
    plt.figure(figsize=(14,6))
    plt.subplot(231)
    plt.plot(x)
    plt.title('Signal')
    plt.xlabel(r'n $\rightarrow$')
    plt.ylabel('Original signal\nx(n) '+r'$\rightarrow$')
    
    plt.subplot(232)
    plt.pcolormesh(t, f*fs/2, Sxx, shading='gouraud',cmap='gray')
    plt.title('Spectrogram')
    plt.ylabel('Frequency [Hz]'+r' $\rightarrow$')
    plt.xlabel(r'n $\rightarrow$')
    plt.axis(ymax=wCut)
    plt.colorbar()

    plt.subplot(233)
    plt.pcolormesh(ta, fa*fs/2, Sxxa, shading='gouraud',cmap='gray')
    plt.title('Phasogram')
    plt.ylabel('Frequency [Hz]'+r' $\rightarrow$')
    plt.xlabel(r'n $\rightarrow$')
    plt.axis(ymax=wCut)
    plt.colorbar()

    plt.subplot(234)
    plt.plot(y)
    plt.xlabel(r'n $\rightarrow$')
    plt.ylabel('Reconstructed signal\ny(n) '+r'$\rightarrow$')
    
    plt.subplot(235)
    plt.pcolormesh(yt, yf*fs/2, Syy, shading='gouraud', cmap='gray')
    plt.ylabel('Frequency [Hz]'+r' $\rightarrow$')
    plt.xlabel(r'n $\rightarrow$')
    plt.axis(ymax=wCut)
    plt.colorbar()

    plt.subplot(236)
    plt.pcolormesh(yta, yfa*fs/2, Syya, shading='gouraud',cmap='gray')
    plt.ylabel('Frequency [Hz]'+r' $\rightarrow$')
    plt.xlabel(r'n $\rightarrow$')
    plt.axis(ymax=wCut)
    plt.colorbar()

    plt.tight_layout()
    plt.show()

    
    
    
