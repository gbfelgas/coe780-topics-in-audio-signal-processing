# UX_fmove_cepstrum.m   [DAFXbook, 2nd ed., chapter 8]
# ==== This function performs a formant warping with cepstrum
#
#-------------------------------------------------------------------------
# This source code is provided without any warranties as published in 
# DAFX book 2nd edition, copyright Wiley & Sons 2011, available at 
# http://www.dafx.de. It may be used for educational purposes and not 
# for commercial applications without further permission.
#--------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import IPython.display as ipd
import soundfile as sf
import librosa

def UX_fmove_cepstrum(input_file, warping_coef, n1, n2, s_win, order, r, fig_plot = False, display = False):
    
    DAFx_in, SR = librosa.load(input_file)

    w1       = np.hanning(s_win)                            # analysis window
    w2       = w1                                           # synthesis window
    hs_win   = s_win/2                                      # half window size
    L        = len(DAFx_in)                                 # signal length
    DAFx_in  = np.pad(DAFx_in, (s_win, s_win - (L % n1)), 
                      'constant', constant_values=(0, 0))   # 0-pad
    DAFx_in  = DAFx_in / np.max(np.abs(DAFx_in))            # normalize
    DAFx_out = np.zeros(L)                                  # output signal
    t        = np.floor(np.arange(s_win)*warping_coef)  # apply the warping
    t = t.astype(int)
    #lmax     = max(s_win,t[s_win])

    #UUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU
    pin  = 0
    pout = 0
    pend = L - s_win

    while pin < pend:
        grain   = DAFx_in[pin:pin + s_win]*w1
    # ===========================================
        f = np.fft.fft(grain) / hs_win                   # spectrum of grain
        flogs   = 20*np.log10(0.00001 + np.abs(f))       # log|X(k)|

        grain1  = DAFx_in[pin+t] * w1                    # linear interpolation of grain
        f1      = np.fft.fft(grain1)/hs_win              # spectrum of interpolated grain
        flogs1  = 20*np.log10(0.00001 + np.abs(f1))      # log|X1(k)|
        flog    = np.log(0.00001+abs(f1)) - np.log(0.00001+abs(f))
        cep     = np.fft.ifft(flog)                      # cepstrum
        cep_cut = np.append(cep[0]/2, cep[1:order])
        cep_cut = np.append(cep_cut, np.zeros(s_win-order))

        aux = np.fft.fft(cep_cut)
        corr    = np.exp(2*np.real(aux)) # spectral shape
        grain   = (np.real(np.fft.ifft(f*corr)))*w2

        fout    = np.fft.fft(grain)
        flogs2  = 20*np.log10(0.00001+np.abs(fout))

        if fig_plot and pin == 13*n1:
            plt.figure()

            range = np.arange(hs_win/2, dtype=int)

            plt.subplot(311)
            plt.plot(range*SR/s_win, flogs[range])
            plt.title('a) original spectrum')
            plt.draw()

            plt.subplot(312)
            plt.plot(range*SR/s_win, flogs1[range])
            plt.title('b) spectrum of time-scaled signal')

            plt.subplot(313)
            plt.plot(range*SR/s_win, flogs2[range])
            plt.title('c) formant changed spectrum')
            plt.xlabel(r'f in Hz $\rightarrow$')
            plt.draw()
            plt.tight_layout()
    # ===========================================
        DAFx_out[pout:pout + s_win] = DAFx_out[pout:pout + s_win] + grain
        pin  = pin + n1
        pout = pout + n2
    #UUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU

    #----- listening and saving the output -----
    if display:
        print('original sound: ')
        ipd.display(ipd.Audio(data=DAFx_in, rate=SR))
        print('format changed: ')
        ipd.display(ipd.Audio(data=DAFx_out, rate=SR))
    DAFx_out_norm = r * DAFx_out/np.max(np.abs(DAFx_out)) # scale for wav output
    sf.write('fmove_'+input_file, DAFx_out_norm, SR)

if __name__ == '__main__':
    input_file   = 'la.wav'
    warping_coef = 2.0
    n1    = 512
    n2    = n1
    s_win = 2048
    order = 50
    r     = 0.99
    UX_fmove_cepstrum(input_file, warping_coef, n1, n2, s_win, order, r)