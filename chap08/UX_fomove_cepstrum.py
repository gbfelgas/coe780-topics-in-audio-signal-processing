# UX_fomove_cepstrum.m   [DAFXbook, 2nd ed., chapter 8]
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

def UX_fomove_cepstrum(input_file, warping_coef, n1, n2, s_win, order, r, display = False):
    
    DAFx_in, SR = librosa.load(input_file)

    w1       = np.hanning(s_win)                                            # analysis window
    w2       = w1                                                           # synthesis window
    hs_win   = s_win//2                                                     # half window size
    L        = len(DAFx_in)                                                 # signal length
    DAFx_in  = np.pad(DAFx_in, (s_win, s_win - (L % n1)), 
                      'constant', constant_values=(0, 0))                   # 0-pad
    DAFx_in  = DAFx_in / np.max(np.abs(DAFx_in))                            # normalize
    DAFx_out = np.zeros(L)                                                  # output signal
    x0 = np.floor(np.minimum(1 + np.arange(hs_win+1)/warping_coef, 1 + hs_win)) # apply the warping
    x = np.pad(x0, (0, hs_win - 1), mode='reflect')                                      # symmetric extension
    x = x.astype(int)

    pin  = 0
    pout = 0
    pend = L - s_win

    while pin < pend:
        grain   = DAFx_in[pin:pin + s_win]*w1
    # ===========================================
        f       = np.fft.fft(grain) / hs_win
        flog    = np.log(0.00001 + np.abs(f))
        cep     = np.fft.ifft(flog)
        cep_cut = np.append(cep[0]/2, cep[1:order])
        cep_cut = np.append(cep_cut, np.zeros(s_win-order))
        # ---- flog_cut1|2 =  spectral shapes before/after formant move
        flog_cut1 = 2*np.real(np.fft.fft(cep_cut))
        flog_cut2 = flog_cut1[x]
        corr      = np.exp(flog_cut2 - flog_cut1)
        grain     = (np.real(np.fft.ifft(f*corr)))*w2
    # ===========================================
        DAFx_out[pout:pout + s_win] = DAFx_out[pout:pout + s_win] + grain
        pin  = pin + n1
        pout = pout + n2

    #----- listening and saving the output -----
    if display:
        print('original sound: ')
        ipd.display(ipd.Audio(data=DAFx_in, rate=SR))
        print('format changed: ')
        ipd.display(ipd.Audio(data=DAFx_out, rate=SR))
    DAFx_out_norm = r * DAFx_out/np.max(np.abs(DAFx_out)) # scale for wav output
    sf.write('fomove_'+input_file, DAFx_out_norm, SR)