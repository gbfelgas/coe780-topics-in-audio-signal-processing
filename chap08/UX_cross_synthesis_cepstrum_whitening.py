# UX_cross_synthesis_cepstrum_whitening.py   [DAFXbook, 2nd ed., chapter 8]
# ==== This function performs a whitening and cross-synthesis with cepstrum
#
#--------------------------------------------------------------------------
# This source code is provided without any warranties as published in 
# DAFX book 2nd edition, copyright Wiley & Sons 2011, available at 
# http://www.dafx.de. It may be used for educational purposes and not 
# for commercial applications without further permission.
#--------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz
import soundfile as sf
import librosa.display
import librosa
import math
import time
import scipy.signal as s

#----- user data -----
# [DAFx_sou, SR] = wavread('didge_court.wav');  % sound 1: source/excitation
# DAFx_env       = wavread('la.wav');           % sound 2: spectral enveloppe
DAFx_sou, fs_1 = sf.read("C:/Users/frota/Documents/mestrado/Top_esp_Audio/audios/moore_guitar.wav") # sound 1: source/excitation
DAFx_env, fs_2 = sf.read("C:/Users/frota/Documents/mestrado/Top_esp_Audio/audios/Toms_diner.wav") # sound 2: spectral enveloppe        
s_win     = 1024   # window size
n1        = 256    # step increment
order_sou = 30     # cut quefrency for sound 1
order_env = 30     # cut quefrency for sound 2
r         = 0.99   # sound output normalizing ratio

#----- initialisations -----
w1 = s.hann(s_win,sym=False) # analysis window
w2 = w1                      # synthesis window
hs_win = s_win/2             # half window size
grain_sou = np.zeros(s_win)    # grain for extracting source
grain_env = np.zeros(s_win)    # grain for extracting spec. enveloppe
pin = 0                 # start index
L = np.min([len(DAFx_sou),len(DAFx_env)])
pend = L - s_win         # end index
DAFx_sou = np.pad(DAFx_sou, (s_win, s_win - (L % n1)), 'constant') / np.max(np.abs(DAFx_sou))
DAFx_env = np.pad(DAFx_env, (s_win, s_win - (L % n1)), 'constant') / np.max(np.abs(DAFx_env))
DAFx_out = np.zeros(L)

#----- cross synthesis -----
while pin<pend:
  grain_sou = DAFx_sou[pin:pin+s_win]*w1 
  grain_env = DAFx_env[pin:pin+s_win]*w1
#===========================================
  f_sou = np.fft.fft(grain_sou) # FFT of source
  f_env = np.fft.fft(grain_env)/hs_win # FFT of filter

#---- computing cepstrum ----
  flog_sou = np.log(0.00001+np.abs(f_sou))
  cep_sou = np.fft.ifft(flog_sou)  # cepstrum of sound 1 / source
  flog_env = np.log(0.00001+np.abs(f_env))
  cep_env = np.fft.ifft(flog_env)  # cepstrum of sound 2 / env.

#---- liftering cepstrum ----
  cep_cut_env = np.zeros(s_win)
  cep_cut_env[0:order_env] = cep_env[0:order_env]
  cep_cut_env[0] = cep_cut_env[0]/2
  flog_cut_env = 2*np.real(np.fft.fft(cep_cut_env))
  cep_cut_sou = np.zeros(s_win)
  cep_cut_sou[0:order_sou] = cep_sou[0:order_sou]
  cep_cut_sou[0] = cep_cut_sou[0]/2
  flog_cut_sou = 2*np.real(np.fft.fft(cep_cut_sou))

#---- computing spectral enveloppe ----
  f_env_out = np.exp(flog_cut_env - flog_cut_sou) # whitening with source
  grain = (np.real(np.fft.ifft(f_sou*f_env_out)))*w2 # resynthesis grain
# ===========================================
  DAFx_out[pin:pin+s_win] = DAFx_out[pin:pin+s_win] + grain
  pin  = pin + n1

#----- listening and saving the output -----
# DAFx_in  = DAFx_in[WLen:WLen+L];
DAFx_out = DAFx_out[s_win:len(DAFx_out)] / np.max(np.abs(DAFx_out))
DAFx_out_norm = r * DAFx_out/np.max(np.abs(DAFx_out)) # scale for wav output
sf.write("C:/Users/frota/Documents/mestrado/Top_esp_Audio/audios/test_UX_cross_syn_cepstrum_whit.wav", DAFx_out_norm, fs_1)
#sf.write("C:/Users/frota/Documents/mestrado/Top_esp_Audio/audios/test_mutation_1.wav", DAFx_out, FS)

fig,ax = plt.subplots(figsize=(12,8))
plt.plot(DAFx_sou, label='sou')
plt.plot(DAFx_env, label='env')
plt.plot(DAFx_out, label='out')
plt.legend()
plt.savefig("C:/Users/frota/Documents/mestrado/Top_esp_Audio/figures/test_UX_whit.jpg", dpi=600, bbox_inches='tight')
  
