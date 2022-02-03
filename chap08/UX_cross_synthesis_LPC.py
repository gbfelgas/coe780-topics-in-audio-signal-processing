 # UX_cross_synthesis_LPC.m   [DAFXbook, 2nd ed., chapter 8]
 # ==== This function performs a cross-synthesis with LPC
 #
 #--------------------------------------------------------------------------
 # This source code is provided without any warranties as published in 
 # DAFX book 2nd edition, copyright Wiley & Sons 2011, available at 
 # http://www.dafx.de. It may be used for educational purposes and not 
 # for commercial applications without further permission.
 #--------------------------------------------------------------------------
import librosa
import soundfile as sf
import numpy as np
from scipy import signal
import math 
from calc_lpc import *

 #----- user data -----
DAFx_in_sou, FS = librosa.load('moore_guitar.wav')   # sound 1: source/excitation
DAFx_in_env, FS   = librosa.load('toms_diner.wav')         # sound 2: spectral env.
long          = 400         # block len for calculation of coefficients
hopsize       = 160         # hop size (is 160)
env_order     = 20           # order of the LPC for source signal
source_order  = 6            # order of the LPC for excitation signal
r             = 0.99        # sound output normalizing ratio

 #----- initializations -----
ly = min(len(DAFx_in_sou), len(DAFx_in_env))
#DAFx_in_sou = [np.zeros(env_order) DAFx_in_sou , np.zeros(env_order-mod(ly,hopsize))] / max(abs(DAFx_in_sou))
DAFx_in_sou = np.append(np.zeros(env_order), DAFx_in_sou)
if env_order - (ly % hopsize)> 0:
    DAFx_in_sou = np.append(DAFx_in_sou, np.zeros(env_order - (ly % hopsize)))
DAFx_in_sou = DAFx_in_sou / max(abs(DAFx_in_sou))
#DAFx_in_env = [np.zeros(env_order) DAFx_in_env , np.zeros(env_order-mod(ly,hopsize))] / max(abs(DAFx_in_env))
DAFx_in_env = np.append(np.zeros(env_order), DAFx_in_env)
if env_order - (ly % hopsize)> 0:
    DAFx_in_env = np.append(DAFx_in_env, np.zeros(env_order- (ly % hopsize)))
DAFx_in_env = DAFx_in_env / max(abs(DAFx_in_env))

DAFx_out = np.zeros(ly)      # result sound
exc      = np.zeros(ly)      # excitation sound
w        = np.hanning(long)    # window
N_frames = math.floor((ly-env_order-long)/hopsize)  # number of frames

 #----- Perform ross-synthesis -----
#tic
gain = np.zeros(N_frames)

for j in range(N_frames):
  k = env_order + hopsize*(j)      # offset of the buffer
   #!!! IMPORTANT: function "lpc" does not give correct results for MATLAB 6 !!!
  A_env, g_env  = calc_lpc(DAFx_in_env[k :k+long]*w, env_order)
  A_sou, g_sou  = calc_lpc(DAFx_in_sou[k: k+long]*w, source_order)
  gain[j] = g_env
  ae      = - A_env[0:env_order]  # LPC coeff. of excitation
  for n in range(hopsize):
    excitation1   = (A_sou/g_sou) @ DAFx_in_sou[k+n+1:k+n-source_order:-1]
    exc[k+n-1]      = np.array(excitation1)
    DAFx_out[k+n-1] = ae @ DAFx_in_sou[k+n:k+n-env_order:-1] + g_env*excitation1
#toc

 #----- playing and saving output signal -----
DAFx_out      = DAFx_out[env_order:len(DAFx_out)-1] / max(abs(DAFx_out))
#soundsc(DAFx_out, FS)
DAFx_out_norm = r * DAFx_out/max(abs(DAFx_out))  # scale for wav output
sf.write('CrossLPC.wav', DAFx_out_norm, FS)