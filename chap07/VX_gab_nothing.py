#VX_gab_nothing
#This program performs signal convolution with gaborets
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz
import soundfile as sf
import librosa.display
import librosa
import math
import time
import scipy.signal as s

def TicTocGenerator():
    # Generator that returns time differences
    ti = 0           # initial time
    tf = time.time() # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti # returns the time difference

TicToc = TicTocGenerator() # create an instance of the TicTocGen generator

# This will be the main function through which we define both tic() and toc()
def toc(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print( "Elapsed time: %f seconds.\n" %tempTimeInterval )

def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)


#----- user data -----
n1 = 128  # analysis step [samples]
n2 = n1   # synthesis step [samples]
s_win = 512  # window size [samples]
DAFx_in, FS = sf.read("C:/Users/frota/Documents/mestrado/Top_esp_Audio/audios/la.wav")

#----- initialize windows, arrays, etc -----
window = s.hann(s_win,sym=False) # input window 
nChannel = s_win/2
L = len(DAFx_in)
DAFx_in = np.pad(DAFx_in, (s_win, s_win - (L % n1)), 'constant') / np.max(np.abs(DAFx_in))
DAFx_out = np.zeros(len(DAFx_in)) # 0-pad & normalize

#----- initialize calculation of gaborets -----
t = np.arange(s_win) - s_win/2
gab  = np.zeros((int(nChannel),s_win),dtype='complex128')
for k in range(1,int(nChannel)+1):
  wk       = 2j*math.pi*(k/s_win)
  gab[(k-1),:] = window*np.exp(wk*t)

tic()
#UUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU
pin  = 0
pout = 0
pend = len(DAFx_in) - s_win

while pin<pend:
  grain = DAFx_in[pin:pin+s_win]
  
#===========================================
  #----- complex vector corresponding to a vertical line    
  vec = np.matmul(gab,grain)
  #----- reconstruction from the vector to a grain
  res = np.real(np.conj(gab.T).dot(vec))
  
# ===========================================
  DAFx_out[pout:pout+s_win] = DAFx_out[pout:pout+s_win] + res
  pin  = pin + n1
  pout = pout + n2
#UUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU
toc()

#----- listening and saving the output -----
# DAFx_in  = DAFx_in(s_win+1:s_win+L);
DAFx_out = DAFx_out[s_win:s_win+L] / np.max(np.abs(DAFx_out))
sf.write("C:/Users/frota/Documents/mestrado/Top_esp_Audio/audios/test_VX_gab_nothing.wav", DAFx_out, FS)

fig,ax = plt.subplots(figsize=(12,8))
plt.plot(DAFx_in, label='in')
plt.plot(DAFx_out, label='out')
plt.legend()
plt.savefig("C:/Users/frota/Documents/mestrado/Top_esp_Audio/audios/test_VX_gab_nothing.jpg", dpi=600, bbox_inches='tight')
