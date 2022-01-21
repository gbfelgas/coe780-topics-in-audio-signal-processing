#VX_mutation
# VX_mutation.m  [DAFXbook, 2nd ed., chapter 7]
#===== this program performs a mutation between two sounds,
#===== taking the phase of the first one and the modulus 
#===== of the second one, and using:
#===== w1 and w2 windows (analysis and synthesis)
#===== WLen is the length of the windows
#===== n1 and n2: steps (in samples) for the analysis and synthesis
#
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
n1 = 512 # analysis step [samples]
n2 = n1 # synthesis step [samples]
WLen = 2048 # window size [samples]
w1 = s.hann(WLen,sym=False) # input window, analysis
w2 = w1 # synthesis
DAFx_in1, FS = sf.read("C:/Users/frota/Documents/mestrado/Top_esp_Audio/audios/x1.wav")
DAFx_in2, fs_2 = sf.read("C:/Users/frota/Documents/mestrado/Top_esp_Audio/audios/x2.wav")

#----- initialize windows, arrays, etc -----
L = np.min([len(DAFx_in1),len(DAFx_in2)])
DAFx_in1 = np.pad(DAFx_in1, (WLen, WLen - (L % n1)), 'constant') / np.max(np.abs(DAFx_in1))
DAFx_in2 = np.pad(DAFx_in2, (WLen, WLen - (L % n1)), 'constant') / np.max(np.abs(DAFx_in2))
DAFx_out = np.zeros(len(DAFx_in1)) # 0-pad & normalize

tic()
#UUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU
pin  = 0
pout = 0
pend = len(DAFx_in1) - WLen

while pin<pend:
  grain1 = DAFx_in1[pin:pin+WLen]*w1 
  grain2 = DAFx_in2[pin:pin+WLen]*w1
#===========================================
  f1 = np.fft.fft(np.fft.fftshift(grain1)) #FFT
  r1 = np.abs(f1) # magnitude
  theta1 = np.angle(f1) # phase

  f2 = np.fft.fft(np.fft.fftshift(grain2)) #FFT
  r2 = np.abs(f2) # magnitude
  theta2 = np.angle(f2) # phase
  #----- the next two lines can be changed according to the effect
  r = r1
  theta = theta2
  ft = (r * np.exp(1j * theta)) # reconstructed FFT
  grain = np.fft.fftshift(np.real(np.fft.ifft(ft))) * w2
# ===========================================
  DAFx_out[pout:pout+WLen] = DAFx_out[pout:pout+WLen] + grain
  pin  = pin + n1
  pout = pout + n2
#UUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU
toc()

#----- listening and saving the output -----
# DAFx_in  = DAFx_in[WLen:WLen+L];
DAFx_out = DAFx_out[WLen:WLen+L] / np.max(np.abs(DAFx_out))
sf.write("C:/Users/frota/Documents/mestrado/Top_esp_Audio/audios/test_mutation.wav", DAFx_out, FS)

#fig,ax = plt.subplots(figsize=(12,8))
#plt.plot(DAFx_in1, label='in1')
#plt.plot(DAFx_in2, label='in2')
#plt.plot(DAFx_out, label='out')
#plt.legend()
#plt.savefig("C:/Users/frota/Documents/mestrado/Top_esp_Audio/audios/test_mutation.jpg", dpi=600, bbox_inches='tight')
