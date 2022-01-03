# crosstalkcanceler.m
# Author: A. Politis, V. Pulkki
# Simplified cross-talk canceler 
#
#--------------------------------------------------------------------------
# This source code is provided without any warranties as published in 
# DAFX book 2nd edition, copyright Wiley & Sons 2011, available at 
# http://www.dafx.de. It may be used for educational purposes and not 
# for commercial applications without further permission.
#--------------------------------------------------------------------------
import numpy as np
from simplehrir import simpleHRIR
from simplehrir import simpleHRIRconv
from simplehrtfconv import *
import soundfile as sf
import librosa

theta=10   # spacing of stereo loudspeakers in azimuth 
Fs= 44100  # sample rate
b=10^-5    # regularization factor
# loudspeaker HRIRs for both ears (ear_num,loudspeaker_num)
# If more realistic HRIRs are available, pls use them
HRIRs = np.zeros((2,2,(simpleHRIR(theta/2,1000,Fs)).size))
HRIRs[0,0,:]=simpleHRIR(theta/2,1000,Fs)  
HRIRs[0,1,:]=simpleHRIR(-theta/2,1000,Fs)  
HRIRs[1,0,:]=HRIRs[0,1,:]
HRIRs[1,1,:]=HRIRs[0,0,:] 
Nh=HRIRs.shape[2]
C_f = np.zeros((2,2,(np.fft.rfft(HRIRs[0,0,:],2*Nh)).size))
C_f = C_f + 0*1j
#transfer to frequency domain
for i in range (2):
    for j in range (2):
        C_f[i,j,:] = np.fft.rfft(HRIRs[i,j,:],2*Nh)
     
# Regularized inversion of matrix C
H_f = np.zeros((2,2,Nh))
H_f = H_f + 0*1j
for k in range(Nh):
    H_f[:,:,k] = np.linalg.inv(np.conj(C_f[:,:,k]).T @ C_f[:,:,k] + np.identity(2)*b) @ np.conj(C_f[:,:,k]).T

H_n = np.zeros((2,2,Nh))
# Moving back to time domain
for k in range (2) :
    for m in range (2):
        H_n[k,m,:]=np.real(np.fft.ifft(H_f[k,m,:])) 
        H_n[k,m,:]=np.fft.fftshift(H_n[k,m,:]) 
      
# Generate binaural signals.  Any binaural recording shoud also be ok
simpleHRTFconv('doorbell.wav','binauralsignal.wav',70)
binauralsignal, Fs = librosa.load('binauralsignal.wav', Fs, mono=False)
#binauralsignal=wavread('road_binaural.wav') 
# np.convolveolve the loudspeaker signals
loudspsig1 = np.convolve(np.reshape(H_n[0,0,:],(Nh,)),binauralsignal[0,:],mode='full') + np.convolve(np.reshape(H_n[0,1,:],(Nh,)),binauralsignal[1,:],mode='full')
loudspsig2 = np.convolve(np.reshape(H_n[1,0,:],(Nh,)),binauralsignal[0,:],mode='full') + np.convolve(np.reshape(H_n[1,1,:],(Nh,)),binauralsignal[1,:],mode='full')

loudspsig = 100*np.array([loudspsig1, loudspsig2])

sf.write('resultado.wav',loudspsig.T,Fs)       # play sound for loudspeakersA