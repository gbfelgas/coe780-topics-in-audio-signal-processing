import numpy as np
import soundfile as sf

# kappa.m
# Author: V. Pulkki, T. Lokki
# Simple example of cardioid decoding of B-format signals
#
#--------------------------------------------------------------------------
# This source code is provided without any warranties as published in 
# DAFX book 2nd edition, copyright Wiley & Sons 2011, available at 
# http://www.dafx.de. It may be used for educational purposes and not 
# for commercial applications without further permission.
#--------------------------------------------------------------------------

Fs = 44100
pi = np.pi
# mono signal
signal = np.linspace(0,Fs*2,Fs*2 + 1)
signal = (signal % 220)/220
# Simulated horizontal-only B-format recording of single 
# sound source in direction of theta azimuth.
# This can be replaced with a real B-format recording. 
theta=0
w = signal/(2**(1/2))
x = signal*np.cos(theta/180*pi)
y = signal*np.sin(theta/180*pi)
# Virtual microphone directions 
# (In many cases the values equal to the directions of loudspeakers)
ls_dir = np.array([30, 90, 150, -150, -90, -30])
ls_dir = ls_dir/180
ls_dir = ls_dir*pi
ls_num = ls_dir.size
# Compute virtual cardioids (kappa = 1) out of the B-format signal
kappa = 1
LSsignal = np.zeros((signal.size, ls_num))
for i in range(ls_num):
    LSsignal[:,i] = (2-kappa)/2*w + kappa/(2*(2**(1/2)))*(np.cos(ls_dir[i])*x+np.sin(ls_dir[i])*y)
    sf.write('firstorderB-formatexample.wav',LSsignal[:,i],Fs,'PCM_16')

# File output
sf.write('firstorderB-formatexample.wav',LSsignal,Fs,'PCM_16')