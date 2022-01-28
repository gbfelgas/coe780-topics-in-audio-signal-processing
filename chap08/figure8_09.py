from calc_lpc import calc_lpc
import numpy as np
import audiofile as af
import matplotlib.pyplot as plt
import scipy.signal as sig

'''
% ===== This function plots the LPC for one block
%       calculation of prediction error and spectra       
% [DAFXbook, 2nd ed., chapter 8]
'''

fname = 'la.wav';
n0   = 5000;   # start index
N    = 1024;   # block length
Nfft = 1024;   # FFT length
p    = 50;     # prediction order
n1   = n0+N-1; # end index
pre  = p;      # filter order= no. of samples required before n0

xin,Fs = af.read(fname);
xin = xin[(n0-pre-1):(n0+N-1)]
win = np.hamming(N)
x   = xin[pre:N+pre].copy(); # block without pre-samples
xw = x*win                   # windowed x
a,g = calc_lpc(xw,p);        # calculate LPC coeffs and gain  a = [1, -a_1, -a_2,..., -a_p]
g_db = 20*np.log10(g)        # gain in dB

ein = sig.lfilter(a,1,xin)                    # pred. error (E(z) = X(z)A(z))
e   = ein[pre:pre+N].copy();                 # without pre-samples
#Gp  = 10*np.log10((x**2).sum()/(e**2).sum()) # prediction gain

Omega  = np.arange(Nfft)/Nfft*Fs/1000;       # frequencies in kHz
offset = 20*np.log10(2/Nfft);                # offset of spectrum in dB
A      = 20*np.log10(abs(np.fft.fft(a,Nfft)));  
H_g    = g_db - A + offset ;                 # spectral envelope Hg(z) = G / A (z)
X      = 20*np.log10(abs(np.fft.fft(xw,Nfft)));
X      = X+offset;

n   =  np.arange(N)
plt.figure(1, figsize=(14,5))
plt.clf()
plt.subplot(121)
plt.plot(n,e,lw=0.5)
plt.title('Time signal of pred. error e(n)')
plt.xlabel(r'n $\rightarrow$')
plt.axis(xmin=0, xmax=N-1)

plt.subplot(122)
plt.plot(Omega,X,lw=0.5);
plt.plot(Omega,H_g,'r');
plt.title(r'Magnitude spectra |X(f)| and |G$\cdot$H(f)| in dB')
plt.xlabel(r'f [kHz] $\rightarrow$')
plt.axis(xmin=0, xmax=8)

plt.show()
