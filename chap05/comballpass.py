####### COMBALLPASS CHAPTER 5 #########

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz
import soundfile as sf
import librosa.display
import librosa
import math

# Input signal: impulse
length_x = 2500
x = np.zeros((length_x,1), dtype=float)
x[0] = 1

# Delay line and read position
length_A = 100
A = np.zeros((length_A,1), dtype=float)
Adelay = 20
tmp = 0

# Output vector
ir = np.zeros((length_x,1), dtype=float)

# Feedback gain
g = 0.7

# Comb-allpass filtering
for n in range(0,(length_x)):
  tmp = A[Adelay-1] + x[n]*(-g)
  Aux = tmp*g + x[n]
  A = np.append(Aux, A[0:(length_A-1)])
  ir[n] = tmp
  
plt.rcParams.update({'font.size': 6})
# Plot the filtering results
plt.subplot(2,2,1)
plt.stem(range(length_x),x)
plt.xlabel(r"$n$ $\rightarrow$" )
plt.ylabel(r"$x(n) \rightarrow$")
plt.axis([-100, length_x, 0, x[0]*1.1])
plt.grid(True, color = '0.7', linestyle='-', which='major', axis='both')
plt.grid(True, color = '0.9', linestyle='-', which='minor', axis='both')
plt.title("All pass filter input signal")
plt.tight_layout()

plt.subplot(2,2,3)
plt.stem(range(length_x),ir)
plt.xlabel(r"$n$ $\rightarrow$" )
plt.ylabel(r"$y(n) \rightarrow$")
plt.axis([-100, length_x, ir[0]*1.3, max(ir)*1.3])
plt.grid(True, color = '0.7', linestyle='-', which='major', axis='both')
plt.grid(True, color = '0.9', linestyle='-', which='minor', axis='both')
plt.title("All pass filter output signal")
plt.tight_layout()

plt.rcParams.update({'font.size': 6})
plt.subplot(2,2,2)
W, H = freqz(ir,1)
plt.plot(W/math.pi, 20*np.log10(abs(H)))
plt.xlabel(r"$f$ x $\pi$ in rad/sample $\rightarrow$" )
plt.ylabel(r"$Magnitude (dB) \rightarrow$")
plt.title("All pass filter Magnitude response")
plt.grid(True, color = '0.7', linestyle='-', which='major', axis='both')
plt.grid(True, color = '0.9', linestyle='-', which='minor', axis='both')
plt.tight_layout()

plt.subplot(2,2,4)
plt.plot(W/math.pi, np.unwrap(np.angle(H))*180/math.pi)
plt.xlabel(r"$f$ x $\pi$ in rad/sample $\rightarrow$")
plt.ylabel(r"$Phase (degrees) \rightarrow$",)
# plt.axis([-0.1, 1.1, -8000, 0])
plt.title(r"All pass filter Phase Response")
plt.grid(True, color = '0.7', linestyle='-', which='major', axis='both')
plt.grid(True, color = '0.9', linestyle='-', which='minor', axis='both')
plt.tight_layout()
    
plt.savefig("C:/Users/frota/Documents/mestrado/Top_esp_Audio/figures/comballpass.jpg", dpi=600, bbox_inches='tight')
