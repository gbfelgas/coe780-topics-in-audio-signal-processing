# Sample by sample IIR Comb

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz
import math

# Creating an unit impulse signal
length_x = 100
x = np.zeros((length_x,1), dtype=float)
x[0] = 1

# Input for user: g and c parameters
g = 0.9
c = 1

# Input for user: delay parameter
delay = 10

Delayline = np.zeros((delay,1), dtype=float)
y = np.zeros((length_x,1), dtype=float)

for n in range(0,(length_x-1)):
  y[n] = float(c*x[n]) + float(g*Delayline[delay-1])
  Delayline = np.append(y[n], Delayline[0:delay-1])

plt.subplot(2,2,1)
plt.stem(range(length_x),x)
plt.xlabel(r"$n$ $\rightarrow$" )
plt.ylabel(r"$x(n) \rightarrow$")
plt.grid(True, color = '0.7', linestyle='-', which='major', axis='both')
plt.grid(True, color = '0.9', linestyle='-', which='minor', axis='both')
plt.title("IIR comb filter input signal")

plt.subplot(2,2,3)
plt.stem(range(length_x),y)
plt.xlabel(r"$n$ $\rightarrow$" )
plt.ylabel(r"$y(n) \rightarrow$")
plt.grid(True, color = '0.7', linestyle='-', which='major', axis='both')
plt.grid(True, color = '0.9', linestyle='-', which='minor', axis='both')
plt.title("IIR comb filter output signal")

plt.subplot(2,2,2)
W, H = freqz(y,1)
plt.plot(W/math.pi, np.abs(H))
plt.xlabel(r"$f$ x $\pi$ in rad/sample $\rightarrow$" )
plt.ylabel(r"$Magnitude \rightarrow$")
plt.title("IIR comb filter Magnitude response")
plt.grid(True, color = '0.7', linestyle='-', which='major', axis='both')
plt.grid(True, color = '0.9', linestyle='-', which='minor', axis='both')
plt.tight_layout()

plt.subplot(2,2,4)
plt.plot(W/math.pi, np.unwrap(np.angle(H))*180/math.pi)
plt.xlabel(r"$f$ x $\pi$ in rad/sample $\rightarrow$")
plt.ylabel(r"$Phase (degrees) \rightarrow$",)
plt.title(r"IIR comb filter Phase Response");
plt.grid(True, color = '0.7', linestyle='-', which='major', axis='both')
plt.grid(True, color = '0.9', linestyle='-', which='minor', axis='both')
plt.tight_layout()
    
plt.savefig("C:/Users/frota/Documents/mestrado/Top_esp_Audio/figures/iircomb.jpg", dpi=600, bbox_inches='tight')