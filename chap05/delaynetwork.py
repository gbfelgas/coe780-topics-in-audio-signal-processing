### FDN CHAPTER 5 ###

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz

import numpy as np
import math

fs=44100
gain=0.97

# Create an impulse
length_x = 1*fs
x = np.zeros((length_x,1), dtype=int) 
x[0] = 1


y = np.zeros((fs,1), dtype=float) 
b = [1, 1, 1, 1]
c = [0.8, 0.8, 0.8, 0.8]
length_y = fs

# Feedback matrix
a = np.zeros((4,4), dtype=float) 
a[0,:] = [0, 1, 1, 0]
a[1,:] = [-1, 0, 0, -1]
a[2,:] = [1, 0, 0, -1]
a[3,:] = [0, 1, -1, 0]

a_1 = a[0,:]*(1/math.sqrt(2)) * gain
a_2 = a[1,:]*(1/math.sqrt(2)) * gain
a_3 = a[2,:]*(1/math.sqrt(2)) * gain
a_4 = a[3,:]*(1/math.sqrt(2)) * gain
a2 = np.array([a_1, a_2, a_3, a_4])

# Delay lines, use prime numbers
m_aux = [149, 211, 263, 293]
m = np.array(m_aux).T
z1 = np.zeros(((max(m)),1), dtype=int) 
z2 = np.zeros(((max(m)),1), dtype=int) 
z3 = np.zeros(((max(m)),1), dtype=int) 
z4 = np.zeros(((max(m)),1), dtype=int) 

for n in range(0,(length_y)):
    tmp = np.array([z1[(m[0]-1)], z2[(m[1]-1)], z3[(m[2]-1)], z4[(m[3]-1)]])
    tmp = np.transpose(tmp)
    y[n] = x[n] + c[0]*z1[(m[0]-1)] + c[1]*z2[(m[1]-1)] + c[2]*z3[(m[2]-1)] + c[3]*z4[(m[3]-1)]
    z1 = np.append((x[n]*b[0] + np.dot(tmp,(a2[0,:]))), z1[0:len(z1)-1])
    z2 = np.append((x[n]*b[1] + np.dot(tmp,(a2[1,:]))), z2[0:len(z2)-1])
    z3 = np.append((x[n]*b[2] + np.dot(tmp,(a2[2,:]))), z3[0:len(z3)-1])
    z4 = np.append((x[n]*b[3] + np.dot(tmp,(a2[3,:]))), z4[0:len(z4)-1])
    print(y[n])
plt.rcParams.update({'font.size': 6})
# Plot the filtering results
plt.subplot(2,2,1)
plt.stem(range(length_x),x)
plt.xlabel(r"$n$ $\rightarrow$" )
plt.ylabel(r"$x(n) \rightarrow$")
#plt.axis([-100, length_x, 0, x[0]*1.1])
plt.grid(True, color = '0.7', linestyle='-', which='major', axis='both')
plt.grid(True, color = '0.9', linestyle='-', which='minor', axis='both')
plt.title("FDN input signal")
plt.tight_layout()

plt.subplot(2,2,3)
plt.stem(range(length_y),y)
plt.xlabel(r"$n$ $\rightarrow$" )
plt.ylabel(r"$y(n) \rightarrow$")
#plt.axis([-100, length_x, ir[0]*1.3, max(ir)*1.3])
plt.grid(True, color = '0.7', linestyle='-', which='major', axis='both')
plt.grid(True, color = '0.9', linestyle='-', which='minor', axis='both')
plt.title("FDN output signal")
plt.tight_layout()

plt.rcParams.update({'font.size': 6})
plt.subplot(2,2,2)
W, H = freqz(y,1)
plt.plot(W/math.pi, 20*np.log10(abs(H)))
plt.xlabel(r"$f$ x $\pi$ in rad/sample $\rightarrow$" )
plt.ylabel(r"$Magnitude (dB) \rightarrow$")
plt.title("FDN Magnitude response")
plt.grid(True, color = '0.7', linestyle='-', which='major', axis='both')
plt.grid(True, color = '0.9', linestyle='-', which='minor', axis='both')
plt.tight_layout()

plt.subplot(2,2,4)
plt.plot(W/math.pi, np.unwrap(np.angle(H))*180/math.pi)
plt.xlabel(r"$f$ x $\pi$ in rad/sample $\rightarrow$")
plt.ylabel(r"$Phase (degrees) \rightarrow$",)
# plt.axis([-0.1, 1.1, -8000, 0])
plt.title(r"FDN Phase Response")
plt.grid(True, color = '0.7', linestyle='-', which='major', axis='both')
plt.grid(True, color = '0.9', linestyle='-', which='minor', axis='both')
plt.tight_layout()
    
plt.savefig("C:/Users/frota/Documents/mestrado/Top_esp_Audio/figures/delaynetwork.jpg", dpi=600, bbox_inches='tight')

