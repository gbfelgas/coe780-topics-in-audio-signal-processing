# Author: U. Zölzer (Matlab)
# Impulse response of 2nd order IIR filter
# Sample-by-sample algorithm
#
#--------------------------------------------------------------------------
# This source code is provided without any warranties as published in 
# DAFX book 2nd edition, copyright Wiley & Sons 2011, available at 
# http://www.dafx.de. It may be used for educational purposes and not 
# for commercial applications without further permission.
#--------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# Coefficients for a high-pass
a=np.array([1, -1.28, 0.47])
b=np.array([0.69, -1.38, 0.69])

# Initialization of state variables
xh1=0;xh2=0;
yh1=0;yh2=0;

# Input signal: unit impulse
N=20;  # length of input signal
x = np.zeros(N)
x[0]=1

# Sample-by-sample algorithm
y = np.zeros(N)
for n in range(N):
    y[n]=b[0]*x[n] + b[1]*xh1 + b[2]*xh2 - a[1]*yh1 - a[2]*yh2;
    xh2=xh1;xh1=x[n];
    yh2=yh1;yh1=y[n];


# Plot results
plt.figure()
plt.subplot(211)
plt.stem(np.arange(N),x,markerfmt='C0.',basefmt='gray')
plt.axis(ymin=-1.2, ymax = 1.2)
plt.xlabel('n \u2192')
ax = plt.gca()
ax.xaxis.set_major_formatter('{x:.0f}')
ax.xaxis.set_major_locator(MultipleLocator(5))
plt.ylabel('x(n) \u2192')

plt.subplot(212)
plt.stem(np.arange(N),y,markerfmt='C0.',basefmt='gray')
plt.axis(ymin=-1.2, ymax = 1.2)
plt.xlabel('n \u2192')
ax = plt.gca()
ax.xaxis.set_major_formatter('{x:.0f}')
ax.xaxis.set_major_locator(MultipleLocator(5))
plt.ylabel('y(n) \u2192')
plt.subplots_adjust(hspace=0.4)
plt.show()

