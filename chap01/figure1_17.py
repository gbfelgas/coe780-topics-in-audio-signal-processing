import numpy as np
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt

x1=[-1, -0.5, 1, 2, 2, 1, 0.5, -1]
x2=[-1, -0.5, 1, 2, 2, 1, 0.5, -1, 0, 0, 0, 0, 0, 0, 0, 0]
plt.subplot(2,2,1)
plt.stem([0, 1, 2, 3, 4, 5, 6, 7],x1)
plt.axis([-0.5, 7.5, -1.5, 2.5])
plt.ylabel('x(n) \u2192')
plt.title('8 samples')
plt.subplot(2,2,2)
plt.stem([0, 1, 2, 3, 4, 5, 6, 7],abs(fft(x1)))
plt.axis([-0.5, 7.5, -0.5, 10])
plt.ylabel('|X(k)| \u2192')
plt.title('8-point FFT')
plt.subplot(2,2,3)
plt.stem([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],x2)
plt.axis([-0.5, 15.5, -1.5, 2.5])
plt.xlabel('n \u2192')
plt.ylabel('x(n) \u2192')
plt.title('8 samples+zero-padding')
plt.subplot(2,2,4)
plt.stem([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],abs(fft(x2)))
plt.axis([-1, 16, -0.5, 10])
plt.xlabel('k \u2192')
plt.ylabel('|X(k)| \u2192')
plt.title('16-point FFT')
plt.tight_layout()
plt.savefig("Ex1.3.png", dpi=300, bbox_inches='tight')
plt.show()
