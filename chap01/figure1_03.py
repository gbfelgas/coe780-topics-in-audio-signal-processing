import audiofile as af
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

x, FS = af.read('audios/ton2.wav')

plt.figure()
plt.subplot(3,1,1)
plt.plot(np.arange(8000,dtype='int64'),x[:8000])
plt.ylabel('x(n)')
plt.axis(ymin = -0.5, ymax = 0.5, xmin = 0, xmax = 8000)
plt.subplot(3,1,2)
plt.plot(np.arange(1000,dtype='int64'),x[:1000])
plt.ylabel('x(n)')
plt.axis(ymin = -0.4, ymax = 0.4, xmin = 0, xmax = 1000)
ax = plt.gca()
ax.xaxis.set_major_locator(MultipleLocator(100))
ax.yaxis.set_major_locator(MultipleLocator(0.2))
plt.subplot(3,1,3)
plt.stem(np.arange(100,dtype='int64'),x[:100],markerfmt='C0.',basefmt='gray')
plt.ylabel('x(n)')
plt.xlabel('n \u2192')
plt.axis(ymin = -0.05, ymax = 0.05, xmin = 0, xmax = 100)
plt.subplots_adjust(hspace = 0.3)

plt.savefig("figures/figure1_03.png", dpi=300, bbox_inches='tight')
