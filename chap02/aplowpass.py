import numpy as np

def aplowpass (x, Wc):

    # y = aplowpass (x, Wc)
    # Author: M. Holters
    # Applies a lowpass filter to the input signal x.
    # Wc is the normalized cut-off frequency 0<Wc<1, i.e. 2*fc/fS.
    
    c = (np.tan(np.pi*Wc/2)-1) / (np.tan(np.pi*Wc/2)+1)
    xh = 0
    N = x.size
    y = np.zeros(x.shape)
    
    for n in range(N):
        xh_new = x[n] - c*xh
        ap_y = c * xh_new + xh
        xh = xh_new
        y[n] = 0.5 * (x[n] + ap_y) #change to minus for highpass
        
    return y