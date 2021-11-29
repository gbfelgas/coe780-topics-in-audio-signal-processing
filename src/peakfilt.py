import numpy as np

def peakfilt(x: np.ndarray, Wc: float, Wb: float, G: int):
    """
    Applies a peak filter to the input signal x.
    
    Args:
        x (np.ndarray): Input signal.
        Wc (float): Normalized center frequency 0 < Wc < 1, i.e. 2 * fc / fs.
        Wb (float): Normalized bandwidth 0 < Wb < 1, i.e. 2 * fb / fs.
        G (int): Gain in dB.
        
    Returns:
        y (np.ndarray): Output signal.
    """
    
    V0 = 10 ** (G / 20)
    H0 = V0 - 1
    
    tan_wb = np.tan(np.pi * Wb / 2)
    if G >= 0:
        c = (tan_wb - 1) / (tan_wb + 1)
    else:
        c = (tan_wb - V0) / (tan_wb + V0)
        
    d = - np.cos(np.pi * Wc)
    xh = [0, 0]
    y = np.zeros_like(x)
    
    for n in range(len(x)):
        xh_new = x[n] - d * (1 - c) * xh[0] + c * xh[1]
        ap_y = - c * xh_new + d * (1 - c) * xh[0] + xh[1]
        xh = [xh_new, xh[0]]

        y[n] = 0.5 * H0 * (x[n] - ap_y) + x[n]
        
    return y