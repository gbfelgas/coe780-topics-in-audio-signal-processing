import numpy as np

def unicomb(x: np.ndarray, BL: float, FB: float, FF: float, M: int):
    """
    Universal comb filter. It can work as a FIR comb filter, IIR comb filter,
    Allpass or Delay.

    Args:
        x (np.ndarray): Input signal.
        BL (float): Blend parameter.
        FB (float): Feedbackward parameter.
        FF (float): Feedforward parameter.
        M (int): Delay number of samples.
    
    Returns:
        y (np.ndarray): Output signal.
    """

    delayline = np.zeros(M) # M-shape memory of x_h
    y = np.zeros_like(x)
    
    for n in range(len(x)):
        x_h = x[n] + FB * delayline[-1]
        y[n] = FF * delayline[-1] + (BL * x_h)
        delayline = np.insert(delayline[:-1], 0, x_h)

    return y