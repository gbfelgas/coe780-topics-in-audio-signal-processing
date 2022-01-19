import numpy as np

def compexp(
    x: np.ndarray,
    CT: float,
    CS: float,
    ET: float,
    ES: float,
    tav: float,
    at: float,
    rt: float,
    delay: int
):
    """
    Compressor and expander function.
    
    Args:
        x (np.ndarray): Input signal.
        CT (float): Compressor threshold.
        CS (float): Compressor Slope.
        ET (float): Expander threshold.
        ES (float): Expander Slope.
        tav (float):
        at (float):
        rt (float):
        delay (float): Buffer delay.
        
    Returns:
        y (np.ndarray): Output signal.
    """

    xrms = 0
    g = 1
    buffer = np.zeros(delay)
    y = np.zeros_like(x)
    final_rms = np.zeros_like(x)
    final_G = np.zeros_like(x)
    final_g = np.zeros_like(x)
    final_f = np.zeros_like(x)

    for n in range(len(x)):
        xrms = (1 - tav) * xrms + tav * x[n]**2
        final_rms[n] = xrms
        X = 10 * np.log10(xrms)
        G = min([0, CS * (CT-X), ES * (ET-X)])
        final_G[n] = G
        f = 10**(G/20)
        final_f[n] = f
        if f < g:
            coeff = at
        else:
            coeff = rt
        g = (1-coeff) * g + coeff * f
        final_g[n] = g
        y[n] = g * buffer[-1]
        buffer = np.insert(buffer[:-1], 0, np.array(x[n]))
        
    return y, final_rms, final_G, final_f, final_g
    