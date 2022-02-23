import numpy as np

def f0detectionyin(x, fs, ws, minf0, maxf0, threshold=0.1, unvoiced_threshold=0.2):
    """Detects f0 using the Yin method

    Args:
        x (np.array): array of input signal samples
        fs (float): sampling frequency
        ws (int): integration window length in samples
        threshold (float): ???
        unvoiced_threshold: maximal minimum of d'(tau) for a voiced signal
        minf0 (float): minimum f0
        maxf0 (float): maximum f0
    """
    maxlag = ws - 2
    
    d = np.zeros(maxlag) 
    d2 = np.zeros(maxlag)
    
    x1 = x[:ws]
    cumsumx = np.sum(x1**2)
    cumsumx1 = cumsumx
    xy = np.correlate(x[:2*ws], np.pad(x1, (0, len(x1))), "full")
    xy = xy[2*ws:3*ws-2]

    for lag in range(maxlag):
        d[lag] = cumsumx - 2*xy[lag] + cumsumx1
        cumsumx1 = cumsumx1 - x[lag]**2 + x[lag + ws + 1]**2
    
    d2[0] = 1
    cumsum = 0
    for lag in range(1, maxlag):
        cumsum = cumsum + d[lag]
        d2[lag] = (d[lag] * lag)/cumsum

    minf0lag = 1 + round(fs/minf0) 
    maxf0lag = 1 + round(fs/maxf0)

    if (maxf0lag > 1) and (maxf0lag < maxlag):
        d2[:maxf0lag] = 100
    if (minf0lag > 1) and (minf0lag < maxlag):
        d2[minf0lag:] = 100
    
    # find the best candidate
    mloc = np.nonzero(
        (d2[1:-1] < d2[2:]) * \
        (d2[1:-1] < d2[:-2])
    )[0] + 1

    candf0lag = 0
    if (len(mloc) > 0):
        I = np.nonzero(d2[mloc] < threshold)[0]

        if (len(I) > 0):
            candf0lag = mloc[I[0]]
        else:
            I2 = np.argmin(d2[mloc])
            candf0lag = mloc[I2[0]]
    
        if (candf0lag > 1) and (candf0lag < maxlag):
            # parabolic interpolation
            lval = d2[candf0lag - 1] 
            val = d2[candf0lag] 
            rval = d2[candf0lag + 1] 
            candf0lag = candf0lag + .5*(lval - rval)/(lval - 2*val + rval)

    ac = np.min(d2)
    f0 = fs/candf0lag          # compute candidate frequency in Hz
    if (ac > 0.2):             # voiced/unvoiced threshold
        f0 = 0                 # set to unvoiced

    return f0
