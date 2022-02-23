import numpy as np
from scipy.signal.windows import triang, blackmanharris
from peakinterp import peakinterp
from genspecsines import genspecsines

def sinemodel(
    x: np.ndarray,
    w: np.ndarray,
    N: int,
    t: int,
    Ns: int = 1024,
    H: int = 256,
) -> np.ndarray:

    """
        Analysis/synthesis of an audio signal using the sinusoidal model

        Args:
            x (np.ndarray): Input audio signal
            w (np.ndarray): Analysis window (odd size)
            N (int): FFT size
            t (int): Threshold in negative dB
            Ns (int, optional): FFT size for synthesis (even)
            H (int, optional): Analysis and synthesis hop size

        Returns:
            y (np.ndarray): Output audio signal
    """

    M = len(w)                                                                    # analysis window size
    N2 = N//2 + 1                                                                 # size of positive spectrum
    audio_len = len(x)                                                            # length of input audio signal array
    hNs = Ns//2                                                                   # half synthesis window size
    hM = (M - 1)//2                                                               # half analysis window size
    y = np.zeros(audio_len)	                                                      # initialize output array
    w = w / np.sum(w)                                                             # normalize analysis window
    
    sw = np.zeros(Ns)
    ow = triang(2 * H - 1, sym=False)                                             # overlapping window
    ovidx = np.arange((Ns//2 + 1 - H), (Ns//2 + H), dtype=np.int64)               # overlap indexes
    sw[ovidx] = ow[:(2 * H)]
    bh = blackmanharris(Ns, sym=False)                                            # synthesis window
    bh = bh / np.sum(bh)                                                          # normalize synthesis window
    sw[ovidx] = sw[ovidx] / bh[ovidx]

    pin = np.maximum(H + 1, 1 + hM)                                               # initialize audio signal pointer to middle of analysis window
    pend = audio_len - np.maximum(H, hM)                                          # last sample to start a frame

    while pin < pend:

        fftbuffer = np.zeros(N)                                                   # reset buffer for FFT# Analysis
        xw = x[(pin - hM - 1):(pin + hM + 1)] * w[:M]

        fftbuffer[:((M + 1)//2)] = xw[((M + 1)//2):M]                             # zero-phase window in fftbuffer
        fftbuffer[(N - (M - 1)//2):N] = xw[:((M - 1)//2)]

        X = np.fft.fft(fftbuffer)                                                 # compute the FFT
        mX = 20 * np.log10(np.abs(X[:N2]))                                        # magnitude spectrum of positive frequencies
        pX = np.unwrap(np.angle(X[:N2]))                                          # unwrapped phase spectrum

        ploc = np.nonzero(
            (mX[1:N2-1] > t) * \
            (mX[1:N2-1] > mX[2:N2]) * \
            (mX[1:N2-1] > mX[:N2-2])
        )[0] + 1                                                                  # find peaks
        ploc, pmag, pphase = peakinterp(mX, pX, ploc)                             # refine peak values# Synthesis
        plocs = ploc * Ns / N                                                     # adapt peak locations to synthesis FFT
        Y = genspecsines(plocs, pmag, pphase, Ns)                                 # generate spec sines
        yw = np.fft.fftshift(np.real(np.fft.ifft(Y)))                             # time domain of sinusoids
        y[(pin - hNs):(pin + hNs)] = y[(pin - hNs):(pin + hNs)] + sw * yw[:Ns]    # overlap-add
        
        pin = pin + H                                                             # advance the audio signal pointer    

    return y
    