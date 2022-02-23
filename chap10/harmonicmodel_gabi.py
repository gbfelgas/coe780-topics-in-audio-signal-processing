import numpy as np

from scipy.signal.windows import triang, blackmanharris
from peakinterp import peakinterp
from genspecsines import genspecsines
from f0detectiontwm import f0detectiontwm
from f0detectionyin_gabi import f0detectionyin
from custom_windows import custom_triang, custom_blackmanharris

def harmonicmodel(
    x: np.ndarray,
    fs: float,
    w: np.ndarray,
    N: int,
    t: int,
    nH: int,
    minf0: float,
    maxf0: float,
    f0et: float,
    maxhd: float,
    f0detection: str = 'twm',
    Ns: int = 1024,
    H: int = 256
):
    """
    Analysis and synthesis of a audio signal using the sinusoidal harmonic model

    Args:
        x (np.ndarray): Input signal
        fs (float): Sampling frequency
        w (np.ndarray): Analysis window
        N (int): FFT size for analysis
        t (float): Threshold in negative dB
        nH (int): Maximum number of harmonics
        minf0 (float): Minimum f0 in the signal
        maxf0 (float): Maximum f0 in the signal
        f0et (float): Error threshold in the f0 detection
        maxhd (float): Maximum relative deviation in harmonic detection
        f0detection (str, optional): Method for f0 detection. Accepts 'yin' and 'twm'. Defaults to 'yin'.
        Ns (int, optional): FFT size for synthesis
        H (int, optional): Hop size for analysis and synthesis

    Returns:
        y (np.ndarray): Output signal
    """
    # tratamento de entradas
    if f0detection not in ['twm', 'yin']:
        raise Exception("Método inválido de detecção de f0. Escolha \'yin\' ou \'twm\'")

    M = len(w)                                                                                  # analysis window size
    N2 = N//2 + 1                                                                               # size of positive spectrum
    audio_len = len(x)                                                                          # length of input audio signal array
    hNs = Ns//2                                                                                 # half synthesis window size
    # hM matlab: 511.5
    # hM python: 511
    hM = (M - 1)//2                                                                             # half analysis window size
    y = np.zeros(audio_len + Ns//2)	                                                            # initialize output array
    w = w / np.sum(w)                                                                           # normalize analysis window

    sw = np.zeros(Ns)

    # como o comportamento da janela triangular estava diferente,
    # fiz uma funcao propria seguindo a regra da funcao do matlab
    # para manter o mesmo comportamento
    ow = custom_triang(2*H - 1)
    # ow = triang(2*H - 1, sym=False)                                                           # overlapping window

    # slidado em -1 em relacao ao matlab
    ovidx = np.arange((Ns//2 + 1 - H), (Ns//2 + H), dtype=np.int64)                             # overlap indexes
    sw[ovidx] = ow[:(2*H)]

    # como o comportamento da blackmanharris estava diferente,
    # fiz uma funcao propria seguindo a regra da funcao do matlab
    # para manter o mesmo comportamento da blackmanharris periodica
    # bh = custom_blackmanharris(Ns)
    bh = blackmanharris(Ns, sym=False)                                                        # synthesis window
    bh = bh / np.sum(bh)                                                                      # normalize synthesis window

    sw[ovidx] = sw[ovidx] / bh[ovidx]

    # pin matlab: 512.5
    # pin python: 512
    pin = np.maximum(H + 1, 1 + hM)                                                             # initialize audio signal pointer to middle of analysis window
    pend = audio_len - hM                                                                       # last sample to start a frame

    while pin < pend:

        # analysis
        fftbuffer = np.zeros(N)                                                                 # reset buffer for FFT
        xw = x[(pin - hM - 1):(pin + hM + 1)] * w[:M]

        fftbuffer[:((M + 1)//2)] = xw[((M + 1)//2):M]                                           # zero-phase window in fftbuffer
        fftbuffer[(N - (M - 1)//2):N] = xw[:((M - 1)//2)]

        X = np.fft.fft(fftbuffer)                                                               # compute the FFT
        mX = 20 * np.log10(np.abs(X[:N2]))                                                      # magnitude spectrum of positive frequencies
        pX = np.unwrap(np.angle(X[:N2]))                                                        # unwrapped phase spectrum

        ploc = np.nonzero(
            (mX[1:N2-1] > t) * \
            (mX[1:N2-1] > mX[2:N2]) * \
            (mX[1:N2-1] > mX[:N2-2])
        )[0] + 1                                                                                # find peaks
        ploc, pmag, pphase = peakinterp(mX, pX, ploc)                                           # refine peak values

        if f0detection == 'twm':
            f0 = f0detectiontwm(mX, fs, ploc, pmag, f0et, minf0, maxf0)
        # callyin.py
        if f0detection == 'yin':
            yinws = round(0.015 * fs) + round(0.015 * fs) % 2                                   # using approx. a 15 ms window for yin and making it even
            yb = pin - yinws//2
            ye = pin + yinws + yinws//2
            if (yb < 1) or (ye > len(x)):
                f0 = 0
            else:
                f0 = f0detectionyin(x[(yb-1):ye], fs, yinws, minf0, maxf0)                      # find f0

        hloc = np.zeros(nH)                                                                     # initialize harmonic locations
        hmag = np.zeros(nH) - 100                                                               # initialize harmonic magnitudes
        hphase = np.zeros(nH)                                                                   # initialize harmonic phases
        hf = (f0 > 0) * f0 * np.arange(1, nH + 1)                                               # initialize harmonic frequencies
        hi = 0                                                                                  # initialize harmonic index

        npeaks = len(ploc)
        while (f0 > 0) and (hi < nH) and (hf[hi] < fs/2):
            peak = np.abs(ploc[:npeaks]/N*fs - hf[hi])                                          # find harmonic peaks
            dev = np.min(peak)                                                                  # closest peak
            pei = np.argmin(peak)

            if ((hi == 0) or not (hloc[:hi] == ploc[pei]).any()) and (dev < maxhd * hf[hi]):
                hloc[hi] = ploc[pei]                                                            # harmonic locations
                hmag[hi] = pmag[pei]                                                            # harmonic magnitudes
                hphase[hi] = pphase[pei]                                                        # harmonic phases
            hi += 1                                                                             # increase harmonic index

        hloc[:hi] = (hloc[:hi] != 0) * ((hloc[:hi] - 1)*Ns/N + 1)                                  # synthesis locs

        # synthesis
        Yh = genspecsines(hloc[:hi], hmag, hphase, Ns)                                      # generate spec sines
        yh = np.fft.fftshift(np.real(np.fft.ifft(Yh)))                                          # time domain of sinusoids
        y[(pin - hNs):(pin + hNs)] = y[(pin - hNs):(pin + hNs)] + sw * yh[:Ns]                  # overlap-add

        pin = pin + H                                                                           # advance the audio signal pointer

    return y
