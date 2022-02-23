import numpy as np

from scipy.signal.windows import triang, blackmanharris, hann
from scipy.signal import decimate, resample_poly
from peakinterp import peakinterp
from f0detectiontwm import f0detectiontwm
from f0detectionyin_gabi import f0detectionyin
from genspecsines import genspecsines
from custom_windows import custom_triang, custom_blackmanharris
from hpsmodel_utils import interpolate_1d_vector

def hpsmodelparams_gabi(
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
    stocf: int,
    timemapping: np.ndarray = None,
    f0detection: str = 'twm',
    Ns: int = 1024,
    H: int = 256
):
    """
    Analysis and synthesis of a sound using the sinusoidal harmonic model

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
        stocf (int): Decimation factor of mag spectrum for stochastic analysis
        timemapping (np.ndarray, optional): mapping between input and output time (sec)
        f0detection (str, optional): Method for f0 detection. Accepts 'yin' and 'twm'. Defaults to 'yin'.
        Ns (int, optional): FFT size for synthesis
        H (int, optional): Hop size for analysis and synthesis

    Returns:
        y (np.ndarray): Output signal
        yh (np.ndarray): Harmonic component
        ys (np.ndarray): Stochastic component
    """
    # tratamento de entradas
    if f0detection not in ['twm', 'yin']:
        raise Exception("Método inválido de detecção de f0. Escolha \'yin\' ou \'twm\'")

    # argument not specified
    if not timemapping:
        timemapping = np.array([[0, len(x)/fs], [0, len(x)/fs]])

    M = len(w)                                                                      # analysis window size
    N2 = N//2 + 1                                                                   # half-size of spectrum
    hNs = Ns//2                                                                     # half synthesis window size
    hM = (M - 1)//2                                                                 # half analysis window size
    out_audio_len = 1 + round(timemapping[1, -1] * fs)                              # length of output sound
    yh = np.zeros(out_audio_len + Ns//2)                                            # output sine component
    ys = np.zeros(out_audio_len + Ns//2)                                            # output residual component
    w = w / np.sum(w)                                                               # normalize analysis window

    sw = np.zeros(Ns)
    # como o comportamento da janela triangular estava diferente,
    # fiz uma funcao propria seguindo a regra da funcao do matlab
    # para manter o mesmo comportamento
    ow = custom_triang(2*H - 1)
    # ow = triang(2*H - 1, sym=False)                                               # overlapping window

    ovidx = np.arange((Ns//2 + 1 - H), (Ns//2 + H), dtype=np.int64)                 # overlap indexes
    sw[ovidx] = ow[:(2*H)]

    # como o comportamento da blackmanharris estava diferente,
    # fiz uma funcao propria seguindo a regra da funcao do matlab
    # para manter o mesmo comportamento da blackmanharris periodica
    # bh = custom_blackmanharris(Ns)
    bh = blackmanharris(Ns, sym=False)                                              # synthesis window
    bh = bh / np.sum(bh)                                                            # normalize synthesis window
    wr = bh.copy()                                                                  # residual window 

    sw[ovidx] = sw[ovidx] / bh[ovidx]

    sws = H * hann(Ns, sym=False)/2                                                 # synthesis window for stochastic

    lastyhloc = np.zeros(nH)                                                        # initialize synthesis harmonic locations
    yhphase = 2 * np.pi * np.random.rand(nH)                                        # initialize systhesis harmonic phases

    minpin = np.maximum(H + 1, 1 + hM)
    maxpin = len(x) - np.maximum(hM, hNs) - 1
    pout = minpin                                                                   # initialize sound pointer to middle of analysis window
    poutend = out_audio_len - np.maximum(hM, H)                                     # last sample to start a frame

    while pout < poutend:
        pin = int(np.round(
            np.interp(
                pout/fs,
                timemapping[0, :],
                timemapping[1, :]
            ) * fs
        ))
        pin = max(minpin, pin)
        pin = min(maxpin, pin)

        # analysis
        fftbuffer = np.zeros(N)                                                     # reset buffer for FFT
        xw = x[(pin - hM - 1):(pin + hM + 1)] * w[:M]

        fftbuffer[:((M + 1)//2)] = xw[((M + 1)//2):M]                               # zero-phase window in fftbuffer
        fftbuffer[(N - (M - 1)//2):N] = xw[:((M - 1)//2)]

        X = np.fft.fft(fftbuffer)                                                   # compute the FFT
        mX = 20 * np.log10(np.abs(X[:N2]))                                          # magnitude spectrum of positive frequencies
        pX = np.unwrap(np.angle(X[:N2]))                                            # unwrapped phase spectrum

        ploc = np.nonzero(
            (mX[1:N2-1] > t) * \
            (mX[1:N2-1] > mX[2:N2]) * \
            (mX[1:N2-1] > mX[:N2-2])
        )[0] + 1                                                                    # find peaks
        ploc, pmag, pphase = peakinterp(mX, pX, ploc)                               # refine peak values

        if f0detection == 'twm':
            f0 = f0detectiontwm(mX, fs, ploc, pmag, f0et, minf0, maxf0)
        # callyin.py
        if f0detection == 'yin':
            yinws = round(0.015 * fs) + round(0.015 * fs) % 2                       # using approx. a 15 ms window for yin and making it even
            yb = pin - yinws//2
            ye = pin + yinws + yinws//2
            if (yb < 1) or (ye > len(x)):
                f0 = 0
            else:
                f0 = f0detectionyin(x[(yb-1):ye], fs, yinws, minf0, maxf0)          # compute f0

        hloc = np.zeros(nH)                                                         # initialize harmonic locations
        hmag = np.zeros(nH) - 100                                                   # initialize harmonic magnitudes
        hphase = np.zeros(nH)                                                       # initialize harmonic phases
        hf = (f0 > 0) * f0 * np.arange(1, nH + 1)                                   # initialize harmonic frequencies
        hi = 0                                                                      # initialize harmonic index

        npeaks = len(ploc)
        while (f0 > 0) and (hi < nH) and (hf[hi] < fs/2):                           # find harmonic peaks
            peak = np.abs(ploc[:npeaks]/N*fs - hf[hi])                              # find harmonic peaks
            dev = np.min(peak)                                                      # closest peak
            pei = np.argmin(peak)

            if ((hi == 0) or not (hloc[:hi] == ploc[pei]).any()) and (dev < maxhd * hf[hi]):
                hloc[hi] = ploc[pei]                                                # harmonic locations
                hmag[hi] = pmag[pei]                                                # harmonic magnitudes
                hphase[hi] = pphase[pei]                                            # harmonic phases
            hi += 1                                                                 # increase harmonic index
        hloc[:hi] = (hloc[:hi] != 0) * ((hloc[:hi] - 1)*Ns/N + 1)                   # synthesis locs

        ri = pin - hNs                                                              # input pointer for residual analysis
        xr = x[ri:(ri + Ns)] * wr[:Ns]                                              # window the input sound
        Xr = np.fft.fft(np.fft.fftshift(xr))                                        # compute fft for residual analysis
        Xh = genspecsines(hloc[:hi], hmag, hphase, Ns)                              # generate sines
        Xr = Xr - Xh                                                                # get the residual complex spectrum
        mXr = 20 * np.log10(np.abs(Xr[:Ns//2 + 1]))                                 # magnitude spectrum of residual
        mXsenv = decimate(np.maximum(np.zeros_like(mXr) - 200, mXr), stocf)         # decimate the magnitude spectrum

        yhloc = hloc                                                                # synthesis harmonic locs
        yhmag = hmag                                                                # synthesis harmonic amplitudes
        mYsenv = mXsenv                                                             # synthesis residual envelope
        yf0 = f0                                                                    # synthesis f0

        # transformations

        # synthesis
        yhphase = yhphase + 2 * np.pi * (lastyhloc + yhloc)/2/Ns * H                # propagate phases
        lastyhloc = yhloc
        Yh = genspecsines(yhloc, yhmag, yhphase, Ns)                                # generate sines
        mYs = interpolate_1d_vector(mYsenv, stocf)                                  # interpolate to original size
        roffset = int(np.ceil(stocf/2) - 1)                                         # interpolated array offset
        mYs = np.append(
            mYs[0] * np.ones(roffset),
            mYs[:(Ns//2 + 1 - roffset)]
        )
        mYs = 10**(mYs/20)                                                          # dB to linear magnitude

        if f0 > 0:
            mYs = mYs * np.cos(np.pi * np.arange(Ns//2 + 1)/Ns * fs/yf0)**2         # filter residual

        fc = 1 + round(500/fs * Ns)                                                 # 500Hz

        mYs[:fc] = mYs[:fc] * (np.arange(fc)/(fc-1))**2                             # HPF
        pYs = 2 * np.pi * np.random.rand(Ns//2 + 1, 1)                              # generate phase spectrum with random values
        mYs1 = np.append(mYs[:Ns//2+1], mYs[Ns//2:1:-1])                            # create complete magnitude spectrum
        pYs1 = np.append(pYs[:Ns//2+1], -1 * pYs[Ns//2:1:-1])                       # create complex phase spectrum
        Ys = mYs1 * np.cos(pYs1) + 1j * mYs1 * np.sin(pYs1)                         # compute complex spectrum

        yhw = np.fft.fftshift(np.real(np.fft.ifft(Yh)))                             # sines in time domain using IFFT
        ysw = np.fft.fftshift(np.real(np.fft.ifft(Ys)))                             # stochastic in time domain using IFFT

        ro = pout - hNs                                                             # output sound pointer for overlap
        yh[ro:ro+Ns] += yhw[:Ns] * sw                                               # overlap-add for sines
        ys[ro:ro+Ns] += ysw[:Ns] * sws                                              # overlap-add for stochastic

        pout += H                                                                   # advance the sound pointer
    y = yh + ys                                                                     # sum sines and stochastic


    return y, yh, ys
