import numpy as np

def diranalysis(
    s: np.ndarray,
    theta: np.ndarray,
    fs: float,
    with_noise: bool = False, 
    hopsize: int = 256,
    winsize: int = 512
):
    """
    Args:
        s (np.ndarray): (N, number of signals)
        theta (np.ndarray): (number of signals, )
    """

    N, _ = s.shape

    x_elements = np.cos(theta / 180 * np.pi) # (number of signals, )
    y_elements = np.sin(theta / 180 * np.pi) # (number of signals, )

    w = np.sum(s, axis=1) / np.sqrt(2)
    x = np.sum(np.multiply(s, x_elements), axis=1)
    y = np.sum(np.multiply(s, y_elements), axis=1)

    if with_noise:
        # Add fading in diffuse noise with  36 sources evenly in the horizontal plane
        # Each direction will produce a different source
        for direction in range(0, 360, 10):
            noise = (np.random.rand(N) - 0.5) * (10 ** ((np.arange(N) / fs) * 2))
            w += noise / np.sqrt(2)
            x += noise * np.cos(direction / 180 * np.pi)
            y += noise * np.sin(direction / 180 * np.pi)

    alpha = 1. / (0.02 * fs / winsize)

    iterable = range(0, (len(x) - winsize), hopsize)
    intensity = np.zeros((hopsize, 2)) + np.finfo(float).eps
    energy = np.zeros((hopsize, len(iterable))) + np.finfo(float).eps
    azimuth = np.zeros((hopsize, len(iterable))) + np.finfo(float).eps
    diffuseness = np.zeros((hopsize, len(iterable))) + np.finfo(float).eps

    for i, time in enumerate(iterable):
        W = np.fft.fft(w[time : (time + winsize)] * np.hanning(winsize))
        X = np.fft.fft(x[time : (time + winsize)] * np.hanning(winsize))
        Y = np.fft.fft(y[time : (time + winsize)] * np.hanning(winsize))
        W = W[:hopsize]
        X = X[:hopsize]
        Y = Y[:hopsize]
        U = np.concatenate((X.reshape(-1, 1), Y.reshape(-1, 1)), axis=1) # hopsize x 2
        P_ = np.repeat(np.conjugate(W).reshape(-1, 1), 1, axis=1) * np.sqrt(2) # hopsize x 2

        temp_intensity = np.real(P_ * U)
        intensity = temp_intensity * alpha + intensity * (1 - alpha)

        azimuth[:, i] = np.around(np.arctan2(intensity[:, 1], intensity[:, 0]) * (180/np.pi)) # hopsize x _

        temp_energy = \
            (1 / 4) * np.sum(np.absolute(U) ** 2, axis=1) \
            + np.absolute(W * np.sqrt(2)) ** 2 \
            + np.finfo(float).eps
        energy[:, i] = temp_energy * alpha + energy[:, (i - 1)] * (1 - alpha)

        diffuseness[:, i] = 1 - np.sqrt(np.sum(intensity ** 2, axis=1)) / energy[:, i]

    return energy, azimuth, diffuseness
