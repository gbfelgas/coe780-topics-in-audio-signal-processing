import numpy as np

def custom_triang(N):
    w = np.zeros(N)

    if N % 2 != 0: # odd
        for n in range(1, N + 1):
            if n <= (N + 1) / 2:
                w[n - 1] = 2*n/(N + 1)
            else:
                w[n - 1] = 2 - 2*n/(N + 1)
    else: # even
        for n in range(1, N + 1):
            if n <= N/2:
                w[n - 1] = 2*(n - 1)/N
            else:
                w[n - 1] = 2 - 2*(n - 1)/N

    return w

def custom_blackmanharris(N):
    a0 = 0.35875
    a1 = 0.48829
    a2 = 0.14128
    a3 = 0.01168

    return a0 - \
        a1 * np.cos(2 * np.pi * np.arange(N) / N) + \
        a2 * np.cos(4 * np.pi * np.arange(N) / N) - \
        a3 * np.cos(6 * np.pi * np.arange(N) / N)