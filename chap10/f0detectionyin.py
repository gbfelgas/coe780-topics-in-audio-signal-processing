from logging import raiseExceptions
import numpy as np
from librosa import load

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
    # inicialização
    maxlag = ws - 2 # atraso máximo -- até 1 janela de integração
    
    d = np.zeros(maxlag) 
    d2 = np.zeros(maxlag)
    
    # preencher o vetor d(tau)
    x_aux = x[0:ws] # janela
    cumsumx = np.sum(x_aux ** 2) # sum(x(j)^2)
    cumsumx_aux = cumsumx # sum(x(j+tau)^2)
    corr = np.correlate(x[0:2*ws], x_aux, "full") # sum(x(j)x(j+tau))
    corr = corr[2*ws:3*ws-2]
    for lag in range(maxlag-1):
        d[lag] = cumsumx - 2*corr[lag] + cumsumx_aux
        cumsumx_aux = cumsumx_aux - x[lag] ** 2 + x[lag + ws] ** 2 # atualiza sum(x(j+tau)^2) pegando a amostra do quadro seguinte
    
    # preencher o vetor d'(tau)
    d2[0] = 1
    cumsum = 0
    for lag in range(1, maxlag-1):
        cumsum = cumsum + d[lag]
        d2[lag] = (d[lag] * lag)/cumsum

    # ???
    if np.min(d2) > unvoiced_threshold:
        raise Exception("Sinal sem pitch ...")

    # converter as frequencias minima e maxima de busca em atrasos em amostras
    minf0lag = 1 + round(fs/minf0) # quanto menor minf0, maior esse atraso
    maxf0lag = 1 + round(fs/maxf0) # quanto maior maxf0, menor esse atraso

    # por segurança, coloca um valor alto em todos os atrasos menores que maxf0lag e maiores que minf0lag
    if (maxf0lag < maxlag):
        d2[:maxf0lag] = 100
    if (minf0lag < maxlag):
        d2[minf0lag:] = 100

    # pegar minimos
    idx_minima = np.where((d2[1:-1] < d2[0:-2]) * (d2[1:-1] < d2[2:]))[0] + 1

    if len(idx_minima) == 0:
        raise Exception("Não encontrou mínimos locais em  d'(tau) ...")

    # esse pedaço está confuso ...
    if len(np.where(d2[idx_minima] < threshold)[0]) > 0:
        candf0lag = idx_minima[np.where(d2[idx_minima] < threshold)[0][0]]
    else:
        candf0lag = idx_minima[np.argmin(d2[idx_minima])]
    
    # interpolação parabólica
    if (candf0lag < 1) or (candf0lag > maxlag):
        raise Exception("Encontrou um F0 pequeno ou grande demais, tente mudar o tamanho da janela ...")

    lval = d2[candf0lag - 1]
    val = d2[candf0lag]
    rval = d2[candf0lag + 1]
    f0lag = candf0lag + 0.5 * (lval - rval) / (lval - 2 * val + rval)

    f0 = fs/f0lag
    return f0
    
if __name__=="__main__":
    # x, fs = load("violin-B3.wav", sr=None, mono=True)
    x, fs = load("soprano-E4.wav", sr=None, mono=True)
    # minf0 = 233.08 # A#3 
    minf0 = 311.13 # D#4
    # maxf0 = 261.63 # C4
    maxf0 = 349.23 # F4
    f0 = f0detectionyin(x, fs, 4096, minf0, maxf0)
    print(f0)






