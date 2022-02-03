# UX_cross_synthesis_CV.m [DAFXbook, 2nd ed., chapter 8]
# ==== This function performs a cross-synthesis with channel vocoder

import librosa
import soundfile as sf
import numpy as np
from scipy import signal
import math 

def UX_cross_synthesis_CV(sou = 'moore_guitar.wav', env='toms_diner.wav'):
    #----- setting user data -----
    DAFx_in_sou, FS = librosa.load(sou)  # signal for source extraction
    DAFx_in_env, FS = librosa.load(env)  # signal for spec. env. extraction
    ly = min(len(DAFx_in_sou), len(DAFx_in_env))  # min signal length
    DAFx_out = np.zeros(ly)  # result signal
    r = 0.99  # sound output normalizing ratio
    lp = np.array([1, -2*r, +r*r])  # low-pass filter used
    epsi = 0.00001 
    #----- init bandpass frequencies
    f0 = 10  # start freq in Hz
    f0 = f0/FS *2  # normalized freq
    fac_third = 2**(1/3)  # freq factor for third octave
    K = math.floor(np.log(1/f0) / np.log(fac_third))  # number of bands
    #----- performing the vocoding or cross synthesis effect -----
    print('band number (max. {0}):'.format(K)) 
    #tic
    #print(DAFx_in_env.shape)
    #print(DAFx_in_sou.shape)
    for k in range (K):
        print('{0} '.format(k)) 
        f1 = f0 * fac_third  # upper freq of bandpass
        [b, a] = signal.cheby1(2, 3, [f0, f1], btype = 'bandpass')  # Chebyshev-type 1 filter design
        f0 = f1  # start freq for next band
        #-- filtering the two signals --
        z_sou = signal.lfilter(b, a, DAFx_in_sou) 
        z_env = signal.lfilter(b, a, DAFx_in_env)
        rms_env = np.sqrt(signal.lfilter(np.array([1]), lp, z_env*z_env))  # RMS value of sound 2
        rms_sou = np.sqrt(epsi+ signal.lfilter(np.array([1]), lp, z_sou*z_sou))  # with whitening
        # rms_sou = 1.  # without whitening
        DAFx_out = DAFx_out + z_sou*rms_env/rms_sou  # add result to output buffer

        #toc
    #----- playing and saving output sound -----
    #soundsc(DAFx_out, FS)
    DAFx_out_norm = r * DAFx_out/max(abs(DAFx_out))  # scale for wav output
    print('cheguei aqui')
    sf.write('CrossCV.wav', DAFx_out_norm, FS)

if __name__ == '__main__':
    UX_cross_synthesis_CV()