
import scipy.signal as sig
import numpy as np

def UX_spectral_interp(x1, x2, n1 = 512, n2 = 512, s_win = 2048, cut = 50):
   '''
   UX_spectral_interp.py   [DAFXbook, 2nd ed., chapter 8]
   ==== This function performs a spectral interpolation with cepstrum
   ---- inputs:
      x1             input signal 1
      x2             input signal 2
      n1             analysis step [samples]
      n2             synthesis step [samples]
      s_win          window size [samples]
      cut            cut-off quefrency (for quefrency LP filtering)
   ---- outputs:
      y             spectrally interpolated mix of both input signals

   k: spectral mix, calculated at every step in this example, as
      starts with gain=0 for sound 1 and gain=1 for sound 2
      finishes with gain=1 for sound 1 and gain=0 for sound 2
      so we move from sound 1 to sound 2
   '''
   # Adapting inputs' shape to (sample, channel)
   if x1.ndim == 1:
      DAFx_in1 = x1.reshape((x1.shape[0],1))
   elif x1.ndim == 2:
      if x1.shape[0]>x1.shape[1]:
         DAFx_in1 = x1.copy()
      else:
         DAFx_in1 = x1.T.copy()
   else:
      raise TypeError('unknown audio data format !!!')
   
   if x2.ndim == 1:
      DAFx_in2 = x2.reshape((x2.shape[0],1))
   elif x2.ndim == 2:
      if x2.shape[0]>x2.shape[1]:
         DAFx_in2 = x2.copy()
      else:
         DAFx_in2 = x2.T.copy()
   else:
      raise TypeError('unknown audio data format !!!')
   
   if DAFx_in1.shape[1] != DAFx_in2.shape[1]:
      raise Exception('mismatching number of channels !')
   else:
      nChan = DAFx_in1.shape[1]
      print('Number of Channels: ' + str(nChan))

   #----- initializations -----
   w1 = sig.windows.hann(s_win, sym=False)
   w1 = np.tile(w1,nChan).reshape((nChan,len(w1))).T
   w2 = w1.copy()    # output window       
   L  = min(DAFx_in1.shape[0], DAFx_in2.shape[0])

   # 0-pad + normalize
   DAFx_in1 = np.vstack((np.zeros((s_win, nChan)),DAFx_in1,np.zeros((s_win-L%n1,nChan))))/abs(DAFx_in1).max() 
   DAFx_in2 = np.vstack((np.zeros((s_win, nChan)),DAFx_in2,np.zeros((s_win-L%n1,nChan))))/abs(DAFx_in2).max()

   DAFx_out = np.zeros((DAFx_in1.shape[0],nChan))

   #UUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU
   pin  = 0
   pout = 0
   pend = L - s_win
   while pin<pend:

      #---- k factor (spectral mix) wich varies between 0 and 1
      k      = pin / pend
      kp     = 1 - k
      #---- extracting input grains 
      grain1 = DAFx_in1[pin:pin+s_win,:] * w1
      grain2 = DAFx_in2[pin:pin+s_win,:] * w1
   #===========================================
      #---- computing spectral shape of sound 1
      f1    = np.fft.fft(np.fft.fftshift(grain1,axes=0),axis=0)
      flog  = np.log(0.00001+abs(f1))
      cep   = np.fft.fft(flog,axis=0)
      cep_coupe   = np.vstack((cep[0,:]/2, cep[1:cut,:], np.zeros((s_win-cut,nChan))))
      flog_coupe1 = 2*np.real(np.fft.ifft(cep_coupe,axis=0))
      spec1 = np.exp(flog_coupe1)
      #---- computing spectral shape of sound 2
      f2    = np.fft.fft(np.fft.fftshift(grain2,axes=0),axis=0)
      flog  = np.log(0.00001+abs(f2))
      cep   = np.fft.fft(flog,axis=0)
      cep_coupe   = np.vstack((cep[0,:]/2, cep[1:cut,:], np.zeros((s_win-cut,nChan))))
      flog_coupe2 = 2*np.real(np.fft.ifft(cep_coupe,axis=0))
      spec2 = np.exp(flog_coupe2)
      #----- interpolating the spectral shapes in dBs
      spec  = np.exp(kp*flog_coupe1 + k*flog_coupe2)
      #----- computing the output spectrum and grain
      ft    = (kp*f1/spec1 + k*f2/spec2) *spec
      grain = np.fft.fftshift(np.real(np.fft.ifft(ft,axis=0)),axes=0)*w2
   #===========================================
      DAFx_out[pout:pout+s_win,:] = DAFx_out[pout:pout+s_win,:] + grain
      pin   = pin + n1
      pout  = pout + n2

   #----- saving the output -----
   # DAFx_in = DAFx_in1(s_win+1:s_win+L);
   DAFx_out = DAFx_out[s_win:s_win+L,:] / abs(DAFx_out).max()

   #return DAFx_out according to original signal shape
   if x1.ndim == 1:
      return DAFx_out[:,0]
   else:
      if x1.shape[1] == DAFx_out.shape[1]:
         return DAFx_out
      else:
         return DAFx_out.T

#Test
if __name__=='__main__':
   import matplotlib.pyplot as plt
   import audiofile as af
   from pytictoc import TicToc
   
   inputFile1 = 'audio\\claire_oubli_voix.wav' # Input file 1
   inputFile2 = 'audio\\claire_oubli_flute.wav' # Input file 2

   stdName1 = inputFile1.split('.wav')[0]
   stdName1 = stdName1.split('audio\\')[1]
   x1, fs1 = af.read(inputFile1)
   stdName2 = inputFile2.split('.wav')[0]
   stdName2 = stdName2.split('audio\\')[1]
   x2, fs2 = af.read(inputFile2)

   if fs1 != fs2:
      raise Exception('Mismatching sampling rates. The two audio files must have the same sampling rate.')
   else:
      fs = fs1

   n1       = 512;            
   n2       = n1;             
   s_win    = 2048;           
   cut      = 50

   t = TicToc()
   t.tic() #Start timer
   y = UX_spectral_interp(x1,x2,n1,n2)
   t.toc()
   af.write('audio\\spec_interp_'+stdName1+'_'+stdName2+'.wav',y, fs)