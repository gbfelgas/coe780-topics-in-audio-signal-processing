from hashlib import new
import numpy as np
import soundfile as sf
from librosa import load

def discrete_cepstrum_basic(F, A, order):
    #---- initialize matrices and vectors
    L = len(A)
    M = np.zeros((L, order + 1))
    R = np.zeros(order + 1)
    W = np.zeros((L, L))
    for i in range(L):
        M[i, 1] = 0.5
        for k in range(1, order + 1):
            M[i, k] = np.cos(2 * np.pi * (k - 1) * F[i])
        W[i, i] = 1 # weights = 1 by default
    M = 2 * M

    #---- compute the solution, regardless of matrix conditioning
    Mt = M.T
    MtWMR = Mt @ W @ M
    cep = np.linalg.pinv(MtWMR) @ Mt @ W @ np.log(A)
    return cep

def discrete_cepstrum_reg(F, A, order, laambda):

  #---- reject incorrect lambda values
  if (laambda >= 1 or laambda < 0): 
    raise Exception('Lambda must be in [0,1[')

  #---- initialize matrices and vectors
  L = len(A)
  M = np.zeros((L,order+1))
  R = np.zeros((order+1,1))

  for i in range(L) :
    M[i,0] = 1
    for k in range(order):
      M[i,k+1] = 2 * np.cos(2*np.pi*(k+1)*F[i])

  #---- initialize the R vector values
  coef = 8*(np.pi**2)
  for k in range(order+1) :
    R[k,0] = coef * k**2

  #---- compute the solution
  Mt = np.transpose(M)
  MtMR = np.matmul(Mt,M) + (laambda/(1-laambda))*np.diag(R)
  cep = np.matmul(np.linalg.inv(MtMR) , np.matmul(Mt,np.log10(A)))

  return cep

def discrete_cepstrum_random(F, A, order, laambda, n_rand_points, dev):

  #---- reject incorrect lambda values
  if (laambda >= 1 or laambda < 0): 
    raise Exception('Lambda must be in [0,1[')

  #---- generate random points
  L = len(A)
  new_A = np.zeros(L*n_rand_points)
  new_F = np.zeros(L*n_rand_points)
  for k in range(L):
    sigA = dev * A[k]
    sigF = dev * F[k]
    for l in range(L):
      new_A[l*n_rand_points] = A[l]
      new_F[l*n_rand_points] = F[l]
      for n in range(1,n_rand_points):
        new_A[l*n_rand_points + n] = A[l] + np.random.randn() * sigA
        new_F[l*n_rand_points + n] = F[l] + np.random.randn() * sigF 

  #---- initialize matrices and vectors
  L = len(new_A)
  M = np.zeros((L,order+1))
  R = np.zeros((order+1,1))

  for i in range(L) :
    M[i,0] = 1
    for k in range(order):
      M[i,k+1] = 2 * np.cos(2*np.pi*(k+1)*new_F[i])

  #---- initialize the R vector values
  coef = 8*(np.pi**2)
  for k in range(order+1) :
    R[k,0] = coef * k**2

  #---- compute the solution
  Mt = np.transpose(M)
  MtMR = np.matmul(Mt,M) + (laambda/(1-laambda))*np.diag(R)
  cep = np.matmul(np.linalg.inv(MtMR) , np.matmul(Mt,np.log10(new_A)))

  return cep

if __name__ == '__main__':
  x, fs = sf.read('/Users/bernardo/Desktop/COE780/chap08/la.wav')

  w = np.hanning(2049)
  N = 2048
  H = 512
  t = -50
  c = np.zeros(10)

  M = w.size # analysis window size
  N2 = N // 2 + 1 # size of positive spectrum
  soundlength = x.size # length of input sound array
  hM = int((M - 1) / 2) # half analysis window size
  pin = hM # initialize sound pointer in middle of analysis window
  pend = soundlength - hM # last sample to start a frame
  fftbuffer = np.zeros(N, np.complex) # initialize buffer for FFT
  yw = np.zeros(M) # initialize output sound frame
  y = np.zeros(soundlength) # initialize output array
  w /= w.sum() # normalize analysis window
  sw = np.hanning(M) # synthesis window
  sw /= sw.sum()
  while pin < pend:
      #-----analysis-----
      xw = x[pin - hM : pin + hM + 1] * w # window the input sound
      fftbuffer[:] = 0 # reset buffer
      fftbuffer[0 : int((M + 1) / 2)] = xw[int((M - 1) / 2) : M] # zero-phase window in fftbuffer
      fftbuffer[N - (M - 1) // 2 : N] = xw[0 : (M - 1) // 2]
      X = np.fft.fft(fftbuffer)
      mX = 20 * np.log10(np.finfo(float).eps + np.abs(X[0:N2]))
      pX = np.unwrap(np.angle(X[0:N2])) # unwrapped phase spect. of positive freq.
      ploc = np.argwhere((mX[1 : N2 - 1] > t).astype(int) * (mX[1:N2-1] > mX[2:N2]).astype(int) * (mX[1:N2-1] > mX[0:N2-2]).astype(int)) # peaks
      pmag = mX[ploc] # magnitude of peaks
      pphase = pX[ploc] # phase of peaks
      c = discrete_cepstrum_basic(ploc.T[0] * fs/N, 10 ** (pmag.T[0]/20), 10)
      # c = discrete_cepstrum_reg(ploc.T[0] * fs/N, 10 ** (pmag.T[0]/20), 10, 0.5)
      # c = discrete_cepstrum_random(ploc.T[0] * fs/N, 10 ** (pmag.T[0]/20), 10, 0.7, 3, 0.1)
      print(c)
      #-----synthesis-----
      Y = np.zeros(N, np.complex) # initialize output spectrum
      Y[ploc] = 10 ** (pmag / 20) * np.exp(1j * pphase) # generate positive freq.
      Y[N - 1 - ploc] = 10 ** (pmag / 20) * np.exp(-1j * pphase) # generate negative freq.
      # generate neg.freq.
      fftbuffer = np.real(np.fft.ifft(Y)) # inverse FFT
      yw[0 : (M - 1) // 2] = fftbuffer[N - (M - 1) // 2 : N] # undo zero-phase window
      yw[(M - 1) // 2 : M] = fftbuffer[0 : (M + 1) // 2]
      y[pin - hM : pin + hM + 1] = y[pin - hM : pin + hM + 1] + H * N * sw * yw[0:M] # overlap-add
      pin = pin + H # advance sound pointer
  sf.write('/Users/bernardo/Desktop/COE780/chap08/teste.wav', y, fs)
