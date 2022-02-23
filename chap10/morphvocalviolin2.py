import audiofile as af
import numpy as np
import scipy.signal as sig
from hpsmodelmorph import hpsmodelmorph

x,fs  = af.read('soprano-E4.wav');
x2,fs = af.read('violin-B3.wav');
w=sig.windows.blackmanharris(1024)
w = np.insert(w,len(w),0)
dur = (len(x)-1)/fs;
f0intp = np.array([[0,dur],[0, 1]])
htintp = np.array([[0,dur],[0, 1]])
rintp = np.array([[0,dur],[0, 1]])
y,yh,ys = hpsmodelmorph(x,x2,fs,w,2048,-150,200,100,400,1500,1.5,10,f0intp,htintp,rintp);
af.write('soprano_violin2.wav',y,fs)
