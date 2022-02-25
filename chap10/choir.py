import numpy as np
import scipy.signal as sig
import audiofile as af
from hpsmodeltranspositionenv import hpsmodelparams

x, fs = af.read('audios3\\tedeum.wav');
w = np.insert(sig.windows.blackmanharris(2048),2048,0);

men = 7;
women = 8;
N = 2048;
t = -150;
nH = 200;
minf0 = 100; #minimum f0 frequency in Hz, 
maxf0 = 400; #maximim f0 frequency in Hz, 
f0et = 1;
maxhd = 0.2;
stocf = 10;

dur = (len(x)-1)/fs;
fn = int(np.ceil(dur/0.2));
fscale = np.zeros((2,fn));
fscale[0,:] = np.arange(fn)/(fn-1)*dur;

tn = int(np.ceil(dur/0.5));
timemapping = np.zeros((2,tn));
timemapping[0,:] = np.arange(tn)/(tn-1)*dur;
timemapping[1,:] = timemapping[0,:];
ysum = np.vstack((x,x)) # make it stereo
if men==0: ysum = ysum*0

for i in range(men):
    print('Processing men:',str(i+1))
    fscale[1,:] = 2**(np.random.normal(0,1,fn)*30/1200);
    timemapping[1,1:-1] = timemapping[0,1:-1] + np.random.normal(0,1,tn-2)*len(x)/fs/tn/6
    auxtimbre = np.arange(1000,fs/2-1000+1,1000)
    timbremapping = np.array([np.insert(auxtimbre,(0,len(auxtimbre)),(0,fs/2)), \
        np.insert(auxtimbre*(1+0.1*np.random.normal(0,1,len(auxtimbre))),(0,len(auxtimbre)),(0,fs/2))])
    y, yh, ys= hpsmodelparams(x,fs,w,N,t,nH,minf0,maxf0,f0et,maxhd,stocf,timemapping=timemapping,fscale=fscale,timbremapping=timbremapping);

    pan = max(-1,min(1,np.random.normal(0,1)/3)) # [0,1]
    l = np.cos(pan*np.pi/2); #%.^2;
    r = np.sin(pan*np.pi/2); #%1-l;
    difsize = len(y)-ysum.shape[1]
    if difsize>0:
        ysum = np.concatenate((ysum,np.zeros((2,difsize))),axis=1)
    ysum[:,:len(y)] += np.vstack((l*y,r*y));


for i in range(women):
    print('Processing women:',str(i+1))
    fscale[1,:] = 2**(1+np.random.normal(0,1,fn)*30/1200);
    auxtimbre1 = np.arange(1000,fs/2-1000+1,1000)
    auxtimbre2 = np.arange(1000*5/4,5001,1000*5/4)
    auxdif = len(auxtimbre1) - len(auxtimbre2)
    auxtimbre3 = 5000+ np.arange(1,auxdif+1,1)/auxdif * (auxtimbre1[-1]-auxtimbre2[-1])
    auxtimbre2 = np.hstack((auxtimbre2,auxtimbre3))
    timbremapping = np.array([np.insert(auxtimbre1,(0,len(auxtimbre1)),(0,fs/2)), \
        np.insert(auxtimbre2*(1+0.1*np.random.normal(0,1,len(auxtimbre2))),(0,len(auxtimbre2)),(0,fs/2))])
    y, yh, ys= hpsmodelparams(x,fs,w,N,t,nH,minf0,maxf0,f0et,maxhd,stocf,timemapping=timemapping,fscale=fscale,timbremapping=timbremapping);
    pan = max(-1,min(1,np.random.normal(0,1)/3)) # [0,1]
    l = np.cos(pan*np.pi/2); #%.^2;
    r = np.sin(pan*np.pi/2); #%1-l;
    difsize = len(y)-ysum.shape[1]
    if difsize>0:
        ysum = np.concatenate((ysum,np.zeros((2,difsize))),axis=1)
    ysum[:,:len(y)] += np.vstack((l*y,r*y));
    


ysum = ysum/abs(ysum).max()*abs(x).max()

af.write('audios3\\tedeum_choir_misto.wav',ysum,fs)
