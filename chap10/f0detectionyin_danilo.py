import numpy as np

def f0detectionyin(x,fs,ws,minf0,maxf0):
    '''
    Authors: J. Bonada, X. Serra, X. Amatriain, A. Loscos
    ==> fundamental frequency detection function
    ---Inputs---
    x: input signal
    fs: sampling rate
    ws: integration window length
    minf0: minimum f0
    maxf0: maximum f0
    th=0.1: threshold
    ---Outputs---
    f0: fundamental frequency detected in Hz

    '''
    maxlag = ws-2;                             # maximum lag
    d = np.zeros(maxlag);                      # init variable
    d2 = np.zeros(maxlag);                     # init variable
    th=0.1                                     # compute threshold
    # compute d(tau)
    x1 = x[:ws];
    cumsumx = (x1**2).sum();
    cumsumxl = cumsumx
    xy = np.correlate(x[:ws*2],np.pad(x1,(0,ws)),mode='full');
    xy = xy[ws*2:ws*3-2];
    
    for lag in range(maxlag):                  #for lag=0:maxlag-1
        d[lag] = cumsumx + cumsumxl - 2*xy[lag];
        cumsumxl = cumsumxl - x[lag]**2 + x[lag+ws+1]**2;
    
    cumsum = 0;
    # compute d'(tau)
    d2[0] = 1;
    for lag in range(1,maxlag,1):               #for lag=1:maxlag-1
        cumsum = cumsum + d[lag];
        d2[lag] = d[lag]*lag/cumsum;
    
    # limit the search to the target range
    minf0lag = 1+round(fs/minf0);       # compute lag corresponding to minf0
    maxf0lag = 1+round(fs/maxf0);       # compute lag corresponding to maxf0
    if (maxf0lag>1 and maxf0lag<maxlag):
        d2[:maxf0lag] = 100;                 # avoid lags shorter than maxf0lag
    
    if (minf0lag>1 and minf0lag<maxlag):
        d2[minf0lag-1:] = 100;               # avoid lags larger than minf0lag
    
    # find the best candidate
    #mloc = 1 + find((d2(2:end-1)<d2(3:end)).*(d2(2:end-1)<d2(1:end-2)));  % minima
    auxmloc = np.where(d2[1:-1]<d2[2:],1,0) * np.where(d2[1:-1]<d2[:-2],1,0)
    mloc = 1 + np.where(auxmloc>0)[0]
    candf0lag = 0;
    if (len(mloc)>0):
        I = np.where(d2[mloc]<th)[0];
        if (len(I)>0):
            candf0lag = mloc[I[0]];
        else:
            #[Y,I2] = min(d2(mloc));
            Y = d2[mloc].min()
            I2 = np.where(d2[mloc]==Y)[0][0]
            candf0lag = mloc[I2];
        
        candf0lag = candf0lag;                # this is zero-based indexing
        
        if (candf0lag>0 and candf0lag<maxlag-1):
            # parabolic interpolation
            lval = d2[candf0lag-1]; 
            val = d2[candf0lag]; 
            rval= d2[candf0lag+1]; 
            candf0lag = candf0lag + 0.5*(lval-rval)/(lval-2*val+rval);  
                
    ac = d2.min(); 
    f0lag = candf0lag;                        # convert to zero-based indexing
    f0 = fs/f0lag;                            # compute candidate frequency in Hz
    if (ac > 0.2):                            # voiced/unvoiced threshold
        f0 = 0;                               # set to unvoiced
    
    return f0

if __name__ == '__main__':
    import audiofile as af
    import matplotlib.pyplot as plt
    
    x,fs = af.read('audios2\\violin-B3.wav')
    w = np.hanning(1026)
    w = w[:-1].copy()
    M = len(w);                           # analysis window size (odd)
    H = 256;                              # hop size for analysis and synthesis
    hM = (M-1)//2;                        # half analysis window size
    pin = max(H,hM);        #initialize sound pointer to middle of analysis window
    pend = len(x)-max(hM,H)-1;            # last sample to start a frame
    minf0 = 100
    maxf0 = 400

    saidaF0=[]
    saidapin=[]
    while pin<pend:
        #----callyin.m-----
        yinws = round(fs*0.015); # using approx. a 15 ms window for yin
        yinws = yinws + yinws%2; # make it even
        yb = pin-yinws//2;
        ye = pin+yinws//2+yinws+1;
        if (yb<0 or ye>len(x)): # out of boundaries
            f0 = 0;
        else:
            f0 = f0detectionyin(x[yb:ye],fs,yinws,minf0,maxf0);
        saidapin.append(pin)
        saidaF0.append(f0)
        pin+=H
    
    f0B3 = 246.94
    plt.figure()
    plt.plot(saidapin,saidaF0,label=r'$F_0$ (measured)')
    plt.plot([saidapin[0],saidapin[-1]],[f0B3,f0B3],'k--',label='B3')
    plt.title('$F_0$ detection - Yin ("violin-B3.wav")')
    plt.xlabel(r'n $\rightarrow$')
    plt.ylabel(r'Frequency [HZ]')
    plt.axis(ymin=230)
    plt.legend()
    plt.show()
