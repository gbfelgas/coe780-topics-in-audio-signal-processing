import numpy as np
import scipy.signal as sig

def noisegt(x,holdtime,ltrhold,utrhold,release,attack,a,Fs,output='y'):
    '''
    % function y=noisegt(x,holdtime,ltrhold,utrhold,release,attack,a,Fs)
    % Author: R. Bendiksen

    % noise gate with hysteresis
    % holdtime	- time in seconds the sound level has to be below the 
    %		      threshhold value before the gate is activated
    % ltrhold	- threshold value for activating the gate
    % utrhold	- threshold value for deactivating the gate > ltrhold
    % release	- time in seconds before the sound level reaches zero
    % attack	- time in seconds before the output sound level is the 
    %		      same as the input level after deactivating the gate
    % a		- pole placement of the envelope detecting filter <1
    % Fs 	- sampling frequency
    % output    - 'y' for processed signal
    %           - 'yg' for processed signal and applied gain
    %           - 'ye' for processed signal and envelope
    %           - 'yge' for processed signal, gain and envelope
    %
    %--------------------------------------------------------------------------
    % This source code is provided without any warranties as published in 
    % DAFX book 2nd edition, copyright Wiley & Sons 2011, available at 
    % http://www.dafx.de. It may be used for educational purposes and not 
    % for commercial applications without further permission.
    %--------------------------------------------------------------------------
    '''
    
    #Adapting x shape to (sample, channel)
    if x.ndim == 1:
        x_adj = x.reshape((x.shape[0],1))
    elif x.ndim == 2:
        if x.shape[0]>x.shape[1]:
            x_adj = x.copy()
        else:
            x_adj = x.T.copy()
    else:
        raise TypeError('unknown audio data format !!!')
        return
    nChan = x_adj.shape[1]
    rel = round(release*Fs);  #number of samples for fade (release)
    att = round(attack*Fs);   #number of samples for fade (attack)
    ht  = round(holdtime*Fs);

    #envelope detection filter
    coefs_b = [(1-a)**2]
    coefs_a = [1.0000, -2*a, a**2]
    h = sig.lfilter(coefs_b,coefs_a,abs(x_adj),axis = 0)
    h = h/h.max(axis=0);
    g = np.zeros(x_adj.shape)

    for chan in range(nChan):
        lthcnt = 0          # counter for samples below the lower threshold
        uthcnt = 0          # counter for samples above the upper threshold
        for i in range(h.shape[0]):
            if (h[i,chan]<=ltrhold) or ((h[i,chan]<utrhold) and (lthcnt>0)): #gate activation
                # Value below the lower threshold?
                lthcnt = lthcnt+1
                uthcnt = 0
                if lthcnt>ht:			
                    # Time below the lower threshold longer than the hold time?
                    if lthcnt>(rel+ht):
                        g[i,chan] = 0;
                    elif ((i+1<=ht+rel) and (lthcnt==i+1)):
                        g[i,chan] = 0;       #keeps gain to 0 if signal starts below lower threshold
                    else:
                        g[i,chan] = 1 - (lthcnt-ht)/rel;   #fades the signal to zero
                elif ((i+1<=ht) and (lthcnt==i+1)):  
                    g[i,chan]=0;             #keeps gain to 0 if signal starts below lower threshold
                else:
                    g[i,chan]=1;
            
            elif (h[i,chan]>= utrhold) or ((h[i,chan]>ltrhold) and (uthcnt>0)):	
                # Value above the upper threshold or is the signal being faded in?
                uthcnt = uthcnt+1;
                lthcnt=0;
                if i>0: #avoid error starting above upper threshold
                    if (g[i-1,chan]<1):				
                        # Has the gate been activated or isn't the signal faded in yet?
                        g[i,chan]=max(uthcnt/att,g[i-1,chan]);
                    else:
                        g[i,chan]=1;
                else:
                    g[i,chan]=1;
            else:
                if i>0:
                    g[i,chan]=g[i-1,chan];
                else:
                    g[i,chan]=1
                lthcnt=0;
                uthcnt=0;
                
    y = x_adj*g;
    y = y*(abs(x_adj).max(axis=0)/(abs(y).max(axis=0)))

    #return y according to original x shape
    if x.ndim == 1:
        if output=='yg':
            return y[:,0], g[:,0]
        elif output=='yge':
            return y[:,0], g[:,0], h[:,0]
        elif output=='ye':
            return y[:,0], h[:,0]
        else:
            return y[:,0]
    else:
        if x.shape[0] == x_adj.shape[0]:
            if output=='yg':
                return y, g
            elif output=='yge':
                return y, g, h
            elif output=='ye':
                return y, h
            else:
                return y
        else:
            if output=='yg':
                return y.T, g.T
            elif output=='yge':
                return y.T, g.T, h.T
            elif output=='ye':
                return y.T, h.T
            else:
                return y.T


