def tube(x, gain, Q, dist, rh, rl, mix):
    '''

    "Tube distortion" simulation, asymmetrical function

    Parameters
    ----------
    x    - input

    gain - the amount of distortion, >0->

    Q    - work point. Controls the linearity of the transfer
            function for low input levels, more negative=more linear

    dist - controls the distortion's character, a higher number gives 
            a harder distortion, >0

    rh   - abs(rh)<1, but close to 1. Placement of poles in the HP 
            filter which removes the DC component

    rl   - 0<rl<1. The pole placement in the LP filter used to 
            simulate capacitances in a tube amplifier

    mix  - mix of original and distorted sound, 1=only distorted

    Return
    ----------
    y - Tube-distorted signal
    
    '''

    import numpy as np
    from scipy import signal

    q = gain*x / np.max(np.abs(x))		        

    if Q == 0:
        z = q / (1-np.exp(-dist*q))
        for i in range(len(q)):                        
            if q[i] == Q:			                     
                z[i] = 1/dist			                     
    else:
        z = (q-Q) / (1-np.exp(-dist*(q-Q))) + Q / (1-np.exp(dist*Q))
        for i in range(len(q)):				        
            if q[i] == Q:					            
                z[i] = 1 / dist + Q / (1-np.exp(dist*Q))  

    y = mix * z * np.max(np.abs(x)) / np.max(np.abs(z)) + (1-mix) * x
    y = y * np.max(np.abs(x)) / np.max(np.abs(y))
    y = signal.lfilter([1, -2, 1],[1, -2*rh, rh**2],y)                                  #HP filter
    y = signal.lfilter([1-rl],[1, -rl],y)			                         #LP filter

    return y