import numpy as np

def peakinterp(mX, pX, ploc):
    '''
    Authors: J. Bonada, X. Serra, X. Amatriain, A. Loscos
    Parabolic interpolation of spectral peaks
    ---- Inputs ----
    mX: magnitude spectrum
    pX: phase spectrum
    ploc: locations of peaks
    --- Outputs ----
    iploc, ipmag, ipphase: interpolated values
    
     note that ploc values are assumed to be between 1 and length(mX)-2
    '''
    val = mX[ploc];                                    # magnitude of peak bin
    lval = mX[ploc-1];                                 # magnitude of bin at left
    rval= mX[ploc+1];                                  # magnitude of bin at right 
    iploc = ploc + 0.5*(lval-rval)/(lval-2*val+rval);  # center of parabola
    ipmag = val-.25*(lval-rval)*(iploc-ploc);         # magnitude of peaks
    ipphase = np.interp(iploc,np.arange(len(pX)),pX);  # phase of peaks 

    return iploc, ipmag, ipphase
