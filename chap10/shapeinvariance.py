# shape invariance: 
# assume pulse onsets to match zero phase of the fundamental
# and ideal harmonic distribution
#
#--------------------------------------------------------------------------
# This source code is provided without any warranties as published in 
# DAFX book 2nd edition, copyright Wiley & Sons 2011, available at 
# http://www.dafx.de. It may be used for educational purposes and not 
# for commercial applications without further permission.
#--------------------------------------------------------------------------

pos = np.mod(hphase[0],2*np.pi)/2/np.pi;         # input normalized period position
ypos = np.mod(yhphase[0],2*np.pi)/2/np.pi;       # output normalized period position
yhphase = hphase + (ypos-pos)*2*np.pi*np.linspace(1,len(yhloc),len(yhloc))