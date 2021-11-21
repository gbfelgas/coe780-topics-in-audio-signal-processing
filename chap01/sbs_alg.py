# Author: U. Zölzer (Matlab)
#
#--------------------------------------------------------------------------
# This source code is provided without any warranties as published in 
# DAFX book 2nd edition, copyright Wiley & Sons 2011, available at 
# http://www.dafx.de. It may be used for educational purposes and not 
# for commercial applications without further permission.
#--------------------------------------------------------------------------
import soundfile as sf
import numpy as np

#Parameters
inputFile = 'ton2.wav'
outputFile = 'outton2.wav'
a = 2
Nbits = 16

# Read input sound file into vector x(n) and sampling frequency FS
x, FS = sf.read(inputFile)

# Sample-by sample algorithm y(n)=a*x(n)
y = a * x 

# Write y(n) into output sound file with number of 
# bits Nbits and sampling frequency FS
sf.write(outputFile,y,FS,subtype='PCM_'+str(Nbits))
