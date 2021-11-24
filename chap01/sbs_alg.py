# Author: U. Zölzer (Matlab)
#
#--------------------------------------------------------------------------
# This source code is provided without any warranties as published in 
# DAFX book 2nd edition, copyright Wiley & Sons 2011, available at 
# http://www.dafx.de. It may be used for educational purposes and not 
# for commercial applications without further permission.
#--------------------------------------------------------------------------
import audiofile as af
import numpy as np

#Parameters
inputFile = 'ton2.wav'       #input filename
outputFile = 'ton2_out.wav'  #output filename
a = 2
Nbits = 16

# Read input sound file into vector x(n) and sampling frequency FS
x, FS = af.read(inputFile)

# Sample-by sample algorithm y(n)=a*x(n)
y = a * x 

# Write y(n) into output sound file with number of 
# bits Nbits and sampling frequency FS
af.write(outputFile,y,FS,Nbits)
