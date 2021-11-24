import audiofile as af

#Parameters
input_file = 'audios/ton2.wav'       #input filename
output_file = 'audios/ton2_out.wav'  #output filename
a = 2
Nbits = 16

# Read input sound file into vector x(n) and sampling frequency FS
x, FS = af.read(input_file)

# Sample-by sample algorithm y(n)=a*x(n)
y = a * x

# Write y(n) into output sound file with number of
# bits Nbits and sampling frequency FS
af.write(output_file, y, FS, Nbits)
