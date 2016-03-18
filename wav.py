from scipy.io.wavfile import read, write
from scipy.fftpack import rfft, irfft
from numpy.fft import hfft, ihfft, fft, ifft
import numpy as np

def loadfft(wavfile):
    rate, input = read('input.wav')

    rfftx = rfft(input)
    transformed_raw = np.array(rfftx, dtype=np.float32)
    transformed = transformed_raw / transformed_raw.max(axis=0)

    return {
            "transformed": transformed,
            "raw": transformed_raw,
            "rate": rate
            }

def sanity(wavobj):
    transformed = wavobj['transformed']
    transformed_raw = wavobj['raw']
    rate = wavobj['rate']
    output = irfft(transformed)
    san_output = irfft(transformed)* transformed_raw.max(axis=0)

    write('sanity.wav', rate, np.array(san_output, dtype=np.int16))
    print("Sanity.wav written")

def savefft(wavfile, wavobj, filtered):
    transformed = wavobj['transformed']
    transformed_raw = wavobj['raw']
    rate = wavobj['rate']
    data = filtered * transformed_raw.max(axis=0)
    output = irfft(data)
    write(wavfile, rate, np.array(output, dtype=np.int16))




