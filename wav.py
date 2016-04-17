from scipy.io.wavfile import read, write
from scipy.fftpack import rfft, irfft
from numpy.fft import hfft, ihfft, fft, ifft
import numpy as np
import math
import wave


def best_length_fft(n):
    return int(int(n/2)*2)
    #return math.pow(2, max(factorint(n)))
def loadfft(wavfile):
    rate, input = read(wavfile)

    transformed_raw = np.array(input, dtype=np.float32)

    return {
            "raw": transformed_raw,
            "rate": rate
            }

def savefft(wavfile, wavobj, filtered):
    transformed_raw = wavobj['raw']
    rate = wavobj['rate']
    write(wavfile, rate, np.array(transformed_raw, dtype=np.int16))

def loadfft2(wav):
    return loadfft(wav)

def savefft2(wav, wavobj, filtered):
    return savefft(wav, wavobj, filtered)


# Returns the file object in complex64
def get_wav(path):

    wav = wave.open(path, 'rb')
    rate, data = read(path)
    results={}
    results['rate']=rate
    results['channels']=wav.getnchannels()
    results['sampwidth']=wav.getsampwidth()
    results['framerate']=wav.getframerate()
    results['nframes']=wav.getnframes()
    results['compname']=wav.getcompname()
    processed = np.array(data).astype(np.int16, copy=False)
    results['data']=processed
    return results

def save_wav(in_wav, path):

    print("Saving to ", path)
    wav = wave.open(path, 'wb')
    wav.setnchannels(in_wav['channels'])
    wav.setsampwidth(in_wav['sampwidth'])

    wav.setframerate(in_wav['framerate'])

    wav.setnframes(in_wav['nframes'])

    wav.setcomptype('NONE', 'processed')

    processed = np.array(in_wav['data'], dtype=np.int16)
    wav.writeframes(processed)


