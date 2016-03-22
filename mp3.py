
SAMPLES= 500
BANDS= 20
NUM_FREQS= 256

import pymedia.audio.acodec as acodec
import pymedia.audio.sound as sound

dec= acodec.Decoder( 'mp3' )
analyzer= sound.SpectrAnalyzer( 1, SAMPLES, NUM_FREQS )

# This doesn't work yet.  Still researching how to save a file after analysis 
# https://mail.python.org/pipermail/python-list/2004-August/259867.html
def loadmp3(mp3file):
    READ_SIZE=20000
    with open(mp3file, 'rb') as f:
        buf=f.read(READ_SIZE)
        while(len(s):
            buf=f.read(READ_SIZE)

def savemp3():
