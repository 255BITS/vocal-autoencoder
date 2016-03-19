import os
import sys
import glob

def do(command):
    print("Running " + command)
    print(os.system(command))

i = 0
if(len(sys.argv) > 1):
    do("cd training/to_process && scdl  -l "+sys.argv[1])

    for file in glob.glob('training/to_process/**/*.mp3'):
        wav_out = "training/processed/wav"+str(i)+".wav"
        do("ffmpeg -i \""+file+"\" -ac 1 "+wav_out)
        #do("rm \""+file+"\"")
        i+=1

else:
    print("Usage: " + sys.argv[0]+" [link to soundcloud playlist]")
