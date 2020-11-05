# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 20:36:28 2018

@author: Gokul
"""

import matplotlib.pyplot as plt
import numpy as np
import wave
import sys


spf = wave.open('../Dataset/nsynth-test/audio/bass_electronic_018-024-100.wav','r')

#Extract Raw Audio from Wav File
signal = spf.readframes(-1)
signal = np.fromstring(signal, 'Int16')
for index in range(100):
    print(signal[index])

#If Stereo
if spf.getnchannels() == 2:
    print("Just mono files")
    sys.exit(0)

plt.figure(1)
plt.title('Signal Wave...')
plt.plot(signal)
plt.show()