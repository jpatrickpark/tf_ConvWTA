#import tensorflow as tf
import numpy as np
from PIL import Image
import os


import sys
import util
import pickle

encoded = pickle.load(open("faces.p","rb"))
#print(encoded)
a, b = [], []
for i in range(10):
    a.append(encoded[0][0,:,:,i])
util.plot_reconstruction(a,a,(64,64))
