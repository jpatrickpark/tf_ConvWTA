#import tensorflow as tf
import numpy as np
from PIL import Image
import os

import matplotlib.pyplot as plt
import matplotlib.patches  as patches
import sys
import util
import pickle


def plot_reconstruction(truth, reconstructed, shape, indices, to_threshold, num_shown=10):
    """Plots reconstructed images below the ground truth images."""
    plt.close()
    for i, image in enumerate(truth[:num_shown]):
        plt.subplot(2, num_shown, i + 1)
        plt.axis('off')
        image = np.clip(image,0,1)
        plt.imshow(image.reshape(shape), cmap=plt.cm.gray)
    for i, image in enumerate(reconstructed[:num_shown]):
        temp = plt.subplot(2, num_shown, i + num_shown + 1)
        plt.axis('off')
        if (len(shape)>2):
            image = (image-np.min(image))/(np.max(image)-np.min(image))
            #image = np.clip(image,0,1)
        plt.imshow(image.reshape(shape), cmap=plt.cm.gray)
        j = 0
        for y,x in indices[i]:
            if to_threshold[i][j] > 0.5:
                if i==5:
                    # this deconv layer has offset. I could have implemented it, but it is a quick fix.
                    temp.add_patch(patches.Rectangle((x,y-4),12,8,linewidth=1,edgecolor='r',facecolor='none'))
                else:
                    temp.add_patch(patches.Rectangle((x-7,y-7),12,8,linewidth=1,edgecolor='r',facecolor='none'))
            j+=1
    plt.show()


encoded = pickle.load(open("faces.p","rb"))
train, labels = util.read_data_and_labels(3, 64)
#encoded[0],..., encoded[9] = first 10 images
box_sizes=[(15,7),(15,6),(14,9),(11,10),(11,7),(8,6),(12,8),(15,8),(15,10)]
for t in range(10):
    a, b = [], []
    search_results, nonzero_entries = [], []
    for i in [5,33,57,67,72,79,91,95,127]:
        sparse = encoded[t][0,:,:,i]
        search_result = []
        values = []
        for j in range(64):
            for k in range(64):
                if sparse[j,k] != 0:
                    search_result.append((j,k))
                    values.append(sparse[j,k])
        print(search_result)
        print(values)
        a.append(sparse)
        image = train[t,:]
        b.append(image)
        search_results.append(search_result)
        nonzero_entries.append(values)
    plot_reconstruction(a,b,(64,64),search_results,nonzero_entries)



