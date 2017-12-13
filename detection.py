import numpy as np
from PIL import Image
import os

import matplotlib.pyplot as plt
import matplotlib.patches  as patches
import sys
import util
import pickle


def plot_detection(truth, reconstructed, shape, indices, to_threshold,offsets, box_sizes,num_shown=10,threshold=0.5,filename="default.png"):
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
        plt.imshow(image.reshape(shape), cmap=plt.cm.gray)
        j = 0
        for y,x in indices[i]:
            if to_threshold[i][j] > threshold:
                temp.add_patch(patches.Rectangle((x-7+offsets[i][0],y-7+offsets[i][1]),box_sizes[i][0],box_sizes[i][1],linewidth=1,edgecolor='r',facecolor='none'))
            j+=1
    plt.savefig(filename,dpi=300)
    #plt.show()


#encoded[0],..., encoded[9] = first 10 images
encoded = pickle.load(open("faces.p","rb"))
train, labels = util.read_data_and_labels(3, 64)
# Selected deconv filters to use
deconv_filter_indices=[5,33,57,67,72,91,95,127]
# offsets of bounding boxes which depends on what each convolutional filter looks like
offsets=[(0,0),(0,-3),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)]
# box sizes, same goes with offsets
box_sizes=[(15,7),(15,9),(14,9),(11,10),(11,7),(12,8),(15,8),(15,10)]
# Threshold, whose value is determined by trying out differnt values and choosing one with best performance.
threshold=0.5
dirname = "detection"+str(threshold)
os.mkdir(dirname)
for t in range(10):
    a, b = [], []
    search_results, nonzero_entries = [], []
    for i in deconv_filter_indices:
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
    plot_detection(a,b,(64,64),search_results,nonzero_entries,offsets,box_sizes,len(offsets),threshold,filename=dirname+"/"+str(t)+".png")
