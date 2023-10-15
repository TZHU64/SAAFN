# -*- coding: utf-8 -*-
# This code was written by Philip Sellars if you use this code in your research
# please cite the associated paper. For any questions or queries please email me at 
# ps644@cam.ac.uk


# %% Modules to Import
import time
import numpy as np 
import data_analysis
import lcmr_functions as lcmr
import processing_data as pd
import classification_functions as cf
from sklearn.preprocessing import MinMaxScaler
import sklearn.metrics
import sklearn.manifold
import random
import HMS
import graphclass
import lcmr_cython
import matplotlib.pyplot as plt
import cli
 

args = cli.parse_commandline_args()
if args.dataset == "Indiana":
    args.k = 1200
    args.b = random.uniform(0.89,0.9) 
    args.sigma_l = 10**random.uniform(-0.8,-0.6)  
elif args.dataset == "Salinas":
    args.k = 1500
    args.b = random.uniform(0.89,0.9) 
    args.sigma_l = 10**random.uniform(0.8,1.3)  
elif args.dataset == "PaviaU":
    args.k=2400
    args.b = random.uniform(0.09,0.1) 
    args.sigma_l = 10**random.uniform(2.5,3.0)  
elif args.dataset == "KSC":
    args.k=2400
    args.b = random.uniform(0.09,0.1) 
    args.sigma_l = 10**random.uniform(2.5,3.0) 
    

#### Load Hyperspectral Data and Ground Truth
time_1 = time.time()
## Different names are Indian, Salinas and PaviaU
dataset = args.dataset
print("Using the",dataset,"dataset")
[spectral_original,ground_truth] = data_analysis.loading_hyperspectral(dataset)
no_class = np.max(ground_truth)
time_2 = time.time()
print("Time to load data", time_2 - time_1)
 
### Perform dimesnionality reduction using PCA. Extracts enough components to meet required variance.
time_1 = time.time() 
spectral_data_pca = data_analysis.principal_component_extraction(spectral_original,args.pca_v)
time_2 = time.time()
print("Time to perform PCA", time_2 - time_1)

### LCMR Matrix Construction
start = time.time()
spectral_mnf = lcmr.dimensional_reduction_mnf(spectral_original,args.cm_feats)
lcmr_matrices = lcmr.create_logm_matrices(spectral_mnf,args.cm_size,args.cm_nn)
final = time.time()
print("Time to produce covariance matrices",final-start)
      
##### HMS Over-segmentation
print("Time to over-segment using HMS", end =" ")
start = time.time()

hms = HMS.HMSProcessor(image = spectral_data_pca,lcmr_m = lcmr_matrices, k=args.k, m = 4,mc=True)
labels = hms.main_work_flow()
final = time.time()
print(final-start)


imgColor = spectral_original[:,:,[19 ,15 ,12]]
imgColor = imgColor.astype(np.double)
imgColor = 255*(imgColor - np.amin(imgColor)) / (np.amax(imgColor)-np.amin(imgColor))
imgColor = imgColor.astype(np.uint8)
cluster_averages = np.zeros((hms.ClusterNumber,3))
cluster_counts = np.zeros((hms.ClusterNumber))

for i in range(labels.shape[0]):
    for j in range(labels.shape[1]):
        cluster_averages[labels[i,j],:] += imgColor[i,j,:]
        cluster_counts[labels[i,j]] += 1

for i in range(hms.ClusterNumber):
    cluster_averages[i,:] = cluster_averages[i,:] / cluster_counts[i]

for i in range(labels.shape[0]):
    for j in range(labels.shape[1]):
        imgColor[i,j,:] = cluster_averages[labels[i,j],:]
        
# Saves the image and the superpixel representation
mean_image = np.copy(imgColor)
plt.imshow(mean_image)
plt.imsave("mean.png",mean_image)  

imgColor2 = spectral_original[:,:,[29 ,15 ,12]]
imgColor2 = 255*(imgColor2 - np.amin(imgColor2)) / (np.amax(imgColor2)-np.amin(imgColor2))
imgColor2 = imgColor2.astype(np.uint8)

test = hms.Clusters[:,:2]

for i in range(imgColor2.shape[0]):
    for j in range(imgColor2.shape[1]):
        
        for s in range(4):
            di = np.asarray([1,0,0,-1])
            dj = np.asarray([0,1,-1,0])
            
            s_i = i + di[s]
            s_j = j + dj[s]
            
            if(s_i < imgColor.shape[0] and s_j < imgColor.shape[1]):
                if(labels[s_i,s_j] != labels[i,j]):
                    imgColor2[i,j,:] = [255,0,0]
     
boundary_image = np.copy(imgColor2)
plt.imshow(boundary_image)
plt.imsave("boundary.png",boundary_image)  

maxedge1=0
maxedge2=0
for i in range(imgColor2.shape[0]):
    for j in range(imgColor2.shape[1]):
        if (imgColor2[i,j,[0]]!=255 and (i!=0 or j!=0)):
            ii=i
            jj=j
            edge1=0
            edge2=0
            while (imgColor2[ii,j,[0]]!=255 and ii < imgColor2.shape[0]-1):
                ii+=1
                edge1+=1
            if edge1>maxedge1:
                maxedge1=edge1
            while (imgColor2[i,jj,[0]]!=255 and jj < imgColor2.shape[1]-1):
                jj+=1
                edge2+=1
            if edge2>maxedge2:
                maxedge2=edge2
            
        
print ("Max. horizontal pixel",maxedge1)
print ("Max. vertical pixel",maxedge2)
