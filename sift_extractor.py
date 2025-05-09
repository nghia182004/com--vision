import cv2
import numpy as np
import os
from tqdm import tqdm
import time
import faiss                                      
from faiss import StandardGpuResources, GpuIndexFlatL2, Clustering
from cyvlfeat.sift.dsift import dsift
import pickle

def get_images(path, size):
    total_pic = {}
    labels = []
    for i, doc in enumerate(os.listdir(path)):
        tmp = []
        for file in os.listdir(os.path.join(path, doc)):      
            
            if file.endswith(".jpg") or file.endswith(".png"):
                img = cv2.imread(os.path.join(path, doc, file), cv2.IMREAD_GRAYSCALE)
                pic = cv2.resize(img, (size, size))
                tmp.append(pic)
                labels.append(i)
        total_pic[doc] = tmp
    return total_pic, labels

# get images with resize
train, train_digit_labels = get_images('/split_dataset/train/', 256)
#val, val_digit_labels = get_images('./split_dataset/val/', 256)
#test, test_digit_labels = get_images('./split_dataset/test/', 256)


# visual_words
def sift_features_faiss(images, n_clusters, gpu_id=0):
    # 1) Extract and stack descriptors
    bag = []
    for _, imgs in tqdm(images.items()):
        for img in imgs:
            _, desc = dsift(img, step=[5,5], fast=True)
            if desc is not None:
                bag.append(desc)
    if not bag:
        return np.zeros((0,128), dtype='float32')
    xb = np.vstack(bag).astype('float32')
    
    # 2) Set up GPU resources & index
    res = StandardGpuResources()                     
    cfg = faiss.GpuIndexFlatConfig(); cfg.device = gpu_id
    index = GpuIndexFlatL2(res, xb.shape[1], cfg)    
    
    # 3) Clustering parameters
    cp = faiss.Clustering(xb.shape[1], n_clusters)
    cp.niter = 100
    cp.max_points_per_centroid = 100000
    cp.verbose = True

    # 4) Train
    start = time.time()
    cp.train(xb, index)                             
    centroids = faiss.vector_to_array(cp.centroids).reshape(n_clusters, xb.shape[1])
    elapsed = time.time() - start

    print(f"FAISS KMeans fit time: {elapsed:.2f}s on GPU")
    return centroids



features = sift_features_faiss(train, 200)

print("Writing features to file...")
with open('/content/com--vision/sift_features.pkl', 'wb') as f:
    pickle.dump(features, f)

print("SIFT features saved to sift_features.pkl")