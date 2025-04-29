import cv2
import numpy as np
import os
from tqdm import tqdm
import time
from kmeans_gpu import KMeans  # Already imported
from cyvlfeat.sift.dsift import dsift
import pickle
import torch

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
train, train_digit_labels = get_images('./split_dataset/train/', 256)
#val, val_digit_labels = get_images('./split_dataset/val/', 256)
#test, test_digit_labels = get_images('./split_dataset/test/', 256)


# visual_words
import torch  # Import PyTorch

def sift_features(images, size):
    print("feature number", size)
    bag_of_features = []
    
    print("Extract SIFT features...")
    for key, value in tqdm(images.items()):
        for img in value:
            _, descriptors = dsift(img, step=[5,5], fast=True)
            if descriptors is not None:
                for des in descriptors:
                    bag_of_features.append(des)

    print("Compute kmeans in dimensions:", size)
    bag_array = np.array(bag_of_features).astype('float32')
    
    # Convert NumPy arrays to PyTorch tensors
    points = torch.tensor(bag_array[None, ...])  # shape: (1, num_pts, pts_dim)
    features = torch.tensor(bag_array[None, ...])  # shape: (1, num_pts, pts_dim)

    # Time the kmeans fitting
    start_time = time.time()
    kmeans = KMeans(n_clusters=size, max_iter=100)
    centroids, assignments = kmeans(points, features)
    elapsed = time.time() - start_time

    # Compute inertia (sum of squared distances to closest cluster center)
    assigned_centroids = centroids[0][assignments[0]]
    inertia = torch.sum((points[0] - assigned_centroids) ** 2).item()

    print(f"KMeans inertia: {inertia}")
    print(f"KMeans fitting time: {elapsed:.2f} seconds")

    return centroids[0].cpu().numpy()


features = sift_features(train, size=200)

print("Writing features to file...")
with open('sift_features.pkl', 'wb') as f:
    pickle.dump(features, f)

print("SIFT features saved to sift_features.pkl")