import cv2
import numpy as np
import os
from tqdm import tqdm
from cyvlfeat.kmeans import kmeans
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
train, train_digit_labels = get_images('./split_dataset/train/', 256)
#val, val_digit_labels = get_images('./split_dataset/val/', 256)
#test, test_digit_labels = get_images('./split_dataset/test/', 256)


# visual_words
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
    km = kmeans(bag_array, size, initialization="PLUSPLUS")   

    distances = np.zeros(len(bag_array))
    for i, point in enumerate(bag_array):
        # Calculate distance to each center and take minimum
        min_dist = np.min(np.sum((km - point) ** 2, axis=1))
        distances[i] = min_dist
            
    # Sum of squared distances is our distortion
    distortion = np.sum(distances)
    print(f"Completed the 200 k-means clustering with distortion: {distortion}")

    return km


features = sift_features(train, size=200)

print("Writing features to file...")
with open('sift_features.pkl', 'wb') as f:
    pickle.dump(features, f)

print("SIFT features saved to sift_features.pkl")