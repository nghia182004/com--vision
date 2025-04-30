import cv2, os, pickle
import numpy as np
import faiss
from tqdm import tqdm
from cyvlfeat.sift.dsift import dsift

# --- 3.1 Load images (already in your code) ---
def get_images(path, size):
    total, labels = {}, []
    for i, cls in enumerate(os.listdir(path)):
        imgs = []
        for file in os.listdir(os.path.join(path, cls)):
            if file.lower().endswith(('.jpg','.png')):
                img = cv2.imread(os.path.join(path, cls, file), cv2.IMREAD_GRAYSCALE)
                imgs.append(cv2.resize(img, (size, size)))
                labels.append(i)
        total[cls] = imgs
    return total, labels

train, train_digit_labels = get_images('/content/com--vision/split_dataset/train/', 256)
val, val_digit_labels = get_images('/content/com--vision/split_dataset/val/', 256)
test, test_digit_labels = get_images('/content/com--vision/split_dataset/test/', 256)

# --- 3.2 Load centroids & build FAISS index ---
with open('/content/com--vision/sift_features.pkl', 'rb') as f:
    centroids = pickle.load(f).astype('float32')

K, dim = centroids.shape
index = faiss.IndexFlatL2(dim)
index.add(centroids)

# --- 3.3 Encode all training images ---
def build_histogram(imgs):
    histograms = []
    for cls, imgs in tqdm(imgs.items(), desc="Encoding images..."):
        for img in imgs:
            _, desc = dsift(img, step=[5,5], fast=True)
            if desc is not None and len(desc)>0:
                _, assign = index.search(desc.astype('float32'), 1)
                hist, _ = np.histogram(assign, bins=np.arange(K+1))
            else:
                hist = np.zeros(K, dtype='int32')
            # L2-normalize
            hist = hist.astype('float32')
            n = np.linalg.norm(hist)
            histograms.append(hist / n if n>0 else hist)
    histograms = np.vstack(histograms)  # shape: (num_images, K)
    return histograms

train_histograms = build_histogram(train)
val_histograms = build_histogram(val)
test_histograms = build_histogram(test)

# 3.4 Save your final BoVW features
with open('bovw_train_histograms.pkl', 'wb') as f:
    pickle.dump(train_histograms, f)

with open('bovw_val_histograms.pkl', 'wb') as f:
    pickle.dump(val_histograms, f)

with open('bovw_test_histograms.pkl', 'wb') as f:
    pickle.dump(test_histograms, f)
print("BoVW codebook histograms saved!")

with open('train_labels', 'wb') as f:
    pickle.dump(train_digit_labels, f)

with open('val_labels', 'wb') as f:
    pickle.dump(val_digit_labels, f)

with open('test_labels', 'wb') as f:
    pickle.dump(test_digit_labels, f)
print("Labels saved!")

